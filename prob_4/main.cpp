#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <pthread.h>

#include "timer.h"
#include "sse_mathfun.h"
#include "stream.hpp"
#include "vectsimd_sse.hpp"
#include "platform.hpp"
#include "prim_rgb_view.hpp"
#include "array.hpp"
#include "problem_7.hpp"

// verify iostream-free status
#if _GLIBCXX_IOSTREAM
#error rogue iostream acquired
#endif

namespace stream {

// deferred initialization by main()
in cin;
out cout;
out cerr;

} // namespace stream

static const char arg_prefix[]		= "-";
static const char arg_screen[]		= "screen";
static const char arg_bitness[]		= "bitness";
static const char arg_fsaa[]		= "fsaa";
static const char arg_nframes[]		= "frames";
static const char arg_seed[]		= "seed";
static const char arg_smooth[]		= "smooth";

static const size_t nthreads = WORKFORCE_NUM_THREADS;
static const size_t one_less = nthreads - 1;
static const size_t ao_probe_count = AO_NUM_RAYS;

static const compile_assert< ao_probe_count % 4 == 0 > assert_ao_probe_count_even;
static unsigned shape_seed = 42;
static float offset = .0625f;

enum
{
	BARRIER_START,
	BARRIER_FINISH,
	BARRIER_COUNT
};

static pthread_barrier_t barrier[BARRIER_COUNT];

struct __attribute__ ((aligned(128))) compute_arg
{
	uint32_t id;
	uint32_t frame;

	const Timeslice* tree;

	uint8_t (* framebuffer)[4];
	uint16_t w;
	uint16_t h;

	uint32_t seed;

	HitInfo hit;

	simd::vect3 cam[4];

	compute_arg()
	: id(0)
	, frame(0)
	, tree(0)
	, framebuffer(0)
	, w(0)
	, h(0)
	, seed(0)
	{
	}

	compute_arg(
		const size_t arg_id,
		uint8_t (* const arg_framebuffer)[4],
		const unsigned arg_w,
		const unsigned arg_h)
	: id(arg_id)
	, frame(0)
	, tree(0)
	, framebuffer(arg_framebuffer)
	, w(arg_w)
	, h(arg_h)
	, seed(rand())
	{
	}

	compute_arg(
		const size_t arg_id,
		const size_t arg_frame,
		const simd::vect3 (& arg_cam)[4],
		const Timeslice& arg_tree,
		uint8_t (* const arg_framebuffer)[4],
		const unsigned arg_w,
		const unsigned arg_h)
	: id(arg_id)
	, frame(arg_frame)
	, tree(&arg_tree)
	, framebuffer(arg_framebuffer)
	, w(arg_w)
	, h(arg_h)
	, seed(rand())
	{
		cam[0] = arg_cam[0];
		cam[1] = arg_cam[1];
		cam[2] = arg_cam[2];
		cam[3] = arg_cam[3];
	}
};


static void
shade(
	const Timeslice& ts,
	const Ray& ray,
	HitInfo& hit,
	unsigned& seed,
	uint8_t (& pixel)[4])
{
	hit.target = PayloadId(-1);

	if (!ts.traverse(ray, hit))
	{
		pixel[0] = 0;
		pixel[1] = 0;
		pixel[2] = 0;
		pixel[3] = 0;
		return;
	}

#if DRAW_TREE_CELLS == 1
	const uint8_t color_r[8] = { 255, 127,  63,  63,   0,   0,   0,   0 };
	const uint8_t color_g[8] = {   0,  63, 127, 255, 255, 127,  63,   0 };
	const uint8_t color_b[8] = {   0,   0,   0,   0,  63,  63, 127, 255 };

	assert(8 > hit.target);

	pixel[0] = color_r[hit.target];
	pixel[1] = color_g[hit.target];
	pixel[2] = color_b[hit.target];
	return;

#endif
	// decode plane hit - reconstruct its axis and sign
	const __m128 axis_sign = _mm_and_ps(_mm_set1_ps(-0.f), hit.min_mask);

#if __AVX__
	const int xyz = 0x020100; // x-axis: 0 1 2
	const int zxy = 0x010002; // y-axis: 2 0 1
	const int yzx = 0x000201; // z-axis: 1 2 0

	const size_t axis = (xyz & hit.a_mask | zxy & ~hit.a_mask) & hit.b_mask | yzx & ~hit.b_mask;
	const __m128i perm = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(axis));

#else
	const int axis_x = 0;
	const int axis_y = 1;
	const int axis_z = 2;

	const size_t axis = (axis_x & hit.a_mask | axis_y & ~hit.a_mask) & hit.b_mask | axis_z & ~hit.b_mask;

#endif
	const simd::vect3 orig = simd::vect3().add(
		ray.get_origin(), simd::vect3().mul(ray.get_direction(), hit.dist));

	__m128 lit = _mm_set1_ps(0.f);
	__m128 all = _mm_set1_ps(0.f);

	// manually unroll the AO shading loop by 4
	for (size_t i = 0; i < ao_probe_count / 4; ++i)
	{
		const compile_assert< 0 == (RAND_MAX & RAND_MAX + 1L) > assert_rand_pot;
		const int rnd0 = rand_r(&seed);
		const int rnd1 = rand_r(&seed);
		const int rnd2 = rand_r(&seed);
		const int rnd3 = rand_r(&seed);
		const int rnd4 = rand_r(&seed);
		const int rnd5 = rand_r(&seed);
		const int rnd6 = rand_r(&seed);
		const int rnd7 = rand_r(&seed);
		const __m128i ri0 = _mm_setr_epi32(rnd0, rnd1, rnd2, rnd3);
		const __m128i ri1 = _mm_setr_epi32(rnd4, rnd5, rnd6, rnd7);

		// cosine-weighted distribution
		const __m128 r0 = _mm_mul_ps(_mm_cvtepi32_ps(ri0), _mm_setr_ps(
			1.0 / RAND_MAX,   // decl0 (cos^2)
			1.0 / RAND_MAX,   // decl1 (cos^2)
			1.0 / RAND_MAX,   // decl2 (cos^2)
			1.0 / RAND_MAX)); // decl3 (cos^2)
		const __m128 r1 = _mm_mul_ps(_mm_cvtepi32_ps(ri1), _mm_setr_ps(
			M_PI * 2 / (RAND_MAX + 1L),   // azim0
			M_PI * 2 / (RAND_MAX + 1L),   // azim1
			M_PI * 2 / (RAND_MAX + 1L),   // azim2
			M_PI * 2 / (RAND_MAX + 1L))); // azim3
		const __m128 sin_decl = _mm_sqrt_ps(_mm_sub_ps(_mm_set1_ps(1.f), r0));
		const __m128 cos_decl = _mm_sqrt_ps(r0);
		all = _mm_add_ps(all, cos_decl);
		__m128 sin_azim;
		__m128 cos_azim;
		sincos_ps(r1, &sin_azim, &cos_azim);

		// compute a bounce vector in some TBN space, in this case of an assumed normal along x-axis
		simd::vect3 hemi0 = simd::vect3(cos_decl[0], cos_azim[0] * sin_decl[0], sin_azim[0] * sin_decl[0], true);
		simd::vect3 hemi1 = simd::vect3(cos_decl[1], cos_azim[1] * sin_decl[1], sin_azim[1] * sin_decl[1], true);
		simd::vect3 hemi2 = simd::vect3(cos_decl[2], cos_azim[2] * sin_decl[2], sin_azim[2] * sin_decl[2], true);
		simd::vect3 hemi3 = simd::vect3(cos_decl[3], cos_azim[3] * sin_decl[3], sin_azim[3] * sin_decl[3], true);

#if __AVX__
		// permute bounce direction depending on which axial plane was hit
		const __m128 pdir0 = _mm_permutevar_ps(hemi0.getn(), perm);
		const __m128 pdir1 = _mm_permutevar_ps(hemi1.getn(), perm);
		const __m128 pdir2 = _mm_permutevar_ps(hemi2.getn(), perm);
		const __m128 pdir3 = _mm_permutevar_ps(hemi3.getn(), perm);

		simd::vect3 probe_dir0;
		simd::vect3 probe_dir1;
		simd::vect3 probe_dir2;
		simd::vect3 probe_dir3;

#else
#error unsupported ISA for this block

#endif
		probe_dir0.setn(0, _mm_xor_ps(pdir0, axis_sign));
		probe_dir1.setn(0, _mm_xor_ps(pdir1, axis_sign));
		probe_dir2.setn(0, _mm_xor_ps(pdir2, axis_sign));
		probe_dir3.setn(0, _mm_xor_ps(pdir3, axis_sign));

		const Ray probe0(orig, probe_dir0);
		const Ray probe1(orig, probe_dir1);
		const Ray probe2(orig, probe_dir2);
		const Ray probe3(orig, probe_dir3);

		const __m128i shadow_hit = _mm_setr_epi32(
			ts.traverse_litest(probe0, hit) ? 0 : -1,
			ts.traverse_litest(probe1, hit) ? 0 : -1,
			ts.traverse_litest(probe2, hit) ? 0 : -1,
			ts.traverse_litest(probe3, hit) ? 0 : -1);
		lit = _mm_add_ps(lit, _mm_and_ps(cos_decl, _mm_castsi128_ps(shadow_hit)));
	}

	const float intensity = sqrtf((lit[0] + lit[1] + lit[2] + lit[3]) / (all[0] + all[1] + all[2] + all[3]));

	pixel[0] = uint8_t(255.f * intensity);
	pixel[1] = uint8_t(255.f * intensity);
	pixel[2] = uint8_t(255.f * intensity);

	// truncate payload id to 6 LSBs when storing it in the pixel
	pixel[3] = size_t(hit.target) << 2 | (axis & 3) + 1;
}


#if DIVISION_OF_LABOR_VER == 2
static const unsigned batch = 32;
static struct __attribute__ ((aligned(64))) // one per cacheline
{
	unsigned cursor;
}
worker[nthreads];

#elif DIVISION_OF_LABOR_VER == 1
static const unsigned batch = 32;
static unsigned workgroup_cursor;

#endif

static void*
compute(
	void* arg)
{
	compute_arg* const carg = reinterpret_cast< compute_arg* >(arg);
	uint8_t (* const framebuffer)[4] = carg->framebuffer;

#if FB_RES_FIXED_W
	const unsigned w = FB_RES_FIXED_W;

#else
	const unsigned w = carg->w;

#endif
#if FB_RES_FIXED_H
	const unsigned h = FB_RES_FIXED_H;

#else
	const unsigned h = carg->h;

#endif
	pthread_barrier_t* const barrier_start = barrier + BARRIER_START;
	pthread_barrier_t* const barrier_finish = barrier + BARRIER_FINISH;

frame_loop:
	pthread_barrier_wait(barrier_start);

	const size_t id = carg->id;
	const size_t frame = carg->frame;

	if (uint32_t(-1) == uint32_t(id))
		return 0;

	const Timeslice* const ts = carg->tree;
	const simd::vect3 (& cam)[4] = carg->cam;

#if DIVISION_OF_LABOR_VER == 2
	// enhanced dynamic division of labor: each worker finishes its pre-assigned portion, then claims the next batch of work
	// from a sibling's assignment, until that is done; claims are made on batches of pixels to decrease the claim frequency

	for (size_t i = 0; i < nthreads; ++i)
	{
		const size_t effective_id = (id + i) % nthreads;

		while (true)
		{
			const unsigned cursor = __atomic_fetch_add(&worker[effective_id].cursor, batch, __ATOMIC_RELAXED);

			if (cursor >= w * h / unsigned(nthreads))
				break;

			for (unsigned ci = 0; ci < batch; ++ci)
			{
				const unsigned linear = cursor * nthreads + effective_id * batch + ci;
				const unsigned y = linear / w;
				const unsigned x = linear % w;

				if ((y ^ x) % 2 != frame % 2)
					continue;

				const simd::vect3 offs = simd::vect3().add(
					simd::vect3(cam[1]).mul((int(y) * 2 - int(h)) * (1.f / h)),
					simd::vect3(cam[0]).mul((int(x) * 2 - int(w)) * (1.f / w)));

				const Ray ray(simd::vect3().add(cam[3], offs), cam[2]);

				shade(*ts, ray, carg->hit, carg->seed, framebuffer[linear]);

#if COLORIZE_THREADS == 1
				framebuffer[y * w + x][id % 4] += 32;

#endif
			}
		}
	}

#elif DIVISION_OF_LABOR_VER == 1
	// dynamic division of labor: each available worker claims the next batch of work from a shared assignment;
	// claims are made on batches of pixels to decrease the claim frequency

	while (true)
	{
		const unsigned cursor = __atomic_fetch_add(&workgroup_cursor, batch, __ATOMIC_RELAXED);
		const unsigned end = cursor + batch;

		if (cursor >= w * h)
			break;

		for (unsigned ci = cursor; ci < end; ++ci)
		{
			const unsigned y = ci / w;
			const unsigned x = ci % w;

			if ((y ^ x) % 2 != frame % 2)
				continue;

			const simd::vect3 offs = simd::vect3().add(
				simd::vect3(cam[1]).mul((int(y) * 2 - int(h)) * (1.f / h)),
				simd::vect3(cam[0]).mul((int(x) * 2 - int(w)) * (1.f / w)));

			const Ray ray(simd::vect3().add(cam[3], offs), cam[2]);

			shade(*ts, ray, carg->hit, carg->seed, framebuffer[y * w + x]);

#if COLORIZE_THREADS == 1
			framebuffer[y * w + x][id % 4] += 32;

#endif
		}
	}

#else
	// static division of labor: each worker gets pre-assigned an equal portion of the workspace;
	// equal portions do not equate equal amounts of work, though, so some workers will finish early and idle

	for (unsigned y = 0; y < h; ++y)
	{
		for (unsigned x = 0; x < w; ++x)
		{
			if ((y ^ x) % 2 != frame % 2)
				continue;

			if ((y ^ x) / 2 % nthreads != id)
				continue;

			const simd::vect3 offs = simd::vect3().add(
				simd::vect3(cam[1]).mul((int(y) * 2 - int(h)) * (1.f / h)),
				simd::vect3(cam[0]).mul((int(x) * 2 - int(w)) * (1.f / w)));

			const Ray ray(simd::vect3().add(cam[3], offs), cam[2]);

			shade(*ts, ray, carg->hit, carg->seed, framebuffer[y * w + x]);

#if COLORIZE_THREADS == 1
			framebuffer[y * w + x][id % 4] += 32;

#endif
		}
	}

#endif

	pthread_barrier_wait(barrier_finish);

	if (0 != id)
		goto frame_loop;

	return 0;
}


static pthread_t thread[one_less];
static compute_arg record[one_less];

static const compile_assert< COUNT_OF(thread) == COUNT_OF(record) > assert_record_number;


class workforce_t
{
	size_t barriers_created;
	size_t threads_created;
	bool successfully_init;

public:
	workforce_t(
		uint8_t (* const framebuffer)[4],
		const unsigned w,
		const unsigned h);

	~workforce_t();

	bool is_successfully_init() const;

	void update(
		const size_t frame,
		const simd::vect3 (& cam)[4],
		const Timeslice& tree);
};


static void
report_err(
	const char* const func,
	const int line,
	const size_t counter,
	const int err)
{
	stream::cerr << func << ':' << line << ", i: " << counter << ", err: " << err << '\n';
}


workforce_t::workforce_t(
	uint8_t (* const framebuffer)[4],
	const unsigned w,
	const unsigned h)
: barriers_created(0)
, threads_created(0)
, successfully_init(false)
{
	for (size_t i = 0; i < COUNT_OF(barrier); ++i)
	{
		const int r = pthread_barrier_init(barrier + i, 0, nthreads);

		if (0 != r)
		{
			report_err(__FUNCTION__, __LINE__, i, r);
			return;
		}

		++barriers_created;
	}

	for (size_t i = 0; i < COUNT_OF(record); ++i)
	{
		const size_t id = i + 1;
		record[i] = compute_arg(id, framebuffer, w, h);

#if WORKFORCE_THREADS_STICKY == 1
		struct scoped_t
		{
			pthread_attr_t attr;
			bool successfully_init;

			scoped_t(
				const size_t i)
			: successfully_init(false)
			{
				const int r = pthread_attr_init(&attr);

				if (0 != r)
				{
					report_err(__FUNCTION__, __LINE__, i, r);
					return;
				}

				successfully_init = true;
			}

			~scoped_t()
			{
				if (successfully_init)
					pthread_attr_destroy(&attr);
			}
		};

		scoped_t scoped(i);

		if (!scoped.successfully_init)
			return;

		cpu_set_t affin;
		CPU_ZERO(&affin);
		CPU_SET(id, &affin);

		const int ra = pthread_attr_setaffinity_np(&scoped.attr, sizeof(affin), &affin);

		if (0 != ra)
		{
			report_err(__FUNCTION__, __LINE__, i, ra);
			return;
		}

		const int r = pthread_create(thread + i, &scoped.attr, compute, record + i);

#else
		const int r = pthread_create(thread + i, 0, compute, record + i);

#endif
		if (0 != r)
		{
			report_err(__FUNCTION__, __LINE__, i, r);
			return;
		}

		++threads_created;
	}

	successfully_init = true;
}


workforce_t::~workforce_t()
{
	for (size_t i = 0; i < COUNT_OF(record); ++i)
		record[i].id = uint32_t(-1);

	if (barriers_created)
		pthread_barrier_wait(barrier + BARRIER_START);

	for (size_t i = 0; i < barriers_created; ++i)
	{
		const int r = pthread_barrier_destroy(barrier + i);

		if (0 != r)
			report_err(__FUNCTION__, __LINE__, i, r);
	}

	for (size_t i = 0; i < threads_created; ++i)
	{
		const int r = pthread_join(thread[i], 0);

		if (0 != r)
			report_err(__FUNCTION__, __LINE__, i, r);
	}
}


bool
workforce_t::is_successfully_init() const
{
	return successfully_init;
}


void
workforce_t::update(
	const size_t frame,
	const simd::vect3 (& cam)[4],
	const Timeslice& tree)
{
	for (size_t i = 0; i < COUNT_OF(record); ++i)
	{
		record[i].frame = frame;
		record[i].tree = &tree;
		record[i].cam[0] = cam[0];
		record[i].cam[1] = cam[1];
		record[i].cam[2] = cam[2];
		record[i].cam[3] = cam[3];
	}
}


static bool
validate_fullscreen(
	const char* const string,
	unsigned& screen_w,
	unsigned& screen_h)
{
	if (0 == string)
		return false;

	unsigned x, y, hz;

	if (3 != sscanf(string, "%u %u %u", &x, &y, &hz))
		return false;

	if (!x || !y || !hz)
		return false;

	screen_w = x;
	screen_h = y;

	return true;
}


static bool
validate_bitness(
	const char* const string,
	unsigned (& screen_bitness)[4])
{
	if (0 == string)
		return false;

	unsigned bitness[4];

	if (4 != sscanf(string, "%u %u %u %u",
			&bitness[0],
			&bitness[1],
			&bitness[2],
			&bitness[3]))
	{
		return false;
	}

	if (!bitness[0] || 16 < bitness[0] ||
		!bitness[1] || 16 < bitness[1] ||
		!bitness[2] || 16 < bitness[2] ||
		16 < bitness[3])
	{
		return false;
	}

	screen_bitness[0] = bitness[0];
	screen_bitness[1] = bitness[1];
	screen_bitness[2] = bitness[2];
	screen_bitness[3] = bitness[3];

	return true;
}


class matx3_rotate : public simd::matx3
{
	matx3_rotate();

public:
	matx3_rotate(
		const float a,
		const float x,
		const float y,
		const float z)
	{
		__m128 sin_ang;
		__m128 cos_ang;
		sincos_ps(_mm_set_ss(a), &sin_ang, &cos_ang);

		const float sin_a = sin_ang[0];
		const float cos_a = cos_ang[0];

		static_cast< simd::matx3& >(*this) = simd::matx3(
			x * x + cos_a * (1 - x * x),         x * y - cos_a * (x * y) + sin_a * z, x * z - cos_a * (x * z) - sin_a * y,
			y * x - cos_a * (y * x) - sin_a * z, y * y + cos_a * (1 - y * y),         y * z - cos_a * (y * z) + sin_a * x,
			z * x - cos_a * (z * x) + sin_a * y, z * y - cos_a * (z * y) - sin_a * x, z * z + cos_a * (1 - z * z));
	}
};


class matx4_rotate : public simd::matx4
{
	matx4_rotate();

public:
	matx4_rotate(
		const float a,
		const float x,
		const float y,
		const float z)
	{
		__m128 sin_ang;
		__m128 cos_ang;
		sincos_ps(_mm_set_ss(a), &sin_ang, &cos_ang);

		const float sin_a = sin_ang[0];
		const float cos_a = cos_ang[0];

		static_cast< simd::matx4& >(*this) = simd::matx4(
			x * x + cos_a * (1 - x * x),         x * y - cos_a * (x * y) + sin_a * z, x * z - cos_a * (x * z) - sin_a * y, 0.f,
			y * x - cos_a * (y * x) - sin_a * z, y * y + cos_a * (1 - y * y),         y * z - cos_a * (y * z) + sin_a * x, 0.f,
			z * x - cos_a * (z * x) + sin_a * y, z * y - cos_a * (z * y) - sin_a * x, z * z + cos_a * (1 - z * z),         0.f,
			0.f,                                 0.f,                                 0.f,                                 1.f);
	}
};

struct Param {
	unsigned w;             // frame width
	unsigned h;             // frame height
	unsigned bitness[4];    // rgba bitness
	unsigned fsaa;          // fsaa number of samples
	unsigned frames;        // frames to run
};

static int
parse_cli(
	const int argc,
	char** const argv,
	Param& param)
{
	const size_t prefix_len = strlen(arg_prefix);
	bool success = true;

	for (int i = 1; i < argc && success; ++i)
	{
		if (strncmp(argv[i], arg_prefix, prefix_len))
		{
			success = false;
			continue;
		}

		if (!strcmp(argv[i] + prefix_len, arg_screen))
		{
			if (!(++i < argc) || !validate_fullscreen(argv[i], param.w, param.h))
				success = false;

			continue;
		}

		if (!strcmp(argv[i] + prefix_len, arg_bitness))
		{
			if (!(++i < argc) || !validate_bitness(argv[i], param.bitness))
				success = false;

			continue;
		}

		if (!strcmp(argv[i] + prefix_len, arg_fsaa))
		{
			if (!(++i < argc) || (1 != sscanf(argv[i], "%u", &param.fsaa)))
				success = false;

			continue;
		}

		if (!strcmp(argv[i] + prefix_len, arg_nframes))
		{
			if (!(++i < argc) || (1 != sscanf(argv[i], "%u", &param.frames)))
				success = false;

			continue;
		}

		if (!strcmp(argv[i] + prefix_len, arg_seed))
		{
			if (!(++i < argc) || (1 != sscanf(argv[i], "%u", &shape_seed)))
				success = false;

			continue;
		}

		if (!strcmp(argv[i] + prefix_len, arg_smooth))
		{
			offset = 0.f;
			continue;
		}

		success = false;
	}

	if (!success)
	{
		stream::cerr << "usage: " << argv[0] << " [<option> ...]\n"
			"options (multiple args to an option must constitute a single string, eg. -foo \"a b c\"):\n"
			"\t" << arg_prefix << arg_screen << " <width> <height> <Hz>\t\t: set fullscreen output of specified geometry and refresh\n"
			"\t" << arg_prefix << arg_bitness << " <r> <g> <b> <a>\t\t: set GLX config of specified RGBA bitness; default is screen's bitness\n"
			"\t" << arg_prefix << arg_fsaa << " <positive_integer>\t\t: set GL fullscreen antialiasing; default is none\n"
			"\t" << arg_prefix << arg_nframes << " <unsigned_integer>\t\t: set number of frames to run; default is max unsigned int\n"
			"\t" << arg_prefix << arg_seed << " <unsigned_integer>\t\t: set game's PRNG seed\n"
			"\t" << arg_prefix << arg_smooth << "\t\t\t\t\t: align pieces, forming a smooth wall\n";

		return 1;
	}

	return 0;
}


template < typename T >
class generic_free
{
public:

	void operator()(T* arg)
	{
		assert(0 != arg);
		free(arg);
	}
};


static const size_t playfield_rows = 18;
static const size_t playfield_cols = 6;

static const compile_assert< playfield_cols % 2 == 0 > assert_playfield_cols_even;

struct Shape
{
	uint8_t shape_width[4]; // fragment_count * 16 + shape_width
	int8_t correction[4 * 2]; // rotational correction
	Voxel fragment[4 * 2];
};

static const Shape shape_def[7] =
{
	//
	// ####
	//

	{
		{ (1 << 4) + 4, (1 << 4) + 1, (1 << 4) + 4, (1 << 4) + 1 },
		{
			 1, -1,
			-1,  1,
			 1, -1,
			-1,  1
		},
		{
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(4.f, 1.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(1.f, 4.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(4.f, 1.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(1.f, 4.f,  .5f)),
		}
	},

	//
	// #
	// ###
	//

	{
		{ (2 << 4) + 3, (2 << 4) + 2, (2 << 4) + 3, (2 << 4) + 2 },
		{
			-1,  0,
			-1, -1,
			 2, -1,
			 0,  2
		},
		{
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(1.f, 2.f,  .5f)),
			Voxel(simd::vect3(1.f, 0.f, -.5f), simd::vect3(3.f, 1.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(2.f, 1.f,  .5f)),
			Voxel(simd::vect3(1.f, 1.f, -.5f), simd::vect3(2.f, 3.f,  .5f)),
			Voxel(simd::vect3(2.f, 0.f, -.5f), simd::vect3(3.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 1.f, -.5f), simd::vect3(2.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(1.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 2.f, -.5f), simd::vect3(2.f, 3.f,  .5f))
		}
	},

	//
	//   #
	// ###
	//

	{
		{ (2 << 4) + 3, (2 << 4) + 2, (2 << 4) + 3, (2 << 4) + 2 },
		{
			 1, -2,
			 1,  1,
			 0,  1,
			-2,  0
		},
		{
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(2.f, 1.f,  .5f)),
			Voxel(simd::vect3(2.f, 0.f, -.5f), simd::vect3(3.f, 2.f,  .5f)),
			Voxel(simd::vect3(1.f, 0.f, -.5f), simd::vect3(2.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 2.f, -.5f), simd::vect3(2.f, 3.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(1.f, 2.f,  .5f)),
			Voxel(simd::vect3(1.f, 1.f, -.5f), simd::vect3(3.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(2.f, 1.f,  .5f)),
			Voxel(simd::vect3(0.f, 1.f, -.5f), simd::vect3(1.f, 3.f,  .5f))
		}
	},

	//
	// ##
	// ##
	//

	{
		{ (1 << 4) + 2, (1 << 4) + 2, (1 << 4) + 2, (1 << 4) + 2 },
		{},
		{
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(2.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(2.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(2.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(2.f, 2.f,  .5f))
		}
	},

	//
	//  ##
	// ##
	//

	{
		{ (2 << 4) + 3, (2 << 4) + 2, (2 << 4) + 3, (2 << 4) + 2 },
		{},
		{
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(2.f, 1.f,  .5f)),
			Voxel(simd::vect3(1.f, 1.f, -.5f), simd::vect3(3.f, 2.f,  .5f)),
			Voxel(simd::vect3(1.f, 0.f, -.5f), simd::vect3(2.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 1.f, -.5f), simd::vect3(1.f, 3.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(2.f, 1.f,  .5f)),
			Voxel(simd::vect3(1.f, 1.f, -.5f), simd::vect3(3.f, 2.f,  .5f)),
			Voxel(simd::vect3(1.f, 0.f, -.5f), simd::vect3(2.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 1.f, -.5f), simd::vect3(1.f, 3.f,  .5f))
		}
	},

	//
	//  #
	// ###
	//

	{
		{ (2 << 4) + 3, (2 << 4) + 2, (2 << 4) + 3, (2 << 4) + 2 },
		{
			 0, -1,
			 0,  0,
			 1,  0,
			-1,  1
		},
		{
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(3.f, 1.f,  .5f)),
			Voxel(simd::vect3(1.f, 1.f, -.5f), simd::vect3(2.f, 2.f,  .5f)),
			Voxel(simd::vect3(1.f, 0.f, -.5f), simd::vect3(2.f, 3.f,  .5f)),
			Voxel(simd::vect3(0.f, 1.f, -.5f), simd::vect3(1.f, 2.f,  .5f)),
			Voxel(simd::vect3(1.f, 0.f, -.5f), simd::vect3(2.f, 1.f,  .5f)),
			Voxel(simd::vect3(0.f, 1.f, -.5f), simd::vect3(3.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(1.f, 3.f,  .5f)),
			Voxel(simd::vect3(1.f, 1.f, -.5f), simd::vect3(2.f, 2.f,  .5f))
		}
	},

	//
	// ##
	//  ##
	//

	{
		{ (2 << 4) + 3, (2 << 4) + 2, (2 << 4) + 3, (2 << 4) + 2 },
		{},
		{
			Voxel(simd::vect3(1.f, 0.f, -.5f), simd::vect3(3.f, 1.f,  .5f)),
			Voxel(simd::vect3(0.f, 1.f, -.5f), simd::vect3(2.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(1.f, 2.f,  .5f)),
			Voxel(simd::vect3(1.f, 1.f, -.5f), simd::vect3(2.f, 3.f,  .5f)),
			Voxel(simd::vect3(1.f, 0.f, -.5f), simd::vect3(3.f, 1.f,  .5f)),
			Voxel(simd::vect3(0.f, 1.f, -.5f), simd::vect3(2.f, 2.f,  .5f)),
			Voxel(simd::vect3(0.f, 0.f, -.5f), simd::vect3(1.f, 2.f,  .5f)),
			Voxel(simd::vect3(1.f, 1.f, -.5f), simd::vect3(2.f, 3.f,  .5f))
		}
	}
};


static inline bool
check_pileup_collision(
	const Voxel* const fragment,
	const size_t fragment_count,
	const Array< Voxel >& pileup)
{
	const size_t pileup_count = pileup.getCount();

	for (size_t i = pileup_count; i > 0; --i)
	{
		const Voxel& obstacle = pileup.getElement(i - 1);

		for (size_t j = 0; j < fragment_count; ++j)
			if (fragment[j].get_bbox().has_overlap_open_subdim< 2 >(obstacle.get_bbox()))
				return true;
	}

	return false;
}


static void
game_frame(
	const Array< Voxel >& static_scene,
	Array< Voxel >& pileup,
	Array< Voxel >& payload,
	const unsigned input,
	Timeslice& ts)
{
	static const Voxel* fragment;
	static size_t fragment_count;
	static unsigned shape_width;
	static size_t shape_r;
	static size_t shape;

	static float pos_x;
	static float pos_y;
	static float pos_z;

	const size_t pileup_count = pileup.getCount();

	if (pileup_count >= 2 &&
		(pileup.getElement(pileup_count - 1).get_bbox().get_max()[1] >= playfield_rows ||
		 pileup.getElement(pileup_count - 2).get_bbox().get_max()[1] >= playfield_rows))
	{
		return;
	}

	if (0 == shape)
	{
		shape = rand_r(&shape_seed) % 7 + 1;

		shape_width    = shape_def[shape - 1].shape_width[shape_r] & 0xf;
		fragment_count = shape_def[shape - 1].shape_width[shape_r] >> 4 & 0xf;
		fragment       = shape_def[shape - 1].fragment + fragment_count * shape_r;

		pos_x = -float(shape_width / 2);
		pos_y = playfield_rows;
		pos_z = (shape - 1) * offset;
	}

	const size_t old_r = shape_r;
	const float old_x = pos_x;
	const float old_y = pos_y;

	if (input & INPUT_MASK_ALT_UP)
	{
		const size_t rotated = (shape_r + 1) % 4;

		// counter-clockwise rotation - positional correction is taken from the current orientation
		const float corr_x = shape_def[shape - 1].correction[shape_r * 2 + 0];
		const float corr_y = shape_def[shape - 1].correction[shape_r * 2 + 1];

		if (pos_y + corr_y >= 0 &&
			pos_x + corr_x <= playfield_cols / 2 - float(shape_def[shape - 1].shape_width[rotated] & 0xf) &&
			pos_x + corr_x >= -float(playfield_cols / 2))
		{
			shape_r = rotated;

			shape_width    = shape_def[shape - 1].shape_width[shape_r] & 0xf;
			fragment_count = shape_def[shape - 1].shape_width[shape_r] >> 4 & 0xf;
			fragment       = shape_def[shape - 1].fragment + fragment_count * shape_r;

			pos_x += corr_x;
			pos_y += corr_y;
		}
	}
	else
	if (input & INPUT_MASK_ALT_DOWN)
	{
		const size_t rotated = (shape_r + 3) % 4;

		// clockwise rotation - negated positional correction is taken from the previous orientation
		const float corr_x = -shape_def[shape - 1].correction[rotated * 2 + 0];
		const float corr_y = -shape_def[shape - 1].correction[rotated * 2 + 1];

		if (pos_y + corr_y >= 0 &&
			pos_x + corr_x <= playfield_cols / 2 - float(shape_def[shape - 1].shape_width[rotated] & 0xf) &&
			pos_x + corr_x >= -float(playfield_cols / 2))
		{
			shape_r = rotated;

			shape_width    = shape_def[shape - 1].shape_width[shape_r] & 0xf;
			fragment_count = shape_def[shape - 1].shape_width[shape_r] >> 4 & 0xf;
			fragment       = shape_def[shape - 1].fragment + fragment_count * shape_r;

			pos_x += corr_x;
			pos_y += corr_y;
		}
	}

	if (input & INPUT_MASK_ALT_LEFT)
		if (pos_x > -float(playfield_cols / 2))
		{
			pos_x -= 1.f;
		}

	if (input & INPUT_MASK_ALT_RIGHT)
		if (pos_x < (playfield_cols / 2) - float(shape_width))
		{
			pos_x += 1.f;
		}

	Voxel projection[2];

	for (size_t i = 0; i < fragment_count; ++i)
	{
		projection[i] = Voxel(
			simd::vect3().add(simd::vect3(pos_x, pos_y, pos_z), fragment[i].get_min()),
			simd::vect3().add(simd::vect3(pos_x, pos_y, pos_z), fragment[i].get_max()));
	}

	if (check_pileup_collision(projection, fragment_count, pileup))
	{
		shape_r = old_r;
		pos_x = old_x;
		pos_y = old_y;

		shape_width    = shape_def[shape - 1].shape_width[shape_r] & 0xf;
		fragment_count = shape_def[shape - 1].shape_width[shape_r] >> 4 & 0xf;
		fragment       = shape_def[shape - 1].fragment + fragment_count * shape_r;
	}

	const float step_y = .125f;

	for (size_t i = 0; i < fragment_count; ++i)
	{
		projection[i] = Voxel(
			simd::vect3().add(simd::vect3(pos_x, pos_y - step_y, pos_z), fragment[i].get_min()),
			simd::vect3().add(simd::vect3(pos_x, pos_y - step_y, pos_z), fragment[i].get_max()));
	}

	const bool pileup_hit = check_pileup_collision(projection, fragment_count, pileup);

	payload.resetCount();

	if (pileup_hit || 0.f == pos_y)
	{
		for (size_t i = 0; i < fragment_count; ++i)
		{
			const Voxel piece(
				simd::vect3().add(simd::vect3(pos_x, pos_y, pos_z), fragment[i].get_min()),
				simd::vect3().add(simd::vect3(pos_x, pos_y, pos_z), fragment[i].get_max()));

			if (!pileup.addElement(piece))
			{
				stream::cerr << "game error: out of pileup capacity\n";
				return;
			}
		}

		shape_r = 0;
		shape = 0;
	}
	else
	{
		for (size_t i = 0; i < fragment_count; ++i)
		{
			if (!payload.addElement(projection[i]))
			{
				stream::cerr << "game error: out of payload capacity\n";
				return;
			}
		}

		pos_y -= step_y;
	}

	for (size_t i = 0; i < pileup.getCount(); ++i)
		if (!payload.addElement(pileup.getElement(i)))
		{
			stream::cerr << "game error: out of payload capacity\n";
			return;
		}

	for (size_t i = 0; i < static_scene.getCount(); ++i)
		if (!payload.addElement(static_scene.getElement(i)))
		{
			stream::cerr << "game error: out of payload capacity\n";
			return;
		}

	if (!ts.set_payload_array(payload))
		stream::cerr << "game error: failed setting tree payload\n";
}

////////////////////////////////////////////////////////////////////////////////
// this is a marker for the diff tool to spot main easier this is a marker for
// this is a marker for the diff tool to spot main easier this is a marker for
// this is a marker for the diff tool to spot main easier this is a marker for
// this is a marker for the diff tool to spot main easier this is a marker for
// this is a marker for the diff tool to spot main easier this is a marker for
// this is a marker for the diff tool to spot main easier this is a marker for
// this is a marker for the diff tool to spot main easier this is a marker for
// this is a marker for the diff tool to spot main easier this is a marker for
////////////////////////////////////////////////////////////////////////////////

int main(
	int argc,
	char** argv)
{
	stream::cin.open(stdin);
	stream::cout.open(stdout);
	stream::cerr.open(stderr);

	const unsigned default_w = 512;
	const unsigned default_h = 512;

	Param param = {
		default_w, // param.w 
		default_h, // param.h 
		{ 0 },     // param.bitness
		0,         // param.fsaa
		-1U,       // param.frames
	};
 
	const int result_cli = parse_cli(argc, argv, param);

	if (0 != result_cli)
		return result_cli;

#if FB_RES_FIXED_W
	const unsigned w = FB_RES_FIXED_W;

#else
	const unsigned w = param.w;

#endif
#if FB_RES_FIXED_H
	const unsigned h = FB_RES_FIXED_H;

#else
	const unsigned h = param.h;

#endif
	unsigned (& bitness)[4] = param.bitness;
	const unsigned fsaa = param.fsaa;
	const unsigned frames = param.frames;

#if DIVISION_OF_LABOR_VER == 2
	if (w * h % (nthreads * batch))
	{
		stream::cerr << "error: screen resolution product (H x W) must be multiple of " << nthreads * batch << '\n';
		return -1;
	}

#elif DIVISION_OF_LABOR_VER == 1
	if (w * h % batch)
	{
		stream::cerr << "error: screen resolution product (H x W) must be multiple of " << batch << '\n';
		return -1;
	}

#endif
#if VISUALIZE != 0
	const int result_gl = testbed::initGL(
		w,
		h,
		bitness,
		-1,
		fsaa);

	if (0 != result_gl)
		return result_gl;

	const testbed::scoped_ptr< testbed::deinit_resources_t, testbed::scoped_functor > deinitGL(testbed::deinitGL);

	if (!testbed::util::reportGLCaps() ||
		!testbed::rgbv::init_resources(w, h))
	{
		stream::cerr << "failed to initialise GL or resources; bailing out\n";
		return 1;
	}

	const testbed::scoped_ptr< testbed::deinit_resources_t, testbed::scoped_functor > deinit(testbed::rgbv::deinit_resources);

#endif
	// static parts of the scene
	Array< Voxel > static_scene;
	static_scene.setCapacity(4);
	static_scene.addElement(Voxel(
		simd::vect3(-5.f, -1.f, -1.5f),
		simd::vect3( 5.f,  0.f,  1.5f)));
	static_scene.addElement(Voxel(
		simd::vect3(-float(playfield_cols / 2), 0.f, -1.5f),
		simd::vect3( float(playfield_cols / 2), float(playfield_rows), -.5f)));
	static_scene.addElement(Voxel(
		simd::vect3(-float(playfield_cols / 2 + 1), 0.f, -.5f),
		simd::vect3(-float(playfield_cols / 2), float(playfield_rows), .5f)));
	static_scene.addElement(Voxel(
		simd::vect3(float(playfield_cols / 2), 0.f, -.5f),
		simd::vect3(float(playfield_cols / 2 + 1), float(playfield_rows), .5f)));

	// compute a world bbox from the static parts of the scene
	BBox world_bbox;

	for (size_t i = 0; i < static_scene.getCount(); ++i)
		world_bbox.grow(static_scene.getElement(i).get_bbox());

	// use the world bbox to compute a normalization (pan_n_zoom) matrix
	const __m128 bbox_min = world_bbox.get_min();
	const __m128 bbox_max = world_bbox.get_max();
	const __m128 centre = _mm_mul_ps(
		_mm_add_ps(bbox_max, bbox_min),
		_mm_set1_ps(.5f));
	const __m128 extent = _mm_mul_ps(
		_mm_sub_ps(bbox_max, bbox_min),
		_mm_set1_ps(.5f));
	const float max_extent = std::max(extent[0], std::max(extent[1], extent[2])) / .875f;

	const simd::matx4 zoom_n_pan(
		max_extent, 0.f, 0.f, 0.f,
		0.f, max_extent, 0.f, 0.f,
		0.f, 0.f, max_extent, 0.f,
		centre[0],
		centre[1],
		centre[2], 1.f);

	simd::matx4 rot = simd::matx4().identity();

	const size_t frame_size = w * h * sizeof(uint8_t[4]); // our rendering produces RGBA8 pixels
	
	const size_t cacheline_size = 64;
	const size_t cacheline_pad = cacheline_size - 1;

	const testbed::scoped_ptr< uint8_t[4], generic_free > unaligned_fb(
		reinterpret_cast< uint8_t(*)[4] >(malloc(frame_size + cacheline_pad)));

	// get that buffer cacheline aligned
	uint8_t (* const framebuffer)[4] = reinterpret_cast< uint8_t(*)[4] >(uintptr_t(unaligned_fb()) + uintptr_t(cacheline_pad) & ~uintptr_t(cacheline_pad));
	memset(framebuffer, 0, frame_size);

	workforce_t workforce(framebuffer, w, h);

	if (!workforce.is_successfully_init())
	{
		stream::cerr << "failed to raise workforce; bailing out\n";
		return -1;
	}

	Array< Voxel > pileup;

	if (!pileup.setCapacity(64))
	{
		stream::cerr << "failed to allocate pileup; bailing out\n";
		return -1;
	}

	Array< Voxel > payload;

	if (!payload.setCapacity(128))
	{
		stream::cerr << "failed to allocate payload; bailing out\n";
		return -1;
	}

#if defined(prob_4_H__)
	Timeslice ts;

#elif defined(prob_7_H__)
	const testbed::scoped_ptr< TimesliceBalloon, generic_free > unaligned_ts(
		reinterpret_cast< TimesliceBalloon* >(malloc(sizeof(TimesliceBalloon) + 4095)));
	Timeslice& ts = *new (reinterpret_cast<void*>(uintptr_t(unaligned_ts()) + uintptr_t(4095) & ~uintptr_t(4095))) TimesliceBalloon;

#else
	#error prob_4_H__ or prob_7_H__ required

#endif
	unsigned input = 0;
	unsigned nframes = 0;
	const uint64_t t0 = timer_ns();

#if VISUALIZE != 0
	while (testbed::processEvents(input) && nframes != frames)

#else
	while (nframes != frames)

#endif
	{
		game_frame(static_scene, pileup, payload, input, ts);

		// reset rotation input
		input &= ~INPUT_MASK_ALT_UP & ~INPUT_MASK_ALT_DOWN;

		if (input & INPUT_MASK_ACTION)
		{
			input &= ~INPUT_MASK_ACTION;

			// take action here
			FILE* f = fopen("state", "rb");

			if (0 == f)
			{
				f = fopen("state", "wb");
				fwrite(&rot, sizeof(rot), 1, f);
			}
			else
				fread(&rot, sizeof(rot), 1, f);

			fclose(f);
		}

		float decl = 0.f;
		float azim = 0.f;
		const float angular_step = M_PI_2 / (1 << 4);

		if (input & INPUT_MASK_UP)
			decl += angular_step;

		if (input & INPUT_MASK_DOWN)
			decl -= angular_step;

		if (input & INPUT_MASK_LEFT)
			azim += angular_step;

		if (input & INPUT_MASK_RIGHT)
			azim -= angular_step;

		rot.mulr(matx4_rotate(decl, 1.f, 0.f, 0.f));
		rot.mulr(matx4_rotate(azim, 0.f, 1.f, 0.f));

		// forward: pan * zoom * rot * eyep
		// inverse: (eyep)-1 * rotT * (zoom)-1 * (pan)-1

		const simd::matx4 mv_inv = simd::matx4(
			1.f, 0.f, 0.f, 0.f,
			0.f, 1.f, 0.f, 0.f,
			0.f, 0.f, 1.f, 0.f,
			0.f, 0.f, 1.f, 1.f).mulr(simd::matx4().transpose(rot)).mulr(zoom_n_pan);

		const simd::vect3 cam[] = {
			simd::vect3(mv_inv[0][0], mv_inv[0][1], mv_inv[0][2]),
			simd::vect3(mv_inv[1][0], mv_inv[1][1], mv_inv[1][2]).mul(float(h) / w),
			simd::vect3(mv_inv[2][0], mv_inv[2][1], mv_inv[2][2]).normalise().negate(),
			simd::vect3(mv_inv[3][0], mv_inv[3][1], mv_inv[3][2])
		};
		// note: normalisation above is not needed by the tracing arithmetic, but as
		// a source of micro-jitter, helpful when tracing at orthographic projection

		workforce.update(nframes, cam, ts);

#if DIVISION_OF_LABOR_VER == 2
		for (size_t i = 0; i < nthreads; ++i)
			worker[i].cursor = 0;

#elif DIVISION_OF_LABOR_VER == 1
		workgroup_cursor = 0;

#endif
		compute_arg carg(0, nframes, cam, ts, framebuffer, w, h);
		compute(&carg);

#if VISUALIZE != 0
		testbed::rgbv::render(framebuffer);
		testbed::swapBuffers();

#endif
		++nframes;
	}

	const uint64_t sequence_dt = timer_ns() - t0;

	stream::cout << "compute_arg size: " << sizeof(compute_arg) <<
		"\nworker threads: " << nthreads << "\nambient occlusion rays per pixel: " << ao_probe_count <<
		"\ntotal frames rendered: " << nframes << '\n';

	if (sequence_dt)
	{
		const double sec = double(sequence_dt) * 1e-9;

		stream::cout << "elapsed time: " << sec << " s"
			"\naverage FPS: " << nframes / sec << '\n';
	}

	return 0;
}
