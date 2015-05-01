#include <GL/gl.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <pthread.h>

#if FRAMEGRAB_RATE != 0
	#include <png.h>
#endif

#include "sse_mathfun.h"
#include "testbed.hpp"
#include "stream.hpp"
#include "vectsimd_sse.hpp"
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

static const size_t nthreads = WORKFORCE_NUM_THREADS;
static const size_t one_less = nthreads - 1;
static const size_t ao_probe_count = AO_NUM_RAYS;

static const compile_assert< ao_probe_count % 2 == 0 > assert_ao_probe_count_even;

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

#if __AVX__ != 0
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

	float data = 0.f;

	// manually unroll the AO shading loop by 2
	for (size_t i = 0; i < ao_probe_count / 2; ++i)
	{
#if 1
		const int angular_granularity = 8192;
#else
		const int angular_granularity = ao_probe_count * 16; // grid coefficient should avoid moire till at least 1024 probes
#endif

#if CHEAP_LINEAR_DISTRIBUTION == 0
		const float decl0 = float(M_PI_2) * (rand_r(&seed) % angular_granularity) / angular_granularity;
		const float azim0 = float(M_PI) * (rand_r(&seed) % (angular_granularity * 4)) / (angular_granularity * 2);
		const float decl1 = float(M_PI_2) * (rand_r(&seed) % angular_granularity) / angular_granularity;
		const float azim1 = float(M_PI) * (rand_r(&seed) % (angular_granularity * 4)) / (angular_granularity * 2);

		__m128 sin_dazim;
		__m128 cos_dazim;
		const __m128 ang_dazim = _mm_setr_ps(decl0, azim0, decl1, azim1);
		sincos_ps(ang_dazim, &sin_dazim, &cos_dazim);

		const float sin_decl0 = sin_dazim[0];
		const float cos_decl0 = cos_dazim[0];
		const float sin_azim0 = sin_dazim[1];
		const float cos_azim0 = cos_dazim[1];
		const float sin_decl1 = sin_dazim[2];
		const float cos_decl1 = cos_dazim[2];
		const float sin_azim1 = sin_dazim[3];
		const float cos_azim1 = cos_dazim[3];

		// compute a bounce vector in some TBN space, in this case of an assumed normal along x-axis
		simd::vect3 hemi0 = simd::vect3(cos_decl0, cos_azim0 * sin_decl0, sin_azim0 * sin_decl0, true);
		simd::vect3 hemi1 = simd::vect3(cos_decl1, cos_azim1 * sin_decl1, sin_azim1 * sin_decl1, true);

#else
		simd::vect3 hemi0 = simd::vect3(
			float(rand_r(&seed) % angular_granularity),
			float(rand_r(&seed) % angular_granularity - angular_granularity / 2),
			float(rand_r(&seed) % angular_granularity - angular_granularity / 2)).normalise();
		simd::vect3 hemi1 = simd::vect3(
			float(rand_r(&seed) % angular_granularity),
			float(rand_r(&seed) % angular_granularity - angular_granularity / 2),
			float(rand_r(&seed) % angular_granularity - angular_granularity / 2)).normalise();

#endif
		// now, direct the bounce vector in accordance with the actual normal

#if __AVX__ != 0
		// permute bounce direction depending on which axial plane was hit
		const __m128 pdir0 = _mm_permutevar_ps(hemi0.getn(), perm);
		const __m128 pdir1 = _mm_permutevar_ps(hemi1.getn(), perm);

		simd::vect3 probe_dir0;
		probe_dir0.setn(0, pdir0);

		simd::vect3 probe_dir1;
		probe_dir1.setn(0, pdir1);

#elif BOUNCE_COMPUTE_VER == 2
		// this bounce computation does 0, 1 or 2 rotations of the vector, depending
		// on which axial plane was hit: x-axis: 012 -> y-axis: 201 -> z-axis: 120

		__m128 pdir0 = hemi0.getn();
		__m128 pdir1 = hemi1.getn();

		for (size_t i = 0; i < axis; ++i)
		{
			pdir0 = _mm_shuffle_ps(pdir0, pdir0, 0xd2);
			pdir1 = _mm_shuffle_ps(pdir1, pdir1, 0xd2);
		}

		simd::vect3 probe_dir0;
		probe_dir0.setn(0, pdir0);

		simd::vect3 probe_dir1;
		probe_dir1.setn(0, pdir1);

#elif BOUNCE_COMPUTE_VER == 1
		// this bounce computation does at most one rotation, but
		// can cause the compiler to split the control path to
		// this loop into three branches, triggering significant
		// changes in the inlining cost estimates -- use with care!

		__m128 pdir0 = hemi0.getn();
		__m128 pdir1 = hemi1.getn();

		switch (axis)
		{
		case 1:
			pdir0 = _mm_shuffle_ps(pdir0, pdir0, 0xd2);
			pdir1 = _mm_shuffle_ps(pdir1, pdir1, 0xd2);
			break;
		case 2:
			pdir0 = _mm_shuffle_ps(pdir0, pdir0, 0xc9);
			pdir1 = _mm_shuffle_ps(pdir1, pdir1, 0xc9);
			break;
		}

		simd::vect3 probe_dir0;
		probe_dir0.setn(0, pdir0);

		simd::vect3 probe_dir1;
		probe_dir1.setn(0, pdir1);

#else
		// this bounce computation forces spills of the vectors,
		// creating a hotspot at the immediately following load

		simd::vect3 probe_dir0;
		probe_dir0.set((axis + 0),     hemi0.get(0));
		probe_dir0.set((axis + 1) % 3, hemi0.get(1));
		probe_dir0.set((axis + 2) % 3, hemi0.get(2));

		simd::vect3 probe_dir1;
		probe_dir1.set((axis + 0),     hemi1.get(0));
		probe_dir1.set((axis + 1) % 3, hemi1.get(1));
		probe_dir1.set((axis + 2) % 3, hemi1.get(2));

#endif
		probe_dir0.setn(0, _mm_xor_ps(probe_dir0.getn(), axis_sign));
		probe_dir1.setn(0, _mm_xor_ps(probe_dir1.getn(), axis_sign));

		const Ray probe0(orig, probe_dir0);
		const Ray probe1(orig, probe_dir1);

#if 0
		if (!ts.traverse_litest(probe0, hit))
			data += cos_decl;

		if (!ts.traverse_litest(probe1, hit))
			data += cos_decl;
	}

	pixel[0] = std::min(255.f, data * (float(1.414213562 * 255.) / ao_probe_count));
	pixel[1] = std::min(255.f, data * (float(1.414213562 * 255.) / ao_probe_count));
	pixel[2] = std::min(255.f, data * (float(1.414213562 * 255.) / ao_probe_count));

#else
		if (!ts.traverse_litest(probe0, hit))
			data += 1.f;

		if (!ts.traverse_litest(probe1, hit))
			data += 1.f;
	}

	pixel[0] = data * (255.f / ao_probe_count);
	pixel[1] = data * (255.f / ao_probe_count);
	pixel[2] = data * (255.f / ao_probe_count);

#endif
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
	const unsigned w = carg->w;
	const unsigned h = carg->h;

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

				const Ray ray(cam[3], simd::vect3().add(cam[2], offs));

				shade(*ts, ray, carg->hit, carg->seed, framebuffer[y * w + x]);

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

			const Ray ray(cam[3], simd::vect3().add(cam[2], offs));

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

			const Ray ray(cam[3], simd::vect3().add(cam[2], offs));

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

static const compile_assert< sizeof(thread) / sizeof(thread[0]) == sizeof(record) / sizeof(record[0]) > assert_record_number;


class workforce_t
{
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
: successfully_init(false)
{
	// before actually creating them barrier and thread handles must be zero-initialized
	for (size_t i = 0; i < sizeof(barrier) / sizeof(barrier[0]); ++i)
	{
		const int r = pthread_barrier_init(barrier + i, 0, nthreads);

		if (0 != r)
		{
			report_err(__FUNCTION__, __LINE__, i, r);
			return;
		}
	}

	for (size_t i = 0; i < sizeof(record) / sizeof(record[0]); ++i)
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
	}

	successfully_init = true;
}


workforce_t::~workforce_t()
{
	for (size_t i = 0; i < sizeof(record) / sizeof(record[0]); ++i)
		record[i].id = uint32_t(-1);

	pthread_barrier_wait(barrier + BARRIER_START);

	for (size_t i = 0; i < sizeof(barrier) / sizeof(barrier[0]); ++i)
	{
		const int r = pthread_barrier_destroy(barrier + i);

		if (0 != r)
			report_err(__FUNCTION__, __LINE__, i, r);
	}

	for (size_t i = 0; i < sizeof(record) / sizeof(record[0]); ++i)
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
	for (size_t i = 0; i < sizeof(record) / sizeof(record[0]); ++i)
	{
		record[i].frame = frame;
		record[i].tree = &tree;
		record[i].cam[0] = cam[0];
		record[i].cam[1] = cam[1];
		record[i].cam[2] = cam[2];
		record[i].cam[3] = cam[3];
	}
}


static uint64_t
timer_nsec()
{
#if defined(CLOCK_MONOTONIC_RAW)
	const clockid_t clockid = CLOCK_MONOTONIC_RAW;

#else
	const clockid_t clockid = CLOCK_MONOTONIC;

#endif

	timespec t;
	clock_gettime(clockid, &t);

	return t.tv_sec * 1000000000ULL + t.tv_nsec;
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


static int
parse_cli(
	const int argc,
	char** const argv,
	unsigned (& bitness)[4],
	unsigned& w,
	unsigned& h,
	unsigned& fsaa,
	unsigned& frames)
{
	const unsigned prefix_len = strlen(arg_prefix);
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
			if (!(++i < argc) || !validate_fullscreen(argv[i], w, h))
				success = false;

			continue;
		}

		if (!strcmp(argv[i] + prefix_len, arg_bitness))
		{
			if (!(++i < argc) || !validate_bitness(argv[i], bitness))
				success = false;

			continue;
		}

		if (!strcmp(argv[i] + prefix_len, arg_fsaa))
		{
			if (!(++i < argc) || (1 != sscanf(argv[i], "%u", &fsaa)))
				success = false;

			continue;
		}

		if (!strcmp(argv[i] + prefix_len, arg_nframes))
		{
			if (!(++i < argc) || (1 != sscanf(argv[i], "%u", &frames)))
				success = false;

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
			"\t" << arg_prefix << arg_nframes << " <unsigned_integer>\t\t: set number of frames to run; default is max unsigned int\n";

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


#if FRAMEGRAB_RATE != 0

namespace testbed
{

template <>
class scoped_functor< FILE >
{
public:

	void operator()(FILE* arg)
	{
		assert(0 != arg);
		fclose(arg);
	}
};

} // namespace testbed


static bool
write_png(
	const bool grayscale,
	const unsigned w,
	const unsigned h,
	void* const bits,
	FILE* const fp)
{
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr)
		return false;

	png_infop info_ptr = png_create_info_struct(png_ptr);

	if (!info_ptr)
	{
		png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
		return false;
	}

	// declare any RAII before the longjump, lest no destruction at longjump
	const testbed::scoped_ptr< png_bytep, generic_free > row((png_bytepp) malloc(sizeof(png_bytep) * h));

	if (setjmp(png_jmpbuf(png_ptr)))
	{
		png_destroy_write_struct(&png_ptr, &info_ptr);
		return false;
	}

	size_t pixel_size = sizeof(png_byte[3]);
	int color_type = PNG_COLOR_TYPE_RGB;

	if (grayscale)
	{
		pixel_size = sizeof(png_byte);
		color_type = PNG_COLOR_TYPE_GRAY;
	}

	for (size_t i = 0; i < h; ++i)
		row()[i] = (png_bytep) bits + w * (h - 1 - i) * pixel_size;

	png_init_io(png_ptr, fp);
	png_set_compression_level(png_ptr, Z_BEST_COMPRESSION);
	png_set_IHDR(png_ptr, info_ptr, w, h, 8, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_set_rows(png_ptr, info_ptr, row());
	png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

	png_destroy_write_struct(&png_ptr, &info_ptr);
	return true;
}


static bool
saveViewport(
	const unsigned viewport_x,
	const unsigned viewport_y,
	const unsigned viewport_w,
	const unsigned viewport_h,
	const unsigned index,
	const bool grayscale = true)
{
	char name[FILENAME_MAX + 1];

	const char nameBase[] = "seq/frame";
	const char nameExt[] = ".png";

	sprintf(name, "%s%05d%s",
		nameBase, index, nameExt);

	stream::cout << "saving framegrab as '" << name << "'\n";

	const testbed::scoped_ptr< FILE, testbed::scoped_functor > file(fopen(name, "wb"));

	if (0 == file())
	{
		stream::cerr << "failure opening framegrab file '" << name << "'\n";
		return false;
	}

	size_t pixel_size = sizeof(GLubyte[3]);
	GLenum format = GL_RGB;

	if (grayscale)
	{
		pixel_size = sizeof(GLubyte);
		format = GL_RED;
	}

	const unsigned pixels_len = viewport_h * viewport_w * pixel_size;
	const testbed::scoped_ptr< GLvoid, generic_free > pixels(malloc(pixels_len));

	glFinish();
	glReadPixels(viewport_x, viewport_y, viewport_w, viewport_h, format, GL_UNSIGNED_BYTE, pixels());

	if (!write_png(grayscale, viewport_w, viewport_h, pixels(), file()))
	{
		stream::cerr << "failure writing framegrab file '" << name << "'\n";
		return false;
	}

	return true;
}

#endif // FRAMEGRAB_RATE != 0

static inline float
wrap_at_period(
	const float x,
	const float period)
{
	const __m128 mask = _mm_cmpge_ss(_mm_set_ss(x), _mm_set_ss(period));
	return x - _mm_and_ps(mask, _mm_set_ss(period))[0];
}


static inline int32_t
reset_at_period(
	const int32_t x,
	const int32_t period)
{
	const __m128i mask = _mm_cmplt_epi32(_mm_cvtsi32_si128(x), _mm_cvtsi32_si128(period));
	return _mm_and_si128(mask, _mm_cvtsi32_si128(x))[0];
}

////////////////////////////////////////////////////////////////////////////////
// scene support
////////////////////////////////////////////////////////////////////////////////

class Scene
{
protected:
	// scene offset in model space
	float offset_x;
	float offset_y;
	float offset_z;

	// scene orientation
	float azim; // around z
	float decl; // around x
	float roll; // around y

	// scene camera position
	float cam_x;
	float cam_y;
	float cam_z;

public:
	Scene()
	: offset_x(0.f)
	, offset_y(0.f)
	, offset_z(0.f)
	, azim(0.f)
	, decl(0.f)
	, roll(0.f)
	, cam_x(0.f)
	, cam_y(0.f)
	, cam_z(0.f)
	{
	}

	virtual bool init(Timeslice& scene) = 0;
	virtual bool frame(Timeslice& scene, const float dt) = 0;

	// scene offset in model space
	float get_offset_x() const
	{
		return offset_x;
	}

	float get_offset_y() const
	{
		return offset_y;
	}

	float get_offset_z() const
	{
		return offset_z;
	}

	// scene orientation
	float get_azim() const
	{
		return azim;
	}

	float get_decl() const
	{
		return decl;
	}

	float get_roll() const
	{
		return roll;
	}

	// scene camera position
	float get_cam_x() const
	{
		return cam_x;
	}

	float get_cam_y() const
	{
		return cam_y;
	}

	float get_cam_z() const
	{
		return cam_z;
	}
};

////////////////////////////////////////////////////////////////////////////////
// Scene1: Deathstar Treadmill
////////////////////////////////////////////////////////////////////////////////

class Scene1 : public virtual Scene
{
	// scene camera properties
	float accum_x;
	float accum_y;

	enum {
		grid_rows = 20,
		grid_cols = 20,
		dist_unit = 1
	};

	float accum_time;
	float generation;

	Array< Voxel > content;

	bool update(
		Timeslice& scene,
		const float generation);

	void camera(
		const float dt);

public:
	// virtual from Scene
	bool init(
		Timeslice& scene);

	// virtual from Scene
	bool frame(
		Timeslice& scene,
		const float dt);
};


bool Scene1::init(
	Timeslice& scene)
{
	accum_x = 0.f;
	accum_y = 0.f;
	accum_time = 0.f;
	generation = grid_rows;

	if (!content.setCapacity(grid_rows * grid_cols))
		return false;

	const float unit = dist_unit;
	const float alt = unit * .5f;

	for (int y = 0; y < grid_rows; ++y)
		for (int x = 0; x < grid_cols; ++x)
		{
			content.addElement(Voxel(
				simd::vect3(x * unit,        y * unit,        0.f),
				simd::vect3(x * unit + unit, y * unit + unit, alt * (rand() % 4 + 1))));
		}

	return scene.set_payload_array(content);
}


inline bool Scene1::update(
	Timeslice& scene,
	const float generation)
{
	const float unit = dist_unit;
	const float alt = unit * .5f;
	size_t index = 0;

	for (index = 0; index < (grid_rows - 1) * grid_cols; ++index)
		content.getMutable(index) = content.getElement(index + grid_cols);

	const float y = generation;

	for (int x = 0; x < grid_cols; ++x, ++index)
	{
		content.getMutable(index) = Voxel(
			simd::vect3(x * unit,        y * unit,        0.f),
			simd::vect3(x * unit + unit, y * unit + unit, alt * (rand() % 4 + 1)));
	}

	return scene.set_payload_array(content);
}


inline void Scene1::camera(
	const float dt)
{
	const float period_x = 3.f; // seconds
	const float period_y = 2.f; // seconds

	const float deviate_x = 1.f / 32.f; // distance
	const float deviate_y = 1.f / 32.f; // distance

	const float roll_factor = 1 / 32.f; // of pi

	accum_x = wrap_at_period(accum_x + dt, period_x);
	accum_y = wrap_at_period(accum_y + dt, period_y);

	__m128 sin_xy;
	__m128 cos_x;
	sincos_ps(_mm_setr_ps(
		accum_x / period_x * float(M_PI * 2.0),
		accum_y / period_y * float(M_PI * 2.0), 0.f, 0.f),
		&sin_xy, &cos_x);

	cam_x = sin_xy[0] * deviate_x;
	cam_y = sin_xy[1] * deviate_y;
	roll  = cos_x[0] * float(M_PI * roll_factor);
}


bool Scene1::frame(
	Timeslice& scene,
	const float dt)
{
	camera(dt);

	const float update_period = .25f;

	offset_y -= dist_unit * dt / update_period;
	accum_time += dt;

	if (accum_time < update_period)
		return true;

	accum_time -= update_period;

	return update(scene, generation++);
}

////////////////////////////////////////////////////////////////////////////////
// Scene2: Sine Floater
////////////////////////////////////////////////////////////////////////////////

class Scene2 : virtual public Scene
{
	// scene camera properties
	float accum_x;
	float accum_y;

	enum {
		grid_rows = 20,
		grid_cols = 20,
		dist_unit = 1
	};

	float accum_time;

	Array< Voxel > content;

	bool update(
		Timeslice& scene,
		const float dt);

	void camera(
		const float dt);

public:
	// virtual from Scene
	bool init(
		Timeslice& scene);

	// virtual from Scene
	bool frame(
		Timeslice& scene,
		const float dt);
};


bool Scene2::init(
	Timeslice& scene)
{
	accum_x = 0.f;
	accum_y = 0.f;
	accum_time = 0.f;

	if (!content.setCapacity(grid_rows * grid_cols))
		return false;

	const float unit = dist_unit;

	for (int y = 0; y < grid_rows; ++y)
		for (int x = 0; x < grid_cols; ++x)
		{
			content.addElement(Voxel(
				simd::vect3(x * unit,        y * unit,        0.f),
				simd::vect3(x * unit + unit, y * unit + unit, 1.f)));

		}

	return scene.set_payload_array(content);
}


inline bool Scene2::update(
	Timeslice& scene,
	const float dt)
{
	const float period = 2.f; // seconds

	accum_time = wrap_at_period(accum_time + dt, period);

	const float time_factor = sin_ps(_mm_set_ss(accum_time / period * float(M_PI * 2.0)))[0];

	const float unit = dist_unit;
	const float alt = unit * .5f;
	size_t index = 0;

	for (int y = 0; y < grid_rows; ++y)
		for (int x = 0; x < grid_cols; ++x, ++index)
		{
			const __m128 sin_xy = sin_ps(_mm_setr_ps(x * alt, y * alt, 0.f, 0.f));

			content.getMutable(index) = Voxel(
				simd::vect3(x * unit,        y * unit,        0.f),
				simd::vect3(x * unit + unit, y * unit + unit, 1.f + time_factor * unit * (sin_xy[0] * sin_xy[1])));
		}

	return scene.set_payload_array(content);
}


inline void Scene2::camera(
	const float dt)
{
	const float period_x = 3.f; // seconds
	const float period_y = 2.f; // seconds

	const float deviate_x = 1.f / 32.f; // distance
	const float deviate_y = 1.f / 32.f; // distance

	const float roll_factor = 1 / 32.f; // of pi

	accum_x = wrap_at_period(accum_x + dt, period_x);
	accum_y = wrap_at_period(accum_y + dt, period_y);

	__m128 sin_xy;
	__m128 cos_x;
	sincos_ps(_mm_setr_ps(
		accum_x / period_x * float(M_PI * 2.0),
		accum_y / period_y * float(M_PI * 2.0), 0.f, 0.f),
		&sin_xy, &cos_x);

	cam_x = sin_xy[0] * deviate_x;
	cam_y = sin_xy[1] * deviate_y;
	roll  = cos_x[0] * float(M_PI * roll_factor);
}


bool Scene2::frame(
	Timeslice& scene,
	const float dt)
{
	camera(dt);

	return update(scene, dt);
}

////////////////////////////////////////////////////////////////////////////////
// Scene3: Serpents
////////////////////////////////////////////////////////////////////////////////

class Scene3 : virtual public Scene
{
	// scene camera properties
	float accum_x;
	float accum_y;

	enum {
		queue_length = 96,
		main_radius = 8,
	};

	float accum_time;

	Array< Voxel > content;

	bool update(
		Timeslice& scene,
		const float dt);

public:
	// virtual from Scene
	bool init(
		Timeslice& scene);

	// virtual from Scene
	bool frame(
		Timeslice& scene,
		const float dt);
};


bool Scene3::init(
	Timeslice& scene)
{
	offset_x = 10.f; // matching the center of scene_1
	offset_y = 10.f; // matching the center of scene_1

	accum_x = 0.f;
	accum_y = 0.f;
	accum_time = 0.f;

	if (!content.setCapacity(queue_length * 2 + 1))
		return false;

	const float radius = main_radius;

	for (int i = 0; i < queue_length; ++i)
	{
		const float angle = float(M_PI * 2.0) * (i / float(queue_length));
		const float sin_i = sin_ps(_mm_set_ss(i / float(queue_length) * float(M_PI * 16.0)))[0];

		{
			const matx3_rotate rot(angle, 0.f, 0.f, 1.f);

			const simd::vect3 pos = simd::vect3(radius + sin_i, 0.f, sin_i).mul(rot);

			content.addElement(Voxel(
				simd::vect3().sub(pos, simd::vect3(.5f, .5f, .5f)),
				simd::vect3().add(pos, simd::vect3(.5f, .5f, .5f))));
		}
		{
			const matx3_rotate rot(-angle, 0.f, 0.f, 1.f);

			const simd::vect3 pos = simd::vect3(radius * .5f + sin_i, 0.f, sin_i).mul(rot);

			content.addElement(Voxel(
				simd::vect3().sub(pos, simd::vect3(.25f, .25f, .25f)),
				simd::vect3().add(pos, simd::vect3(.25f, .25f, .25f))));
		}
	}

	content.addElement(Voxel(
		simd::vect3(-main_radius, -main_radius, -.25f),
		simd::vect3(+main_radius, +main_radius, +.25f)));

	return scene.set_payload_array(content);
}


inline bool Scene3::update(
	Timeslice& scene,
	const float dt)
{
	const float period = 32.f; // seconds

	accum_time = wrap_at_period(accum_time + dt, period);

	const float radius = main_radius;
	size_t index = 0;

	for (int i = 0; i < queue_length; ++i, index += 2)
	{
		const float angle = float(M_PI * 2.0) * (i / float(queue_length)) + accum_time * float(M_PI * 2.0) / period;
		const __m128 sin_iz = sin_ps(_mm_setr_ps(
			i / float(queue_length) * float(M_PI * 16.0),
			i / float(queue_length) * float(M_PI * 16.0) + accum_time * float(M_PI * 32.0) / period, 0.f, 0.f));
		const float sin_i = sin_iz[0];
		const float sin_z = sin_iz[1];

		{
			const matx3_rotate rot(angle, 0.f, 0.f, 1.f);

			const simd::vect3 pos = simd::vect3(radius + sin_i, 0.f, sin_z).mul(rot);

			content.getMutable(index + 0) = Voxel(
				simd::vect3().sub(pos, simd::vect3(.5f, .5f, .5f)),
				simd::vect3().add(pos, simd::vect3(.5f, .5f, .5f)));
		}
		{
			const matx3_rotate rot(-angle, 0.f, 0.f, 1.f);

			const simd::vect3 pos = simd::vect3(radius * .5f + sin_i, 0.f, sin_z).mul(rot);

			content.getMutable(index + 1) = Voxel(
				simd::vect3().sub(pos, simd::vect3(.35f, .35f, .35f)),
				simd::vect3().add(pos, simd::vect3(.35f, .35f, .35f)));
		}
	}

	return scene.set_payload_array(content);
}


bool Scene3::frame(
	Timeslice& scene,
	const float dt)
{
	return update(scene, dt);
}


enum {
	scene_1,
	scene_2,
	scene_3,

	scene_count
};

////////////////////////////////////////////////////////////////////////////////
// the global control state
////////////////////////////////////////////////////////////////////////////////

namespace c
{
	static const float beat_period = 1.714288; // seconds

	static size_t scene_selector;

	// scene properties
	static float decl = 0.f;
	static float azim = 0.f;

	// camera properties
	static float pos_x = 0.f;
	static float pos_y = 0.f;
	static float pos_z = 0.f;

	// accumulators
	static float accum_time = 0.f;
	static float accum_beat = 0.f;
	static float accum_beat_2 = 0.f;

	// view properties
	static float contrast_middle = .5f;
	static float contrast_k = 1.f;
	static float blur_split = -1.f;

} // namespace c

////////////////////////////////////////////////////////////////////////////////
// scripting support
////////////////////////////////////////////////////////////////////////////////

class Action
{
protected:
	float lifespan;

public:
	// start the action; return true if action is alive after the start
	virtual bool start(
		const float delay,
		const float duration) = 0;

	// perform action at tick (ie. at frame); return true if action still alive
	virtual bool frame(
		const float dt) = 0;
};

////////////////////////////////////////////////////////////////////////////////
// ActionSetScene1: establish scene_1
////////////////////////////////////////////////////////////////////////////////

class ActionSetScene1 : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionSetScene1::start(
	const float,
	const float)
{
	c::scene_selector = scene_1;

	c::pos_x = 0.f;
	c::pos_y = .25f;
	c::pos_z = .875f;

	c::decl = M_PI / -2.0;

	return false;
}


bool ActionSetScene1::frame(
	const float)
{
	return false;
}

////////////////////////////////////////////////////////////////////////////////
// ActionSetScene2: establish scene_2
////////////////////////////////////////////////////////////////////////////////

class ActionSetScene2 : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionSetScene2::start(
	const float,
	const float)
{
	c::scene_selector = scene_2;

	c::pos_x = 0.f;
	c::pos_y = .25f;
	c::pos_z = 1.f;

	c::decl = M_PI / -2.0;

	return false;
}


bool ActionSetScene2::frame(
	const float)
{
	return false;
}

////////////////////////////////////////////////////////////////////////////////
// ActionSetScene3: establish scene_3
////////////////////////////////////////////////////////////////////////////////

class ActionSetScene3 : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionSetScene3::start(
	const float,
	const float)
{
	c::scene_selector = scene_3;

	c::pos_x = 0.f;
	c::pos_y = 0.f;
	c::pos_z = 1.125f;

	c::decl = M_PI / -2.125;

	return false;
}


bool ActionSetScene3::frame(
	const float)
{
	return false;
}

////////////////////////////////////////////////////////////////////////////////
// ActionViewBlur: blur the view
////////////////////////////////////////////////////////////////////////////////

class ActionViewBlur : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionViewBlur::start(
	const float,
	const float)
{
	c::blur_split = -1.f;

	return false;
}


bool ActionViewBlur::frame(
	const float)
{
	return false;
}

////////////////////////////////////////////////////////////////////////////////
// ActionViewUnblur: de-blur the view
////////////////////////////////////////////////////////////////////////////////

class ActionViewUnblur : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionViewUnblur::start(
	const float,
	const float)
{
	c::blur_split = 1.f;

	return false;
}


bool ActionViewUnblur::frame(
	const float)
{
	return false;
}

////////////////////////////////////////////////////////////////////////////////
// ActionViewBlurDt: blur the view non-instantaneously
////////////////////////////////////////////////////////////////////////////////

class ActionViewBlurDt : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionViewBlurDt::start(
	const float delay,
	const float)
{
	c::blur_split -= (2.0 / c::beat_period) * delay;

	if (c::blur_split > -1.f)
		return true;

   c::blur_split = -1.f;
   return false;
}


bool ActionViewBlurDt::frame(
	const float dt)
{
	c::blur_split -= (2.0 / c::beat_period) * dt;

	if (c::blur_split > -1.f)
		return true;

	c::blur_split = -1.f;
	return false;
}

////////////////////////////////////////////////////////////////////////////////
// ActionViewUnblurDt: de-blur the view non-instantaneously
////////////////////////////////////////////////////////////////////////////////

class ActionViewUnblurDt : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionViewUnblurDt::start(
	const float delay,
	const float)
{
	c::blur_split += (2.0 / c::beat_period) * delay;

	if (c::blur_split < 1.f)
		return true;

	c::blur_split = 1.f;
	return false;
}


bool ActionViewUnblurDt::frame(
	const float dt)
{
	c::blur_split += (2.0 / c::beat_period) * dt;

	if (c::blur_split < 1.f)
		return true;

	c::blur_split = 1.f;
	return false;
}

////////////////////////////////////////////////////////////////////////////////
// ActionViewSplit: split view in sync to the beat
////////////////////////////////////////////////////////////////////////////////

class ActionViewSplit : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionViewSplit::start(
	const float delay,
	const float duration)
{
	if (delay >= duration)
		return false;

	c::blur_split = sin_ps(_mm_set_ss(c::accum_beat * float(M_PI * 2.0 / c::beat_period)))[0] * .25f;

	lifespan = duration - delay;
	return true;
}


bool ActionViewSplit::frame(
	const float dt)
{
	if (dt >= lifespan)
		return false;

	c::blur_split = sin_ps(_mm_set_ss(c::accum_beat * float(M_PI * 2.0 / c::beat_period)))[0] * .25f;

	lifespan -= dt;
	return true;
}

////////////////////////////////////////////////////////////////////////////////
// ActionContrastBeat: pulse contrast to the beat
////////////////////////////////////////////////////////////////////////////////

class ActionContrastBeat : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionContrastBeat::start(
	const float delay,
	const float duration)
{
	if (delay >= duration)
		return false;

	c::contrast_middle = .5f;
	c::contrast_k = 1.f + pow_ps(sin_ps(_mm_set_ss(_mm_andnot_ps(_mm_set_ss(-0.f), _mm_set_ss(-.5f + c::accum_beat / c::beat_period))[0] * float(M_PI))), _mm_set1_ps(64.f))[0];

	lifespan = duration - delay;
	return true;
}


bool ActionContrastBeat::frame(
	const float dt)
{
	if (dt >= lifespan)
	{
		c::contrast_middle = .5f;
		c::contrast_k = 1.f;
		return false;
	}

	c::contrast_k = 1.f + pow_ps(sin_ps(_mm_set_ss(_mm_andnot_ps(_mm_set_ss(-0.f), _mm_set_ss(-.5f + c::accum_beat / c::beat_period))[0] * float(M_PI))), _mm_set1_ps(64.f))[0];

	lifespan -= dt;
	return true;
}

////////////////////////////////////////////////////////////////////////////////
// ActionCameraSnake: snake-style camera azimuth (singe quadrant only)
////////////////////////////////////////////////////////////////////////////////

class ActionCameraSnake : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionCameraSnake::start(
	const float delay,
	const float duration)
{
	if (delay >= duration)
		return false;

	c::azim += sin_ps(_mm_set_ss(c::accum_beat_2 * float(M_PI / c::beat_period)))[0] * float(M_PI / 4.0) * delay;

	lifespan = duration - delay;
	return true;
}


bool ActionCameraSnake::frame(
	const float dt)
{
	if (dt >= lifespan)
		return false;

	c::azim += sin_ps(_mm_set_ss(c::accum_beat_2 * float(M_PI / c::beat_period)))[0] * float(M_PI / 4.0) * dt;

	lifespan -= dt;
	return true;
}

////////////////////////////////////////////////////////////////////////////////
// ActionCameraBounce: snake-style camera azimuth (full range)
////////////////////////////////////////////////////////////////////////////////

class ActionCameraBounce : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionCameraBounce::start(
	const float delay,
	const float duration)
{
	if (delay >= duration)
		return false;

	c::azim += cos_ps(_mm_set_ss(c::accum_beat_2 * float(M_PI / c::beat_period)))[0] * float(M_PI / 4.0) * delay;

	lifespan = duration - delay;
	return true;
}


bool ActionCameraBounce::frame(
	const float dt)
{
	if (dt >= lifespan)
		return false;

	c::azim += cos_ps(_mm_set_ss(c::accum_beat_2 * float(M_PI / c::beat_period)))[0] * float(M_PI / 4.0) * dt;

	lifespan -= dt;
	return true;
}

////////////////////////////////////////////////////////////////////////////////
// ActionCameraBnF: camera position back'n'forth
////////////////////////////////////////////////////////////////////////////////

class ActionCameraBnF : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionCameraBnF::start(
	const float delay,
	const float duration)
{
	if (delay >= duration)
		return false;

	c::pos_z += cos_ps(_mm_set_ss(c::accum_beat_2 * float(M_PI / c::beat_period)))[0] * delay;

	lifespan = duration - delay;
	return true;
}


bool ActionCameraBnF::frame(
	const float dt)
{
	if (dt >= lifespan)
		return false;

	c::pos_z += cos_ps(_mm_set_ss(c::accum_beat_2 * float(M_PI / c::beat_period)))[0] * dt;

	lifespan -= dt;
	return true;
}

////////////////////////////////////////////////////////////////////////////////
// ActionCameraLean: camera leaning forth then back
////////////////////////////////////////////////////////////////////////////////

class ActionCameraLean : virtual public Action
{
public:
	// virtual from Action
	bool start(const float, const float);

	// virtual from Action
	bool frame(const float);
};


bool ActionCameraLean::start(
	const float delay,
	const float duration)
{
	if (delay >= duration)
		return false;

	c::decl += sin_ps(_mm_set_ss(c::accum_beat_2 * float(M_PI / c::beat_period)))[0] * delay;

	lifespan = duration - delay;
	return true;
}


bool ActionCameraLean::frame(
	const float dt)
{
	if (dt >= lifespan)
		return false;

	c::decl += sin_ps(_mm_set_ss(c::accum_beat_2 * float(M_PI / c::beat_period)))[0] * dt;

	lifespan -= dt;
	return true;
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

	unsigned fsaa = 0;
	unsigned w = default_w;
	unsigned h = default_h;
	unsigned bitness[4] = { 0 };
	unsigned frames = -1U;

	const int result_cli = parse_cli(
		argc,
		argv,
		bitness,
		w,
		h,
		fsaa,
		frames);

	if (0 != result_cli)
		return result_cli;

#if DIVISION_OF_LABOR_VER == 2
	if (w * h % (nthreads * batch))
	{
		stream::cerr << "error: screen resolution product must be multiple of " << nthreads * batch << '\n';
		return -1;
	}

#elif DIVISION_OF_LABOR_VER == 1
	if (w * h % batch)
	{
		stream::cerr << "error: screen resolution product must be multiple of " << batch << '\n';
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
#if defined(prob_4_H__)
	Array< Timeslice > timeline;

#elif defined(prob_7_H__)
	Array< TimesliceBalloon, 4096 > timeline;

#else
	#error prob_4_H__ or prob_7_H__ required

#endif
	timeline.setCapacity(scene_count);
	timeline.addMultiElement(scene_count);

	Scene1 scene1;

	if (!scene1.init(timeline.getMutable(scene_1)))
		return 1;

	Scene2 scene2;

	if (!scene2.init(timeline.getMutable(scene_2)))
		return 2;

	Scene3 scene3;

	if (!scene3.init(timeline.getMutable(scene_3)))
		return 3;

	Scene* const scene[] = {
		&scene1,
		&scene2,
		&scene3
	};

	ActionSetScene1    actionSetScene1;
	ActionSetScene2    actionSetScene2;
	ActionSetScene3    actionSetScene3;
	ActionContrastBeat actionContrastBeat;
	ActionViewBlur     actionViewBlur;
	ActionViewUnblur   actionViewUnblur;
	ActionViewBlurDt   actionViewBlurDt;
	ActionViewUnblurDt actionViewUnblurDt;
	ActionViewSplit    actionViewSplit;
	ActionCameraSnake  actionCameraSnake;
	ActionCameraBounce actionCameraBounce;
	ActionCameraBnF    actionCameraBnF;
	ActionCameraLean   actionCameraLean;

	// master track of the application (entries sorted by start time)
	const struct
	{
		const float start;    // seconds
		const float duration; // seconds
		Action& action;
	}
	track[] = {
		{   0.f,        0.f,      actionSetScene1 },
		{   0.f,        9.428584, actionViewSplit },
		{   0.f,       68.571342, actionContrastBeat },
		{   9.428584,   0.f,      actionViewBlurDt },
		{  68.571342,   0.f,      actionSetScene2 },
		{  68.571342,   0.f,      actionViewUnblur },
		{  68.571342,  27.426758, actionCameraBounce },
		{  82.285824,  6.8551240, actionCameraLean },
		{  95.998100,   0.f,      actionSetScene3 },
		{  95.998100,  39.430652, actionCameraSnake },
		{ 109.714432,  25.714320, actionCameraBnF },
		{ 133.714464,   0.f,      actionViewBlurDt },
		{ 135.428752,   0.f,      actionSetScene1 },
		{ 136.285896,  56.571504, actionContrastBeat },
		{ 198.f,        0.f,      actionViewUnblurDt }
	};

	size_t track_cursor = 0;

	// live actions
	Action* action[8];

	size_t action_count = 0;

	// use first scene's initial world bbox to compute a normalization (pan_n_zoom) matrix
	const BBox& world_bbox = timeline.getElement(scene_1).get_root_bbox();

	const __m128 bbox_min = world_bbox.get_min();
	const __m128 bbox_max = world_bbox.get_max();
	const __m128 centre = _mm_mul_ps(
		_mm_add_ps(bbox_max, bbox_min),
		_mm_set1_ps(.5f));
	const __m128 extent = _mm_mul_ps(
		_mm_sub_ps(bbox_max, bbox_min),
		_mm_set1_ps(.5f));
	const float rcp_extent = 1.f / std::max(extent[0], std::max(extent[1], extent[2]));

	glEnable(GL_CULL_FACE);

	const testbed::scoped_ptr< uint8_t[4], generic_free > unaligned_fb(
		reinterpret_cast< uint8_t(*)[4] >(malloc(w * h * sizeof(uint8_t[4]) + 63)));

	// get that buffer cacheline aligned
	uint8_t (* const framebuffer)[4] = reinterpret_cast< uint8_t(*)[4] >(uintptr_t(unaligned_fb()) + uintptr_t(63) & ~uintptr_t(63));
	memset(framebuffer, 0, w * h * sizeof(*framebuffer));

	workforce_t workforce(framebuffer, w, h);

	if (!workforce.is_successfully_init())
	{
		stream::cerr << "failed to raise workforce; bailing out\n";
		return -1;
	}

	GLuint input = 0;
	unsigned nframes = 0;
	const uint64_t t0 = timer_nsec();
	uint64_t tlast = t0;

#if VISUALIZE != 0
	while (testbed::processEvents(input) && nframes != frames)

#else
	while (nframes != frames)

#endif
	{
#if FRAMEGRAB_RATE != 0
		const float dt = 1.0 / FRAMEGRAB_RATE;

#else
		const uint64_t tframe = timer_nsec();
		const float dt = double(tframe - tlast) * 1e-9;
		tlast = tframe;

#endif

		// upate run time (we aren't supposed to run long - fp32 should do) and beat time
		c::accum_time += dt;
		c::accum_beat   = wrap_at_period(c::accum_beat   + dt, c::beat_period);
		c::accum_beat_2 = wrap_at_period(c::accum_beat_2 + dt, c::beat_period * 2.0);

		// run all live actions, retiring the completed ones
		for (size_t i = 0; i < action_count; ++i)
			if (!action[i]->frame(dt))
				action[i--] = action[--action_count];

		// start any pending actions
		for (; track_cursor < sizeof(track) / sizeof(track[0]) && c::accum_time >= track[track_cursor].start; ++track_cursor)
			if (track[track_cursor].action.start(c::accum_time - track[track_cursor].start, track[track_cursor].duration))
			{
				if (action_count == sizeof(action) / sizeof(action[0]))
				{
					stream::cerr << "error: too many pending actions\n";
					return 999;
				}

				action[action_count++] = &track[track_cursor].action;
			}

		// run the live scene
		scene[c::scene_selector]->frame(timeline.getMutable(c::scene_selector), dt);

		const simd::matx4 pan_n_zoom(
			rcp_extent, 0.f, 0.f, 0.f,
			0.f, rcp_extent, 0.f, 0.f,
			0.f, 0.f, rcp_extent, 0.f,
			(scene[c::scene_selector]->get_offset_x() - centre[0]) * rcp_extent,
			(scene[c::scene_selector]->get_offset_y() - centre[1]) * rcp_extent,
			(scene[c::scene_selector]->get_offset_z() - centre[2]) * rcp_extent, 1.f);

		// obligatory manual controls
		if (input & testbed::INPUT_MASK_ACTION)
		{
			input &= ~testbed::INPUT_MASK_ACTION;

			c::scene_selector = reset_at_period(c::scene_selector + 1, scene_count);
		}

		if (input & testbed::INPUT_MASK_OPTION_1)
		{
			input &= ~testbed::INPUT_MASK_OPTION_1;

			c::blur_split *= -1.f;
		}

		const float angular_step = float(M_PI_4) * dt;

		if (input & testbed::INPUT_MASK_UP)
			c::decl += angular_step;

		if (input & testbed::INPUT_MASK_DOWN)
			c::decl -= angular_step;

		if (input & testbed::INPUT_MASK_LEFT)
			c::azim += angular_step;

		if (input & testbed::INPUT_MASK_RIGHT)
			c::azim -= angular_step;

		const float linear_step = .25f * dt;

		if (input & testbed::INPUT_MASK_ALT_UP)
			c::pos_z += linear_step;

		if (input & testbed::INPUT_MASK_ALT_DOWN)
			c::pos_z -= linear_step;

		if (input & testbed::INPUT_MASK_ALT_LEFT)
			c::pos_x -= linear_step;

		if (input & testbed::INPUT_MASK_ALT_RIGHT)
			c::pos_x += linear_step;

		c::azim = wrap_at_period(c::azim, float(M_PI * 2.0));

		const simd::vect4 eye(
			-(scene[c::scene_selector]->get_cam_x() + c::pos_x),
			-(scene[c::scene_selector]->get_cam_y() + c::pos_y),
			-(scene[c::scene_selector]->get_cam_z() + c::pos_z), 1.f);

		simd::matx4 rot = simd::matx4(matx4_rotate(scene[c::scene_selector]->get_roll(), 0.f, 1.f, 0.f));
		rot.mulr(matx4_rotate(scene[c::scene_selector]->get_azim() + c::azim, 0.f, 0.f, 1.f));
		rot.mulr(matx4_rotate(scene[c::scene_selector]->get_decl() + c::decl, 1.f, 0.f, 0.f));
		rot.set(3, eye);

		const simd::matx4 post_op = simd::matx4().mul(pan_n_zoom, rot);

		const simd::matx4 mv_inv = simd::matx4().inverse(post_op);
		const simd::vect3 cam[] =
		{
			simd::vect3(mv_inv[0][0], mv_inv[0][1], mv_inv[0][2]),
			simd::vect3(mv_inv[1][0], mv_inv[1][1], mv_inv[1][2]).mul(float(h) / w),
			simd::vect3(mv_inv[2][0], mv_inv[2][1], mv_inv[2][2]).negate(),
			simd::vect3(mv_inv[3][0], mv_inv[3][1], mv_inv[3][2])
		};

		workforce.update(nframes, cam, timeline.getElement(c::scene_selector));

#if DIVISION_OF_LABOR_VER == 2
		for (size_t i = 0; i < nthreads; ++i)
			worker[i].cursor = 0;

#elif DIVISION_OF_LABOR_VER == 1
		workgroup_cursor = 0;

#endif
		compute_arg carg(0, nframes, cam, timeline.getElement(c::scene_selector), framebuffer, w, h);
		compute(&carg);

#if VISUALIZE != 0
		testbed::rgbv::render(framebuffer, c::contrast_middle, c::contrast_k, c::blur_split);

#if FRAMEGRAB_RATE != 0
		saveViewport(0, 0, w, h, nframes);

#endif
		testbed::swapBuffers();

#endif
		++nframes;
	}

	const uint64_t dt = timer_nsec() - t0;

	stream::cout << "compute_arg size: " << sizeof(compute_arg) <<
		"\nworker threads: " << nthreads << "\nambient occlusion rays per pixel: " << ao_probe_count <<
		"\ntotal frames rendered: " << nframes << '\n';

	if (dt)
	{
		const double sec = double(dt) * 1e-9;

		stream::cout << "elapsed time: " << sec << " s"
			"\naverage FPS: " << nframes / sec << '\n';
	}

	return 0;
}
