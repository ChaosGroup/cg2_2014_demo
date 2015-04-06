#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <time.h>
#include <stdint.h>
#include <pthread.h>
#if __SSE__
#include <xmmintrin.h>
#endif

#if SIMD_SCALAR
	#include "vectscal.hpp"
#else
	#if SIMD_AGNOSTIC
		#include "vectsimd.hpp"
	#elif __SSE__
		#include "vectsimd_sse.hpp"
	#else
		#include "vectsimd.hpp"
	#endif
#endif

// SIMD_AUTOVECT enum
#define SIMD_1WAY		1 // no SIMD; use scalars instead (compiler might still autovectorize!)
#define SIMD_2WAY		2
#define SIMD_4WAY		3
#define SIMD_8WAY		4
#define SIMD_16WAY		5

// SIMD_INTRINSICS enum
#define SIMD_ALTIVEC	1
#define SIMD_SSE		2
#define SIMD_NEON		3
#define SIMD_MIC		4

#undef SIMD_INTRINSICS
#if SIMD_AUTOVECT == 0
	#if __ALTIVEC__ == 1
		#define SIMD_INTRINSICS SIMD_ALTIVEC
	#elif __SSE__ == 1
		#define SIMD_INTRINSICS SIMD_SSE
	#elif __ARM_NEON__ == 1
		#define SIMD_INTRINSICS SIMD_NEON
	#elif __MIC__ == 1
		#define SIMD_INTRINSICS SIMD_MIC
	#endif
#endif

#if SIMD_INTRINSICS == SIMD_ALTIVEC
	#include <altivec.h>
#elif SIMD_INTRINSICS == SIMD_SSE
	#if __AVX__
		#include <immintrin.h>
	#elif __SSE4_1__
		#include <smmintrin.h>
	#else
		#include <xmmintrin.h>
	#endif
#elif SIMD_INTRINSICS == SIMD_NEON
	#include <arm_neon.h>
#elif SIMD_INTRINSICS == SIMD_MIC
	#include <immintrin.h>
#endif


#undef SIMD_NAMESPACE

#if SIMD_ETALON
#define SIMD_NAMESPACE etal

#elif SIMD_SCALAR
#define SIMD_NAMESPACE scal

namespace scal
{
typedef ivect< 2, int32_t > ivect2;
typedef ivect< 3, int32_t > ivect3;
typedef ivect< 4, int32_t > ivect4;

typedef vect< 2, float > vect2;
typedef vect< 3, float > vect3;
typedef vect< 4, float > vect4;

typedef hamilton< float > quat;

typedef matx< 3, float > matx3;
typedef matx< 4, float > matx4;

} // namespace scal

#else
#define SIMD_NAMESPACE simd

#endif

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

namespace etal
{

class __attribute__ ((aligned(SIMD_ALIGNMENT))) matx4
{
#if SIMD_types_for_the_autovectorizer
#error

#elif SIMD_AUTOVECT == SIMD_2WAY
	typedef __attribute__ ((vector_size(2 * sizeof(float)))) float vect2_float;

#elif SIMD_AUTOVECT == SIMD_4WAY
	typedef __attribute__ ((vector_size(4 * sizeof(float)))) float vect4_float;

#elif SIMD_AUTOVECT == SIMD_8WAY
	typedef __attribute__ ((vector_size(8 * sizeof(float)))) float vect8_float;

#elif SIMD_AUTOVECT == SIMD_16WAY
	typedef __attribute__ ((vector_size(16 * sizeof(float)))) float vect16_float;

#endif // SIMD_types_for_the_autovectorizer

	union row
	{
		float m[4];

#if SIMD_aliases_to_the_above_C_array
#error

#elif SIMD_AUTOVECT == SIMD_2WAY
		vect2_float n[2];

#elif SIMD_AUTOVECT == SIMD_4WAY
		vect4_float n;

#elif SIMD_AUTOVECT == SIMD_8WAY
		vect8_float n;

#elif SIMD_AUTOVECT == SIMD_16WAY
		vect16_float n;

#elif SIMD_INTRINSICS == SIMD_ALTIVEC
		vector float n;

#elif SIMD_INTRINSICS == SIMD_SSE
		__m128 n;

#elif SIMD_INTRINSICS == SIMD_MIC
#if SIMD_ALIGNMENT == SIMD_ALIGNMENT / 64 * 64
		__m512 n;
#endif

#elif SIMD_INTRINSICS == SIMD_NEON
		float32x4_t n;

#endif // SIMD_aliases_to_the_above_C_array
	};

	row m[4];

public:
	matx4()
	{
	}

	matx4(
		const float m00, const float m01, const float m02, const float m03,
		const float m10, const float m11, const float m12, const float m13,
		const float m20, const float m21, const float m22, const float m23,
		const float m30, const float m31, const float m32, const float m33)
	{
		m[0].m[0] = m00;
		m[0].m[1] = m01;
		m[0].m[2] = m02;
		m[0].m[3] = m03;

		m[1].m[0] = m10;
		m[1].m[1] = m11;
		m[1].m[2] = m12;
		m[1].m[3] = m13;

		m[2].m[0] = m20;
		m[2].m[1] = m21;
		m[2].m[2] = m22;
		m[2].m[3] = m23;

		m[3].m[0] = m30;
		m[3].m[1] = m31;
		m[3].m[2] = m32;
		m[3].m[3] = m33;
	}

	matx4(
		const float (&src)[4][4])
	{
		m[0].m[0] = src[0][0];
		m[0].m[1] = src[0][1];
		m[0].m[2] = src[0][2];
		m[0].m[3] = src[0][3];

		m[1].m[0] = src[1][0];
		m[1].m[1] = src[1][1];
		m[1].m[2] = src[1][2];
		m[1].m[3] = src[1][3];

		m[2].m[0] = src[2][0];
		m[2].m[1] = src[2][1];
		m[2].m[2] = src[2][2];
		m[2].m[3] = src[2][3];

		m[3].m[0] = src[3][0];
		m[3].m[1] = src[3][1];
		m[3].m[2] = src[3][2];
		m[3].m[3] = src[3][3];
	}

	float get(
		const size_t i,
		const size_t j) const
	{
		return m[i].m[j];
	}

	inline __attribute__ ((always_inline)) matx4&
	mul(
		const matx4& mat0,
		const matx4& mat1)
	{
		for (unsigned i = 0; i < 4; ++i)
		{
			const float e0 = mat0.m[i].m[0];

#if SIMD_preamble_per_output_row
#error

#elif SIMD_INTRINSICS == SIMD_ALTIVEC
			m[i].n = vec_madd(vec_splat(mat0.m[i].n, 0), mat1.m[0].n, (vector float) { -0.f, -0.f, -0.f -0.f });
			m[i].n = vec_madd(vec_splat(mat0.m[i].n, 1), mat1.m[1].n, m[i].n);
			m[i].n = vec_madd(vec_splat(mat0.m[i].n, 2), mat1.m[2].n, m[i].n);
			m[i].n = vec_madd(vec_splat(mat0.m[i].n, 3), mat1.m[3].n, m[i].n);

#elif SIMD_INTRINSICS == SIMD_SSE
#if 1
			m[i].n =            _mm_mul_ps(_mm_shuffle_ps(mat0.m[i].n, mat0.m[i].n, _MM_SHUFFLE(0, 0, 0, 0)), mat1.m[0].n);
			m[i].n = _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(mat0.m[i].n, mat0.m[i].n, _MM_SHUFFLE(1, 1, 1, 1)), mat1.m[1].n), m[i].n);
			m[i].n = _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(mat0.m[i].n, mat0.m[i].n, _MM_SHUFFLE(2, 2, 2, 2)), mat1.m[2].n), m[i].n);
			m[i].n = _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(mat0.m[i].n, mat0.m[i].n, _MM_SHUFFLE(3, 3, 3, 3)), mat1.m[3].n), m[i].n);

#else
			m[i].n =            _mm_mul_ps(_mm_broadcast_ss(mat0.m[i].m + 0), mat1.m[0].n);
			m[i].n = _mm_add_ps(_mm_mul_ps(_mm_broadcast_ss(mat0.m[i].m + 1), mat1.m[1].n), m[i].n);
			m[i].n = _mm_add_ps(_mm_mul_ps(_mm_broadcast_ss(mat0.m[i].m + 2), mat1.m[2].n), m[i].n);
			m[i].n = _mm_add_ps(_mm_mul_ps(_mm_broadcast_ss(mat0.m[i].m + 3), mat1.m[3].n), m[i].n);

#endif

#elif SIMD_INTRINSICS == SIMD_MIC
#if SIMD_ALIGNMENT == SIMD_ALIGNMENT / 64 * 64
			m[i].n = _mm512_mask_mul_ps(m[i].n, __mmask16(15), _mm512_swizzle_ps(mat0.m[i].n, _MM_SWIZ_REG_AAAA), mat1.m[0].n);
			m[i].n =                     _mm512_mask3_fmadd_ps(_mm512_swizzle_ps(mat0.m[i].n, _MM_SWIZ_REG_BBBB), mat1.m[1].n, m[i].n, __mmask16(15));
			m[i].n =                     _mm512_mask3_fmadd_ps(_mm512_swizzle_ps(mat0.m[i].n, _MM_SWIZ_REG_CCCC), mat1.m[2].n, m[i].n, __mmask16(15));
			m[i].n =                     _mm512_mask3_fmadd_ps(_mm512_swizzle_ps(mat0.m[i].n, _MM_SWIZ_REG_DDDD), mat1.m[3].n, m[i].n, __mmask16(15));

#elif SIMD_ALIGNMENT == 16
			const __m512 vec = _mm512_extload_ps(mat0.m[i].m, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NONE);
			__m512 op0 = _mm512_undefined_ps();
			__m512 op1 = _mm512_undefined_ps();
			__m512 op2 = _mm512_undefined_ps();
			__m512 op3 = _mm512_undefined_ps();
			op0 = _mm512_mask_loadunpacklo_ps(op0, __mmask16(15), mat1.m + 0);
			op1 = _mm512_mask_loadunpacklo_ps(op1, __mmask16(15), mat1.m + 1);
			op2 = _mm512_mask_loadunpacklo_ps(op2, __mmask16(15), mat1.m + 2);
			op3 = _mm512_mask_loadunpacklo_ps(op3, __mmask16(15), mat1.m + 3);
			__m512 res = _mm512_undefined_ps();
			res = _mm512_mask_mul_ps(res, __mmask16(15), _mm512_swizzle_ps(vec, _MM_SWIZ_REG_AAAA), op0);
			res =                  _mm512_mask3_fmadd_ps(_mm512_swizzle_ps(vec, _MM_SWIZ_REG_BBBB), op1, res, __mmask16(15));
			res =                  _mm512_mask3_fmadd_ps(_mm512_swizzle_ps(vec, _MM_SWIZ_REG_CCCC), op2, res, __mmask16(15));
			res =                  _mm512_mask3_fmadd_ps(_mm512_swizzle_ps(vec, _MM_SWIZ_REG_DDDD), op3, res, __mmask16(15));
			_mm512_mask_packstorelo_ps(m[i].m, __mmask16(15), res);

#else
#error unsupported SIMD_ALIGNMENT

#endif

#elif SIMD_INTRINSICS == SIMD_NEON
			m[i].n = vmulq_n_f32(        mat1.m[0].n, vgetq_lane_f32(mat0.m[i].n, 0));
			m[i].n = vmlaq_n_f32(m[i].n, mat1.m[1].n, vgetq_lane_f32(mat0.m[i].n, 1));
			m[i].n = vmlaq_n_f32(m[i].n, mat1.m[2].n, vgetq_lane_f32(mat0.m[i].n, 2));
			m[i].n = vmlaq_n_f32(m[i].n, mat1.m[3].n, vgetq_lane_f32(mat0.m[i].n, 3));

#elif SIMD_AUTOVECT == SIMD_2WAY
			m[i].n[0] = (vect2_float) { e0, e0 } * mat1.m[0].n[0];
			m[i].n[1] = (vect2_float) { e0, e0 } * mat1.m[0].n[1];

#elif SIMD_AUTOVECT == SIMD_4WAY
			m[i].n = (vect4_float) { e0, e0, e0, e0 } * mat1.m[0].n;

#elif SIMD_AUTOVECT == SIMD_8WAY
			m[i].n = (vect8_float) { e0, e0, e0, e0, e0, e0, e0, e0 } * mat1.m[0].n;

#elif SIMD_AUTOVECT == SIMD_16WAY
			m[i].n = (vect16_float) { e0, e0, e0, e0, e0, e0, e0, e0, e0, e0, e0, e0, e0, e0, e0, e0 } * mat1.m[0].n;

#else // scalar (SIMD_AUTOVECT == SIMD_1WAY)
#if SIMD_ETALON_MANUAL_UNROLL == 0
			for (unsigned j = 0; j < 4; ++j)
				m[i].m[j] = e0 * mat1.m[0].m[j];

#endif // SIMD_ETALON_MANUAL_UNROLL
#endif // SIMD_preamble_per_output_row

// Manually-emitted intrinsics are done with their work per
// this output row; following is for the autovectorizer alone.

#if SIMD_INTRINSICS == 0
			for (unsigned j = 1; j < 4; ++j) 
			{
				const float ej = mat0.m[i].m[j];

#if SIMD_process_rows_1_through_4
#error

#elif SIMD_AUTOVECT == SIMD_2WAY
				m[i].n[0] += (vect2_float) { ej, ej } * mat1.m[j].n[0];
				m[i].n[1] += (vect2_float) { ej. ej } * mat1.m[j].n[1];

#elif SIMD_AUTOVECT == SIMD_4WAY
				m[i].n += (vect4_float) { ej, ej, ej, ej } * mat1.m[j].n;

#elif SIMD_AUTOVECT == SIMD_8WAY
				m[i].n += (vect8_float) { ej, ej, ej, ej, ej, ej, ej, ej } * mat1.m[j].n;

#elif SIMD_AUTOVECT == SIMD_16WAY
				m[i].n += (vect16_float) { ej, ej, ej, ej, ej, ej, ej, ej, ej, ej, ej, ej, ej, ej, ej, ej } * mat1.m[j].n;

#else // scalar (SIMD_AUTOVECT == SIMD_1WAY)
#if SIMD_ETALON_MANUAL_UNROLL == 0
				for (unsigned k = 0; k < 4; ++k)
					m[i].m[k] += ej * mat1.m[j].m[k];

#endif // SIMD_ETALON_MANUAL_UNROLL
#endif // SIMD_process_rows_1_through_4
			}

#if SIMD_AUTOVECT == SIMD_1WAY && SIMD_ETALON_MANUAL_UNROLL != 0
			const float e1 = mat0.m[i].m[1];
			const float e2 = mat0.m[i].m[2];
			const float e3 = mat0.m[i].m[3];

			float col0 = e0 * mat1.m[0].m[0];
			float col1 = e0 * mat1.m[0].m[1];
			float col2 = e0 * mat1.m[0].m[2];
			float col3 = e0 * mat1.m[0].m[3];

			col0      += e1 * mat1.m[1].m[0];
			col1      += e1 * mat1.m[1].m[1];
			col2      += e1 * mat1.m[1].m[2];
			col3      += e1 * mat1.m[1].m[3];

			col0      += e2 * mat1.m[2].m[0];
			col1      += e2 * mat1.m[2].m[1];
			col2      += e2 * mat1.m[2].m[2];
			col3      += e2 * mat1.m[2].m[3];

			col0      += e3 * mat1.m[3].m[0];
			col1      += e3 * mat1.m[3].m[1];
			col2      += e3 * mat1.m[3].m[2];
			col3      += e3 * mat1.m[3].m[3];

			m[i].m[0] = col0;
			m[i].m[1] = col1;
			m[i].m[2] = col2;
			m[i].m[3] = col3;

#endif

#endif // SIMD_INTRINSICS == 0
		}

		return *this;
	}
};

} // namespace etal

std::istream& operator >> (
	std::istream& str,
	SIMD_NAMESPACE::matx4& a)
{
	float t[4][4];

	str >> t[0][0];
	str >> t[0][1];
	str >> t[0][2];
	str >> t[0][3];

	str >> t[1][0];
	str >> t[1][1];
	str >> t[1][2];
	str >> t[1][3];

	str >> t[2][0];
	str >> t[2][1];
	str >> t[2][2];
	str >> t[2][3];

	str >> t[3][0];
	str >> t[3][1];
	str >> t[3][2];
	str >> t[3][3];

	a = SIMD_NAMESPACE::matx4(t);

	return str;
}


class formatter
{
	const float m;

public:
	formatter(const float& a)
	: m(a)
	{
	}

	float get() const
	{
		return m;
	}
};


std::ostream& operator << (
	std::ostream& str,
	const formatter& a)
{
	return str << std::setw(12) << std::setfill('_') << a.get();
}


std::ostream& operator << (
	std::ostream& str,
	const SIMD_NAMESPACE::matx4& a)
{
	return str <<
		formatter(a.get(0, 0)) << " " << formatter(a.get(0, 1)) << " " << formatter(a.get(0, 2)) << " " << formatter(a.get(0, 3)) << '\n' <<
		formatter(a.get(1, 0)) << " " << formatter(a.get(1, 1)) << " " << formatter(a.get(1, 2)) << " " << formatter(a.get(1, 3)) << '\n' <<
		formatter(a.get(2, 0)) << " " << formatter(a.get(2, 1)) << " " << formatter(a.get(2, 2)) << " " << formatter(a.get(2, 3)) << '\n' <<
		formatter(a.get(3, 0)) << " " << formatter(a.get(3, 1)) << " " << formatter(a.get(3, 2)) << " " << formatter(a.get(3, 3)) << '\n' <<
		std::endl;
}

static const size_t reps = size_t(1e+7) * 6;
static const size_t nthreads = SIMD_NUM_THREADS;
static const size_t one_less = nthreads - 1;

SIMD_NAMESPACE::matx4 ma[2];
SIMD_NAMESPACE::matx4 ra[nthreads] __attribute__ ((aligned(CACHELINE_SIZE)));

template < bool >
struct compile_assert;

template <>
struct compile_assert< true >
{
	compile_assert() {}
};

static compile_assert< sizeof(SIMD_NAMESPACE::matx4) / CACHELINE_SIZE * CACHELINE_SIZE == sizeof(SIMD_NAMESPACE::matx4) > assert_ra_element_size;

enum {
	BARRIER_START,
	BARRIER_FINISH,
	BARRIER_COUNT
};

static pthread_barrier_t barrier[BARRIER_COUNT];

struct compute_arg
{
	pthread_t thread;
	size_t id;
	size_t offset;

	compute_arg()
	: thread(0)
	, id(0)
	, offset(0)
	{
	}

	compute_arg(
		const size_t arg_id,
		const size_t arg_offs)
	: thread(0)
	, id(arg_id)
	, offset(arg_offs)
	{
	}
};

static void*
compute(
	void* arg)
{
	const compute_arg* const carg = reinterpret_cast< compute_arg* >(arg);
	const size_t id = carg->id;
	const size_t offset = carg->offset;

	pthread_barrier_wait(barrier + BARRIER_START);

	for (size_t i = 0; i < reps; ++i)
	{
		const size_t offs0 = i * offset + 0;
		const size_t offs1 = i * offset + 1;

		ra[id + offs0] = SIMD_NAMESPACE::matx4().mul(ma[offs0], ma[offs1]);
	}

	pthread_barrier_wait(barrier + BARRIER_FINISH);

	return 0;
}


class workforce_t
{
	compute_arg record[one_less];
	bool successfully_init;

public:
	workforce_t(const size_t offset);
	~workforce_t();

	bool is_successfully_init() const
	{
		return successfully_init;
	}
};


static void
report_err(
	const char* const func,
	const int line,
	const size_t counter,
	const int err)
{
	std::cerr << func << ':' << line << ", i: "
		<< counter << ", err: " << err << std::endl;
}


workforce_t::workforce_t(
	const size_t offset) :
	successfully_init(false)
{
	for (size_t i = 0; i < sizeof(barrier) / sizeof(barrier[0]); ++i)
	{
		const int r = pthread_barrier_init(barrier + i, NULL, nthreads);

		if (0 != r)
		{
			report_err(__FUNCTION__, __LINE__, i, r);
			return;
		}
	}

	for (size_t i = 0; i < one_less; ++i)
	{
		const size_t id = i + 1;
		record[i] = compute_arg(id, offset);

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
		CPU_SET(id * SIMD_THREAD_AFFINITY_STRIDE, &affin);

		const int ra = pthread_attr_setaffinity_np(&scoped.attr, sizeof(affin), &affin);

		if (0 != ra)
		{
			report_err(__FUNCTION__, __LINE__, i, ra);
			return;
		}

		const int r = pthread_create(&record[i].thread, &scoped.attr, compute, record + i);

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
	for (size_t i = 0; i < sizeof(barrier) / sizeof(barrier[0]); ++i)
	{
		const int r = pthread_barrier_destroy(barrier + i);

		if (0 != r)
			report_err(__FUNCTION__, __LINE__, i, r);
	}

	for (size_t i = 0; i < one_less && 0 != record[i].thread; ++i)
	{
		const int r = pthread_join(record[i].thread, NULL);

		if (0 != r)
			report_err(__FUNCTION__, __LINE__, i, r);
	}
}


#if __SSE__
static std::ostream& operator << (
	std::ostream& str,
	const __m128i a)
{
	str << std::hex << "{ 0x" <<
		uint32_t(a[0] >>  0) << ", 0x" <<
		uint32_t(a[0] >> 32) << ", 0x" <<
		uint32_t(a[1] >>  0) << ", 0x" <<
		uint32_t(a[1] >> 32) << " }" << std::dec;
	return str;
}

class formatter_m128
{
	union {
		uint32_t u;
		float f;
	};

public:
	formatter_m128(const float a)
	: f(a) {
	}

	uint32_t get() const {
		return u;
	}
};

static std::ostream& operator << (
	std::ostream& str,
	const formatter_m128 a)
{
	str << 
		(a.get() >> 31) << ":" <<
		std::setw(2) << std::setfill('0') << (a.get() >> 23 & 0xff) << ":" <<
		std::setw(6) << std::setfill('0') << (a.get() >>  0 & 0x7fffff);
	return str;
}

static std::ostream& operator << (
	std::ostream& str,
	const __m128 a)
{
	str << "{ " <<
		a[0] << ", " <<
		a[1] << ", " <<
		a[2] << ", " <<
		a[3] << " } " << std::hex << "[ " <<
		formatter_m128(a[0]) << ", " <<
		formatter_m128(a[1]) << ", " <<
		formatter_m128(a[2]) << ", " <<
		formatter_m128(a[3]) << " ]" << std::dec;
	return str;
}

#endif // __SSE__
static bool
conformance()
{
#if SIMD_SCALAR
	using namespace scal;

#else
	using namespace simd;

#endif

	bool success = true;

	const vect2 v2(1.f, 2.f);
	const vect3 v3(1.f, 2.f, 3.f);
	const vect4 v4(1.f, 2.f, 3.f, 4.f);

	const ivect2 iv2(1, 2);
	const ivect3 iv3(1, 2, 3);
	const ivect4 iv4(1, 2, 3, 4);

	// test for false positive
	if (v2 != vect2(1.f, 2.f) ||
		v3 != vect3(1.f, 2.f, 3.f) ||
		v4 != vect4(1.f, 2.f, 3.f, 4.f))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false negative
	if (v2 != vect2(1.f, -2.f) &&
		v3 != vect3(1.f, 2.f, -3.f) &&
		v4 != vect4(1.f, 2.f, 3.f, -4.f))
	{
	}
	else
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false positive
	if (v2 == vect2(1.f, -2.f) ||
		v3 == vect3(1.f, 2.f, -3.f) ||
		v4 == vect4(1.f, 2.f, 3.f, -4.f))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false negative
	if (v2 == vect2(1.f, 2.f) &&
		v3 == vect3(1.f, 2.f, 3.f) &&
		v4 == vect4(1.f, 2.f, 3.f, 4.f))
	{
	}
	else
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false positive
	if (iv2 != ivect2(1, 2) ||
		iv3 != ivect3(1, 2, 3) ||
		iv4 != ivect4(1, 2, 3, 4))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false negative
	if (iv2 != ivect2(1, -2) &&
		iv3 != ivect3(1, 2, -3) &&
		iv4 != ivect4(1, 2, 3, -4))
	{
	}
	else
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false positive
	if (iv2 == ivect2(1, -2) ||
		iv3 == ivect3(1, 2, -3) ||
		iv4 == ivect4(1, 2, 3, -4))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false negative
	if (iv2 == ivect2(1, 2) &&
		iv3 == ivect3(1, 2, 3) &&
		iv4 == ivect4(1, 2, 3, 4))
	{
	}
	else
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	const vect2& precision2 = vect2(
		 1.f + 1.f / (1 << 23),
		-1.f - 1.f / (1 << 23)).mul(2.f);
	const vect3& precision3 = vect3(
		 1.f + 1.f / (1 << 23),
		-1.f - 1.f / (1 << 23),
		 1.f + 1.f / (1 << 23)).mul(2.f);
	const vect4& precision4 = vect4(
		 1.f + 1.f / (1 << 23),
		-1.f - 1.f / (1 << 23),
		 1.f + 1.f / (1 << 23),
		-1.f - 1.f / (1 << 23)).mul(2.f);

	if (precision2[0] !=  2.f + 2.f / (1 << 23) ||
		precision2[1] != -2.f - 2.f / (1 << 23) ||
		precision3[0] !=  2.f + 2.f / (1 << 23) ||
		precision3[1] != -2.f - 2.f / (1 << 23) ||
		precision3[2] !=  2.f + 2.f / (1 << 23) ||
		precision4[0] !=  2.f + 2.f / (1 << 23) ||
		precision4[1] != -2.f - 2.f / (1 << 23) ||
		precision4[2] !=  2.f + 2.f / (1 << 23) ||
		precision4[3] != -2.f - 2.f / (1 << 23))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	const float f[] = { 42.f, 43.f, 44.f, 45.f };
	const int i[] = { 42, 43, 44, 45 };

	if (vect2(*reinterpret_cast< const float(*)[2] >(f)).negate() != vect2(-f[0], -f[1]) ||
		vect3(*reinterpret_cast< const float(*)[3] >(f)).negate() != vect3(-f[0], -f[1], -f[2]) ||
		vect4(*reinterpret_cast< const float(*)[4] >(f)).negate() != vect4(-f[0], -f[1], -f[2], -f[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2(*reinterpret_cast< const int(*)[2] >(i)).negate() != ivect2(-i[0], -i[1]) ||
		ivect3(*reinterpret_cast< const int(*)[3] >(i)).negate() != ivect3(-i[0], -i[1], -i[2]) ||
		ivect4(*reinterpret_cast< const int(*)[4] >(i)).negate() != ivect4(-i[0], -i[1], -i[2], -i[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().add(v2, vect2(*reinterpret_cast< const float(*)[2] >(f))) != vect2(v2[0] + f[0], v2[1] + f[1]) ||
		vect3().add(v3, vect3(*reinterpret_cast< const float(*)[3] >(f))) != vect3(v3[0] + f[0], v3[1] + f[1], v3[2] + f[2]) ||
		vect4().add(v4, vect4(*reinterpret_cast< const float(*)[4] >(f))) != vect4(v4[0] + f[0], v4[1] + f[1], v4[2] + f[2], v4[3] + f[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().sub(v2, vect2(*reinterpret_cast< const float(*)[2] >(f))) != vect2(v2[0] - f[0], v2[1] - f[1]) ||
		vect3().sub(v3, vect3(*reinterpret_cast< const float(*)[3] >(f))) != vect3(v3[0] - f[0], v3[1] - f[1], v3[2] - f[2]) ||
		vect4().sub(v4, vect4(*reinterpret_cast< const float(*)[4] >(f))) != vect4(v4[0] - f[0], v4[1] - f[1], v4[2] - f[2], v4[3] - f[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2().add(iv2, ivect2(*reinterpret_cast< const int(*)[2] >(i))) != ivect2(iv2[0] + i[0], iv2[1] + i[1]) ||
		ivect3().add(iv3, ivect3(*reinterpret_cast< const int(*)[3] >(i))) != ivect3(iv3[0] + i[0], iv3[1] + i[1], iv3[2] + i[2]) ||
		ivect4().add(iv4, ivect4(*reinterpret_cast< const int(*)[4] >(i))) != ivect4(iv4[0] + i[0], iv4[1] + i[1], iv4[2] + i[2], iv4[3] + i[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2().sub(iv2, ivect2(*reinterpret_cast< const int(*)[2] >(i))) != ivect2(iv2[0] - i[0], iv2[1] - i[1]) ||
		ivect3().sub(iv3, ivect3(*reinterpret_cast< const int(*)[3] >(i))) != ivect3(iv3[0] - i[0], iv3[1] - i[1], iv3[2] - i[2]) ||
		ivect4().sub(iv4, ivect4(*reinterpret_cast< const int(*)[4] >(i))) != ivect4(iv4[0] - i[0], iv4[1] - i[1], iv4[2] - i[2], iv4[3] - i[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().mul(v2, vect2(*reinterpret_cast< const float(*)[2] >(f))) != vect2(v2[0] * f[0], v2[1] * f[1]) ||
		vect3().mul(v3, vect3(*reinterpret_cast< const float(*)[3] >(f))) != vect3(v3[0] * f[0], v3[1] * f[1], v3[2] * f[2]) ||
		vect4().mul(v4, vect4(*reinterpret_cast< const float(*)[4] >(f))) != vect4(v4[0] * f[0], v4[1] * f[1], v4[2] * f[2], v4[3] * f[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().div(v2, vect2(*reinterpret_cast< const float(*)[2] >(f))) != vect2(v2[0] / f[0], v2[1] / f[1]) ||
		vect3().div(v3, vect3(*reinterpret_cast< const float(*)[3] >(f))) != vect3(v3[0] / f[0], v3[1] / f[1], v3[2] / f[2]) ||
		vect4().div(v4, vect4(*reinterpret_cast< const float(*)[4] >(f))) != vect4(v4[0] / f[0], v4[1] / f[1], v4[2] / f[2], v4[3] / f[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().div(v2, precision2) != vect2(v2[0] / precision2[0], v2[1] / precision2[1]) ||
		vect3().div(v3, precision3) != vect3(v3[0] / precision3[0], v3[1] / precision3[1], v3[2] / precision3[2]) ||
		vect4().div(v4, precision4) != vect4(v4[0] / precision4[0], v4[1] / precision4[1], v4[2] / precision4[2], v4[3] / precision4[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2().mul(iv2, ivect2(*reinterpret_cast< const int(*)[2] >(i))) != ivect2(iv2[0] * i[0], iv2[1] * i[1]) ||
		ivect3().mul(iv3, ivect3(*reinterpret_cast< const int(*)[3] >(i))) != ivect3(iv3[0] * i[0], iv3[1] * i[1], iv3[2] * i[2]) ||
		ivect4().mul(iv4, ivect4(*reinterpret_cast< const int(*)[4] >(i))) != ivect4(iv4[0] * i[0], iv4[1] * i[1], iv4[2] * i[2], iv4[3] * i[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2().div(ivect2(*reinterpret_cast< const int(*)[2] >(i)), iv2) != ivect2(i[0] / iv2[0], i[1] / iv2[1]) ||
		ivect3().div(ivect3(*reinterpret_cast< const int(*)[3] >(i)), iv3) != ivect3(i[0] / iv3[0], i[1] / iv3[1], i[2] / iv3[2]) ||
		ivect4().div(ivect4(*reinterpret_cast< const int(*)[4] >(i)), iv4) != ivect4(i[0] / iv4[0], i[1] / iv4[1], i[2] / iv4[2], i[3] / iv4[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().mad(v2, v2, vect2(*reinterpret_cast< const float(*)[2] >(f))) != vect2(v2[0] * v2[0] + f[0], v2[1] * v2[1] + f[1]) ||
		vect3().mad(v3, v3, vect3(*reinterpret_cast< const float(*)[3] >(f))) != vect3(v3[0] * v3[0] + f[0], v3[1] * v3[1] + f[1], v3[2] * v3[2] + f[2]) ||
		vect4().mad(v4, v4, vect4(*reinterpret_cast< const float(*)[4] >(f))) != vect4(v4[0] * v4[0] + f[0], v4[1] * v4[1] + f[1], v4[2] * v4[2] + f[2], v4[3] * v4[3] + f[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2().mad(iv2, iv2, ivect2(*reinterpret_cast< const int(*)[2] >(i))) != ivect2(iv2[0] * iv2[0] + i[0], iv2[1] * iv2[1] + i[1]) ||
		ivect3().mad(iv3, iv3, ivect3(*reinterpret_cast< const int(*)[3] >(i))) != ivect3(iv3[0] * iv3[0] + i[0], iv3[1] * iv3[1] + i[1], iv3[2] * iv3[2] + i[2]) ||
		ivect4().mad(iv4, iv4, ivect4(*reinterpret_cast< const int(*)[4] >(i))) != ivect4(iv4[0] * iv4[0] + i[0], iv4[1] * iv4[1] + i[1], iv4[2] * iv4[2] + i[2], iv4[3] * iv4[3] + i[3]))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().wsum(v2, v2, .25f, -.25f) != vect2(0.f, 0.f) ||
		vect3().wsum(v3, v3, .25f, -.25f) != vect3(0.f, 0.f, 0.f) ||
		vect4().wsum(v4, v4, .25f, -.25f) != vect4(0.f, 0.f, 0.f, 0.f))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2().wsum(iv2, iv2, 25, -25) != ivect2(0, 0) ||
		ivect3().wsum(iv3, iv3, 25, -25) != ivect3(0, 0, 0) ||
		ivect4().wsum(iv4, iv4, 25, -25) != ivect4(0, 0, 0, 0))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().cross(vect2(1.f, 0.f), vect2(0.f, 1.f)) != vect2(1.f, 1.f) ||
		vect3().cross(vect3(1.f, 0.f, 0.f), vect3(0.f, 1.f, 0.f)) != vect3(0.f, 0.f, 1.f) ||
		vect4().cross(vect4(1.f, 0.f, 0.f, 0.f), vect4(0.f, 1.f, 0.f, 0.f)) != vect4(0.f, 0.f, 1.f, 0.f))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (matx4(
		 1.f,  2.f,  3.f,  4.f,
		 5.f,  6.f,  7.f,  8.f,
		 9.f, 10.f, 11.f, 12.f,
		13.f, 14.f, 15.f, 16.f).transpose() !=
		matx4(
		 1.f,  5.f,  9.f, 13.f,
		 2.f,  6.f, 10.f, 14.f,
		 3.f,  7.f, 11.f, 15.f,
		 4.f,  8.f, 12.f, 16.f))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// angle = pi, axis = { 0, 0, 1 }
	if (vect3(1.f, 0.f, 0.f).mul(matx3(quat(0.f, 0.f, 1.f, 0.f))) != vect3(-1.f, 0.f, 0.f) ||
		vect4(1.f, 0.f, 0.f, 1.f).mul(matx4(quat(0.f, 0.f, 1.f, 0.f))) != vect4(-1.f, 0.f, 0.f, 1.f))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	const matx4 mat(
		 0.f, 1.f, 0.f, 0.f,
		-1.f, 0.f, 0.f, 0.f, 
		 0.f, 0.f, 1.f, 0.f,
		 0.f, 0.f, 0.f, 1.f);

	if (matx4().inverse(mat) != matx4(
		0.f, -1.f, 0.f, 0.f,
		1.f,  0.f, 0.f, 0.f,
		0.f,  0.f, 1.f, 0.f,
		0.f,  0.f, 0.f, 1.f))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	const vect3 nv3(1.f, 2.f, 4.f);
	const vect4 nv4(1.f, 2.f, 4.f, 8.f);

	if (vect3().normalise(nv3) != vect3(nv3[0] / nv3.norm(), nv3[1] / nv3.norm(), nv3[2] / nv3.norm()) ||
		vect4().normalise(nv4) != vect4(nv4[0] / nv4.norm(), nv4[1] / nv4.norm(), nv4[2] / nv4.norm(), nv4[3] / nv4.norm()))
	{
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;

		std::cout <<
			vect3(nv3[0] / nv3.norm(), nv3[1] / nv3.norm(), nv3[2] / nv3.norm()).normalise().getn() << std::endl <<
			vect3().normalise(nv3).getn() << std::endl;

		std::cout <<
			vect4(nv4[0] / nv4.norm(), nv4[1] / nv4.norm(), nv4[2] / nv4.norm(), nv4[3] / nv4.norm()).normalise().getn() << std::endl <<
			vect4().normalise(nv4).getn() << std::endl;
	}

	return success;
}


int
main(
	int argc,
	char** argv)
{
	std::cout << "conformance test.." << std::endl;
	std::cout << (conformance() ? "passed" : "failed") << std::endl;

	std::cout << "performance test.." << std::endl;

	size_t obfuscator; // invariance obfuscator
	std::ifstream in("vect.input");

	if (in.is_open())
	{
		in >> ma[0];
		in >> ma[1];
		in >> obfuscator;
		in.close();
	}
	else
	{
		std::cout << "enter ma0: ";
		std::cin >> ma[0];
		std::cout << "enter ma1: ";
		std::cin >> ma[1];
		std::cout << "enter 0: ";
		std::cin >> obfuscator;
	}

	const workforce_t workforce(obfuscator);

	if (!workforce.is_successfully_init())
	{
		std::cerr << "failed to raise workforce; bailing out" << std::endl;
		return -1;
	}

	// let the workforce start their engines
	const timespec ts = { 0, one_less * 100000000 };
	nanosleep(&ts, NULL);

	compute_arg carg(0, obfuscator);

	const uint64_t t0 = timer_nsec();

	compute(&carg);

	const uint64_t dt = timer_nsec() - t0;
	const double sec = double(dt) * 1e-9;

	std::cout << "elapsed time: " << sec << " s" << std::endl;

	for (size_t i = 0; i < sizeof(ra) / sizeof(ra[0]); ++i)
		std::cout << ra[i];

	return 0;
}
