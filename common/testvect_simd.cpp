#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <stdint.h>
#include <pthread.h>
#include "timer.h"
#if __APPLE__ != 0
	#include "pthread_barrier.h"
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
	#elif __ARM_NEON__ == 1 || __ARM_NEON == 1
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

#if SIMD_ETALON || SIMD_AUTOVECT
namespace etal {

class __attribute__ ((aligned(SIMD_ALIGNMENT))) matx4 {

#if SIMD_AUTOVECT == SIMD_2WAY
	typedef __attribute__ ((vector_size(2 * sizeof(float)))) float vect2_float;

#elif SIMD_AUTOVECT == SIMD_4WAY
	typedef __attribute__ ((vector_size(4 * sizeof(float)))) float vect4_float;

#elif SIMD_AUTOVECT == SIMD_8WAY
	typedef __attribute__ ((vector_size(4 * sizeof(float)))) float vect4_float;
	typedef __attribute__ ((vector_size(8 * sizeof(float)))) float vect8_float;

#elif SIMD_AUTOVECT == SIMD_16WAY
	typedef __attribute__ ((vector_size(4 * sizeof(float)))) float vect4_float;
	typedef __attribute__ ((vector_size(16 * sizeof(float)))) float vect16_float;

#endif
	union {
		float m[4][4];

#if SIMD_AUTOVECT == SIMD_2WAY
		vect2_float n[8];

#elif SIMD_AUTOVECT == SIMD_4WAY
		vect4_float n[4];

#elif SIMD_AUTOVECT == SIMD_8WAY
		vect8_float n[2];

#elif SIMD_AUTOVECT == SIMD_16WAY
		vect16_float n;

#elif SIMD_INTRINSICS == SIMD_ALTIVEC
		vector float n[4];

#elif SIMD_INTRINSICS == SIMD_SSE
		__m128 n[4];

#elif SIMD_INTRINSICS == SIMD_NEON
		float32x4_t n[4];

#elif SIMD_INTRINSICS == SIMD_MIC
		__m512 n;

#endif
	};

public:
	matx4() {
	}

	matx4(
		const float m00, const float m01, const float m02, const float m03,
		const float m10, const float m11, const float m12, const float m13,
		const float m20, const float m21, const float m22, const float m23,
		const float m30, const float m31, const float m32, const float m33) {

		m[0][0] = m00;
		m[0][1] = m01;
		m[0][2] = m02;
		m[0][3] = m03;

		m[1][0] = m10;
		m[1][1] = m11;
		m[1][2] = m12;
		m[1][3] = m13;

		m[2][0] = m20;
		m[2][1] = m21;
		m[2][2] = m22;
		m[2][3] = m23;

		m[3][0] = m30;
		m[3][1] = m31;
		m[3][2] = m32;
		m[3][3] = m33;
	}

	matx4(
		const float (&src)[4][4]) {

		m[0][0] = src[0][0];
		m[0][1] = src[0][1];
		m[0][2] = src[0][2];
		m[0][3] = src[0][3];

		m[1][0] = src[1][0];
		m[1][1] = src[1][1];
		m[1][2] = src[1][2];
		m[1][3] = src[1][3];

		m[2][0] = src[2][0];
		m[2][1] = src[2][1];
		m[2][2] = src[2][2];
		m[2][3] = src[2][3];

		m[3][0] = src[3][0];
		m[3][1] = src[3][1];
		m[3][2] = src[3][2];
		m[3][3] = src[3][3];
	}

	float get(
		const size_t i,
		const size_t j) const {

		return m[i][j];
	}

	const float (& operator[](const size_t i) const)[4] {
		return m[i];
	}

#if SIMD_ETALON_ALT0
	inline __attribute__ ((always_inline)) matx4& mul(
		const matx4& ma0,
		const matx4& ma1) {

		typedef __attribute__ ((vector_size(16 * sizeof(float)))) float vect16;

		// pass 0
		const vect16 pass0a = (vect16) {
			ma0[0][0], ma0[0][0], ma0[0][0], ma0[0][0],
			ma0[1][0], ma0[1][0], ma0[1][0], ma0[1][0],
			ma0[2][0], ma0[2][0], ma0[2][0], ma0[2][0],
			ma0[3][0], ma0[3][0], ma0[3][0], ma0[3][0] };

		const vect16 pass0b = (vect16) {
			ma1[0][0], ma1[0][1], ma1[0][2], ma1[0][3],
			ma1[0][0], ma1[0][1], ma1[0][2], ma1[0][3],
			ma1[0][0], ma1[0][1], ma1[0][2], ma1[0][3],
			ma1[0][0], ma1[0][1], ma1[0][2], ma1[0][3] };

		// pass 1
		const vect16 pass1a = (vect16) {
			ma0[0][1], ma0[0][1], ma0[0][1], ma0[0][1],
			ma0[1][1], ma0[1][1], ma0[1][1], ma0[1][1],
			ma0[2][1], ma0[2][1], ma0[2][1], ma0[2][1],
			ma0[3][1], ma0[3][1], ma0[3][1], ma0[3][1] };

		const vect16 pass1b = (vect16) {
			ma1[1][0], ma1[1][1], ma1[1][2], ma1[1][3],
			ma1[1][0], ma1[1][1], ma1[1][2], ma1[1][3],
			ma1[1][0], ma1[1][1], ma1[1][2], ma1[1][3],
			ma1[1][0], ma1[1][1], ma1[1][2], ma1[1][3] };

		// pass 2
		const vect16 pass2a = (vect16) {
			ma0[0][2], ma0[0][2], ma0[0][2], ma0[0][2],
			ma0[1][2], ma0[1][2], ma0[1][2], ma0[1][2],
			ma0[2][2], ma0[2][2], ma0[2][2], ma0[2][2],
			ma0[3][2], ma0[3][2], ma0[3][2], ma0[3][2] };

		const vect16 pass2b = (vect16) {
			ma1[2][0], ma1[2][1], ma1[2][2], ma1[2][3],
			ma1[2][0], ma1[2][1], ma1[2][2], ma1[2][3],
			ma1[2][0], ma1[2][1], ma1[2][2], ma1[2][3],
			ma1[2][0], ma1[2][1], ma1[2][2], ma1[2][3] };

		// pass 3
		const vect16 pass3a = (vect16) {
			ma0[0][3], ma0[0][3], ma0[0][3], ma0[0][3],
			ma0[1][3], ma0[1][3], ma0[1][3], ma0[1][3],
			ma0[2][3], ma0[2][3], ma0[2][3], ma0[2][3],
			ma0[3][3], ma0[3][3], ma0[3][3], ma0[3][3] };

		const vect16 pass3b = (vect16) {
			ma1[3][0], ma1[3][1], ma1[3][2], ma1[3][3],
			ma1[3][0], ma1[3][1], ma1[3][2], ma1[3][3],
			ma1[3][0], ma1[3][1], ma1[3][2], ma1[3][3],
			ma1[3][0], ma1[3][1], ma1[3][2], ma1[3][3] };

		const vect16 res =
			pass0a * pass0b +
			pass1a * pass1b +
			pass2a * pass2b +
			pass3a * pass3b;

		return *this = *reinterpret_cast< const matx4* >(&res);
	}

#elif SIMD_ETALON_ALT1
	inline __attribute__ ((always_inline)) matx4& mul(
		const matx4& ma0,
		const matx4& ma1) {

		typedef __attribute__ ((vector_size( 4 * sizeof(float)))) float vect4;
		typedef __attribute__ ((vector_size(16 * sizeof(float)))) float vect16;

#if __AVX__ == 0
		// try hinting the compiler of splatting elements from 4-way vector reads
		const vect4 ma0r0 = *reinterpret_cast< const vect4* >(ma0[0]);
		const vect4 ma0r1 = *reinterpret_cast< const vect4* >(ma0[1]);
		const vect4 ma0r2 = *reinterpret_cast< const vect4* >(ma0[2]);
		const vect4 ma0r3 = *reinterpret_cast< const vect4* >(ma0[3]);

		const vect4 ma1r0 = *reinterpret_cast< const vect4* >(ma1[0]);
		const vect4 ma1r1 = *reinterpret_cast< const vect4* >(ma1[1]);
		const vect4 ma1r2 = *reinterpret_cast< const vect4* >(ma1[2]);
		const vect4 ma1r3 = *reinterpret_cast< const vect4* >(ma1[3]);

#else
		typedef __attribute__ ((vector_size( 8 * sizeof(float)))) float vect8;

		// try hinting the compiler of splatting elements from 8-way vector reads
		const vect8 ma0r01 = *reinterpret_cast< const vect8* >(ma0[0]);
		const vect8 ma0r23 = *reinterpret_cast< const vect8* >(ma0[2]);

		const vect4 ma0r0 = (vect4){ ma0r01[0], ma0r01[1], ma0r01[2], ma0r01[3] };
		const vect4 ma0r1 = (vect4){ ma0r01[4], ma0r01[5], ma0r01[6], ma0r01[7] };
		const vect4 ma0r2 = (vect4){ ma0r23[0], ma0r23[1], ma0r23[2], ma0r23[3] };
		const vect4 ma0r3 = (vect4){ ma0r23[4], ma0r23[5], ma0r23[6], ma0r23[7] };

#if DUBIOUS_AVX_OPTIMISATION != 0
		const vect8 ma1r01 = *reinterpret_cast< const vect8* >(ma1[0]);
		const vect8 ma1r23 = *reinterpret_cast< const vect8* >(ma1[2]);

		const vect4 ma1r0 = (vect4){ ma1r01[0], ma1r01[1], ma1r01[2], ma1r01[3] };
		const vect4 ma1r1 = (vect4){ ma1r01[4], ma1r01[5], ma1r01[6], ma1r01[7] };
		const vect4 ma1r2 = (vect4){ ma1r23[0], ma1r23[1], ma1r23[2], ma1r23[3] };
		const vect4 ma1r3 = (vect4){ ma1r23[4], ma1r23[5], ma1r23[6], ma1r23[7] };

#else
		const vect4 ma1r0 = *reinterpret_cast< const vect4* >(ma1[0]);
		const vect4 ma1r1 = *reinterpret_cast< const vect4* >(ma1[1]);
		const vect4 ma1r2 = *reinterpret_cast< const vect4* >(ma1[2]);
		const vect4 ma1r3 = *reinterpret_cast< const vect4* >(ma1[3]);

#endif
#endif
		// pass 0
		const vect16 pass0a = (vect16) {
			ma0r0[0], ma0r0[0], ma0r0[0], ma0r0[0],
			ma0r1[0], ma0r1[0], ma0r1[0], ma0r1[0],
			ma0r2[0], ma0r2[0], ma0r2[0], ma0r2[0],
			ma0r3[0], ma0r3[0], ma0r3[0], ma0r3[0] };

		const vect16 pass0b = (vect16) {
			ma1r0[0], ma1r0[1], ma1r0[2], ma1r0[3],
			ma1r0[0], ma1r0[1], ma1r0[2], ma1r0[3],
			ma1r0[0], ma1r0[1], ma1r0[2], ma1r0[3],
			ma1r0[0], ma1r0[1], ma1r0[2], ma1r0[3] };

		// pass 1
		const vect16 pass1a = (vect16) {
			ma0r0[1], ma0r0[1], ma0r0[1], ma0r0[1],
			ma0r1[1], ma0r1[1], ma0r1[1], ma0r1[1],
			ma0r2[1], ma0r2[1], ma0r2[1], ma0r2[1],
			ma0r3[1], ma0r3[1], ma0r3[1], ma0r3[1] };

		const vect16 pass1b = (vect16) {
			ma1r1[0], ma1r1[1], ma1r1[2], ma1r1[3],
			ma1r1[0], ma1r1[1], ma1r1[2], ma1r1[3],
			ma1r1[0], ma1r1[1], ma1r1[2], ma1r1[3],
			ma1r1[0], ma1r1[1], ma1r1[2], ma1r1[3] };

		// pass 2
		const vect16 pass2a = (vect16) {
			ma0r0[2], ma0r0[2], ma0r0[2], ma0r0[2],
			ma0r1[2], ma0r1[2], ma0r1[2], ma0r1[2],
			ma0r2[2], ma0r2[2], ma0r2[2], ma0r2[2],
			ma0r3[2], ma0r3[2], ma0r3[2], ma0r3[2] };

		const vect16 pass2b = (vect16) {
			ma1r2[0], ma1r2[1], ma1r2[2], ma1r2[3],
			ma1r2[0], ma1r2[1], ma1r2[2], ma1r2[3],
			ma1r2[0], ma1r2[1], ma1r2[2], ma1r2[3],
			ma1r2[0], ma1r2[1], ma1r2[2], ma1r2[3] };

		// pass 3
		const vect16 pass3a = (vect16) {
			ma0r0[3], ma0r0[3], ma0r0[3], ma0r0[3],
			ma0r1[3], ma0r1[3], ma0r1[3], ma0r1[3],
			ma0r2[3], ma0r2[3], ma0r2[3], ma0r2[3],
			ma0r3[3], ma0r3[3], ma0r3[3], ma0r3[3] };

		const vect16 pass3b = (vect16) {
			ma1r3[0], ma1r3[1], ma1r3[2], ma1r3[3],
			ma1r3[0], ma1r3[1], ma1r3[2], ma1r3[3],
			ma1r3[0], ma1r3[1], ma1r3[2], ma1r3[3],
			ma1r3[0], ma1r3[1], ma1r3[2], ma1r3[3] };

		const vect16 res =
			pass0a * pass0b +
			pass1a * pass1b +
			pass2a * pass2b +
			pass3a * pass3b;

		return *this = *reinterpret_cast< const matx4* >(&res);
	}

#elif SIMD_ETALON_ALT2
	inline __attribute__ ((always_inline)) matx4& mul(
		const matx4& ma0,
		const matx4& ma1) {

		typedef __attribute__ ((vector_size( 4 * sizeof(float)))) float vect4;
		typedef __attribute__ ((vector_size(16 * sizeof(float)))) float vect16;

		// try hinting the compiler of using the fast scalar-broadcast op on AVX processors
		const vect4 ma0r0_0 = (vect4) { ma0[0][0], ma0[0][0], ma0[0][0], ma0[0][0] };
		const vect4 ma0r1_0 = (vect4) { ma0[1][0], ma0[1][0], ma0[1][0], ma0[1][0] };
		const vect4 ma0r2_0 = (vect4) { ma0[2][0], ma0[2][0], ma0[2][0], ma0[2][0] };
		const vect4 ma0r3_0 = (vect4) { ma0[3][0], ma0[3][0], ma0[3][0], ma0[3][0] };

		const vect4 ma0r0_1 = (vect4) { ma0[0][1], ma0[0][1], ma0[0][1], ma0[0][1] };
		const vect4 ma0r1_1 = (vect4) { ma0[1][1], ma0[1][1], ma0[1][1], ma0[1][1] };
		const vect4 ma0r2_1 = (vect4) { ma0[2][1], ma0[2][1], ma0[2][1], ma0[2][1] };
		const vect4 ma0r3_1 = (vect4) { ma0[3][1], ma0[3][1], ma0[3][1], ma0[3][1] };

		const vect4 ma0r0_2 = (vect4) { ma0[0][2], ma0[0][2], ma0[0][2], ma0[0][2] };
		const vect4 ma0r1_2 = (vect4) { ma0[1][2], ma0[1][2], ma0[1][2], ma0[1][2] };
		const vect4 ma0r2_2 = (vect4) { ma0[2][2], ma0[2][2], ma0[2][2], ma0[2][2] };
		const vect4 ma0r3_2 = (vect4) { ma0[3][2], ma0[3][2], ma0[3][2], ma0[3][2] };

		const vect4 ma0r0_3 = (vect4) { ma0[0][3], ma0[0][3], ma0[0][3], ma0[0][3] };
		const vect4 ma0r1_3 = (vect4) { ma0[1][3], ma0[1][3], ma0[1][3], ma0[1][3] };
		const vect4 ma0r2_3 = (vect4) { ma0[2][3], ma0[2][3], ma0[2][3], ma0[2][3] };
		const vect4 ma0r3_3 = (vect4) { ma0[3][3], ma0[3][3], ma0[3][3], ma0[3][3] };

		// pass 0
		const vect16 pass0a = (vect16) {
			ma0r0_0[0], ma0r0_0[1], ma0r0_0[2], ma0r0_0[3],
			ma0r1_0[0], ma0r1_0[1], ma0r1_0[2], ma0r1_0[3],
			ma0r2_0[0], ma0r2_0[1], ma0r2_0[2], ma0r2_0[3],
			ma0r3_0[0], ma0r3_0[1], ma0r3_0[2], ma0r3_0[3] };

		const vect16 pass0b = (vect16) {
			ma1[0][0], ma1[0][1], ma1[0][2], ma1[0][3],
			ma1[0][0], ma1[0][1], ma1[0][2], ma1[0][3],
			ma1[0][0], ma1[0][1], ma1[0][2], ma1[0][3],
			ma1[0][0], ma1[0][1], ma1[0][2], ma1[0][3] };

		// pass 1
		const vect16 pass1a = (vect16) {
			ma0r0_1[0], ma0r0_1[1], ma0r0_1[2], ma0r0_1[3],
			ma0r1_1[0], ma0r1_1[1], ma0r1_1[2], ma0r1_1[3],
			ma0r2_1[0], ma0r2_1[1], ma0r2_1[2], ma0r2_1[3],
			ma0r3_1[0], ma0r3_1[1], ma0r3_1[2], ma0r3_1[3] };

		const vect16 pass1b = (vect16) {
			ma1[1][0], ma1[1][1], ma1[1][2], ma1[1][3],
			ma1[1][0], ma1[1][1], ma1[1][2], ma1[1][3],
			ma1[1][0], ma1[1][1], ma1[1][2], ma1[1][3],
			ma1[1][0], ma1[1][1], ma1[1][2], ma1[1][3] };

		// pass 2
		const vect16 pass2a = (vect16) {
			ma0r0_2[0], ma0r0_2[1], ma0r0_2[2], ma0r0_2[3],
			ma0r1_2[0], ma0r1_2[1], ma0r1_2[2], ma0r1_2[3],
			ma0r2_2[0], ma0r2_2[1], ma0r2_2[2], ma0r2_2[3],
			ma0r3_2[0], ma0r3_2[1], ma0r3_2[2], ma0r3_2[3] };

		const vect16 pass2b = (vect16) {
			ma1[2][0], ma1[2][1], ma1[2][2], ma1[2][3],
			ma1[2][0], ma1[2][1], ma1[2][2], ma1[2][3],
			ma1[2][0], ma1[2][1], ma1[2][2], ma1[2][3],
			ma1[2][0], ma1[2][1], ma1[2][2], ma1[2][3] };

		// pass 3
		const vect16 pass3a = (vect16) {
			ma0r0_3[0], ma0r0_3[1], ma0r0_3[2], ma0r0_3[3],
			ma0r1_3[0], ma0r1_3[1], ma0r1_3[2], ma0r1_3[3],
			ma0r2_3[0], ma0r2_3[1], ma0r2_3[2], ma0r2_3[3],
			ma0r3_3[0], ma0r3_3[1], ma0r3_3[2], ma0r3_3[3] };

		const vect16 pass3b = (vect16) {
			ma1[3][0], ma1[3][1], ma1[3][2], ma1[3][3],
			ma1[3][0], ma1[3][1], ma1[3][2], ma1[3][3],
			ma1[3][0], ma1[3][1], ma1[3][2], ma1[3][3],
			ma1[3][0], ma1[3][1], ma1[3][2], ma1[3][3] };

		const vect16 res =
			pass0a * pass0b +
			pass1a * pass1b +
			pass2a * pass2b +
			pass3a * pass3b;

		return *this = *reinterpret_cast< const matx4* >(&res);
	}

#elif SIMD_ETALON_ALT3
	inline __attribute__ ((always_inline)) matx4& mul(
		const matx4& ma0,
		const matx4& ma1) { // ma1 must be transposed in advance

		const __m128 ma0r0 = ma0.n[0];
		const __m128 ma0r1 = ma0.n[1];
		const __m128 ma0r2 = ma0.n[2];
		const __m128 ma0r3 = ma0.n[3];

		const __m128 ma1c0 = ma1.n[0];
		const __m128 ma1c1 = ma1.n[1];
		const __m128 ma1c2 = ma1.n[2];
		const __m128 ma1c3 = ma1.n[3];

		const __m128 re0_0 = _mm_dp_ps(ma0r0, ma1c0, 0xf1);
		const __m128 re0_1 = _mm_dp_ps(ma0r0, ma1c1, 0xf1);
		const __m128 re0_2 = _mm_dp_ps(ma0r0, ma1c2, 0xf1);
		const __m128 re0_3 = _mm_dp_ps(ma0r0, ma1c3, 0xf1);

		const __m128 r0 = _mm_movelh_ps(
			_mm_unpacklo_ps(re0_0, re0_1),
			_mm_unpacklo_ps(re0_2, re0_3));

		const __m128 re1_0 = _mm_dp_ps(ma0r1, ma1c0, 0xf1);
		const __m128 re1_1 = _mm_dp_ps(ma0r1, ma1c1, 0xf1);
		const __m128 re1_2 = _mm_dp_ps(ma0r1, ma1c2, 0xf1);
		const __m128 re1_3 = _mm_dp_ps(ma0r1, ma1c3, 0xf1);

		const __m128 r1 = _mm_movelh_ps(
			_mm_unpacklo_ps(re1_0, re1_1),
			_mm_unpacklo_ps(re1_2, re1_3));

		const __m128 re2_0 = _mm_dp_ps(ma0r2, ma1c0, 0xf1);
		const __m128 re2_1 = _mm_dp_ps(ma0r2, ma1c1, 0xf1);
		const __m128 re2_2 = _mm_dp_ps(ma0r2, ma1c2, 0xf1);
		const __m128 re2_3 = _mm_dp_ps(ma0r2, ma1c3, 0xf1);

		const __m128 r2 = _mm_movelh_ps(
			_mm_unpacklo_ps(re2_0, re2_1),
			_mm_unpacklo_ps(re2_2, re2_3));

		const __m128 re3_0 = _mm_dp_ps(ma0r3, ma1c0, 0xf1);
		const __m128 re3_1 = _mm_dp_ps(ma0r3, ma1c1, 0xf1);
		const __m128 re3_2 = _mm_dp_ps(ma0r3, ma1c2, 0xf1);
		const __m128 re3_3 = _mm_dp_ps(ma0r3, ma1c3, 0xf1);

		const __m128 r3 = _mm_movelh_ps(
			_mm_unpacklo_ps(re3_0, re3_1),
			_mm_unpacklo_ps(re3_2, re3_3));

		n[0] = r0;
		n[1] = r1;
		n[2] = r2;
		n[3] = r3;
		return *this;
	}

#elif SIMD_ETALON_ALT4

#if __ARM_NEON != 1 && __ARM_NEON__ != 1
	#error ARM NEON required
#endif

	inline __attribute__ ((always_inline)) matx4& mul(
		const matx4& mat0,
		const matx4& mat1) {

		register const float32x4_t mat0_0 asm ("q0") = mat0.n[0];
		register const float32x4_t mat0_1 asm ("q1") = mat0.n[1];
		register const float32x4_t mat0_2 asm ("q2") = mat0.n[2];
		register const float32x4_t mat0_3 asm ("q3") = mat0.n[3];

		register const float32x4_t mat1_0 asm ("q4") = mat1.n[0];
		register const float32x4_t mat1_1 asm ("q5") = mat1.n[1];
		register const float32x4_t mat1_2 asm ("q6") = mat1.n[2];
		register const float32x4_t mat1_3 asm ("q7") = mat1.n[3];

		asm volatile(
			"vmul.f32 %[res0], %[matB_0], %e[matA_0][0]\n\t"
			"vmul.f32 %[res1], %[matB_0], %e[matA_1][0]\n\t"
			"vmul.f32 %[res2], %[matB_0], %e[matA_2][0]\n\t"
			"vmul.f32 %[res3], %[matB_0], %e[matA_3][0]\n\t"
			"vmla.f32 %[res0], %[matB_1], %e[matA_0][1]\n\t"
			"vmla.f32 %[res1], %[matB_1], %e[matA_1][1]\n\t"
			"vmla.f32 %[res2], %[matB_1], %e[matA_2][1]\n\t"
			"vmla.f32 %[res3], %[matB_1], %e[matA_3][1]\n\t"
			"vmla.f32 %[res0], %[matB_2], %f[matA_0][0]\n\t"
			"vmla.f32 %[res1], %[matB_2], %f[matA_1][0]\n\t"
			"vmla.f32 %[res2], %[matB_2], %f[matA_2][0]\n\t"
			"vmla.f32 %[res3], %[matB_2], %f[matA_3][0]\n\t"
			"vmla.f32 %[res0], %[matB_3], %f[matA_0][1]\n\t"
			"vmla.f32 %[res1], %[matB_3], %f[matA_1][1]\n\t"
			"vmla.f32 %[res2], %[matB_3], %f[matA_2][1]\n\t"
			"vmla.f32 %[res3], %[matB_3], %f[matA_3][1]"
			: [res0] "=w" (n[0]),
			  [res1] "=w" (n[1]),
			  [res2] "=w" (n[2]),
			  [res3] "=w" (n[3])
			: [matA_0] "w" (mat0_0),
			  [matA_1] "w" (mat0_1),
			  [matA_2] "w" (mat0_2),
			  [matA_3] "w" (mat0_3),
			  [matB_0] "w" (mat1_0),
			  [matB_1] "w" (mat1_1),
			  [matB_2] "w" (mat1_2),
			  [matB_3] "w" (mat1_3)
			:
		);

		return *this;
	}

#elif SIMD_ETALON_ALT5
#if SIMD_INTRINSICS != SIMD_MIC || SIMD_ALIGNMENT != 64
	#error MIC target and SIMD_ALIGNMENT of 64 required
#endif

	inline __attribute__ ((always_inline)) matx4& mul(
		const matx4& mat0,
		const matx4& mat1) {

		const __m512 mat0_a = mat0.n;
		const __m512 mat0_0 = _mm512_swizzle_ps(mat0_a, _MM_SWIZ_REG_AAAA); // a00,a00,a00,a00 / a10,a10,a10,a10 / a20,a20,a20,a20 / a30,a30,a30,a30
		const __m512 mat0_1 = _mm512_swizzle_ps(mat0_a, _MM_SWIZ_REG_BBBB); // a01,a01,a01,a01 / a11,a11,a11,a11 / a21,a21,a21,a21 / a31,a31,a31,a31
		const __m512 mat0_2 = _mm512_swizzle_ps(mat0_a, _MM_SWIZ_REG_CCCC); // a02,a02,a02,a02 / a12,a12,a12,a12 / a22,a22,a22,a22 / a32,a32,a32,a32
		const __m512 mat0_3 = _mm512_swizzle_ps(mat0_a, _MM_SWIZ_REG_DDDD); // a03,a03,a03,a03 / a13,a13,a13,a13 / a23,a23,a23,a23 / a33,a33,a33,a33

		const __m512 mat1_b = mat1.n;
		const __m512 mat1_0 = _mm512_permute4f128_ps(mat1_b, _MM_PERM_AAAA);
		const __m512 mat1_1 = _mm512_permute4f128_ps(mat1_b, _MM_PERM_BBBB);
		const __m512 mat1_2 = _mm512_permute4f128_ps(mat1_b, _MM_PERM_CCCC);
		const __m512 mat1_3 = _mm512_permute4f128_ps(mat1_b, _MM_PERM_DDDD);

		__m512 res;
		res =         _mm512_mul_ps(mat0_0, mat1_0);
		res = _mm512_mask3_fmadd_ps(mat0_1, mat1_1, res, __mmask16(0xffff));
		res = _mm512_mask3_fmadd_ps(mat0_2, mat1_2, res, __mmask16(0xffff));
		res = _mm512_mask3_fmadd_ps(mat0_3, mat1_3, res, __mmask16(0xffff));

		n = res;
		return *this;
	}

#else // SIMD_ETALON_ALT*
	inline __attribute__ ((always_inline)) matx4& mul(
		const matx4& mat0,
		const matx4& mat1) {

#if SIMD_INTRINSICS == SIMD_ALTIVEC
		for (unsigned i = 0; i < 4; ++i) {

			n[i] = vec_madd(vec_splat(mat0.n[i], 0), mat1.n[0], (vector float) { -0.f, -0.f, -0.f -0.f });
			n[i] = vec_madd(vec_splat(mat0.n[i], 1), mat1.n[1], n[i]);
			n[i] = vec_madd(vec_splat(mat0.n[i], 2), mat1.n[2], n[i]);
			n[i] = vec_madd(vec_splat(mat0.n[i], 3), mat1.n[3], n[i]);
		}

#elif SIMD_INTRINSICS == SIMD_SSE
		for (unsigned i = 0; i < 4; ++i) {

#if __AVX__ == 0
			n[i] =            _mm_mul_ps(_mm_shuffle_ps(mat0.n[i], mat0.n[i], _MM_SHUFFLE(0, 0, 0, 0)), mat1.n[0]);
			n[i] = _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(mat0.n[i], mat0.n[i], _MM_SHUFFLE(1, 1, 1, 1)), mat1.n[1]), n[i]);
			n[i] = _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(mat0.n[i], mat0.n[i], _MM_SHUFFLE(2, 2, 2, 2)), mat1.n[2]), n[i]);
			n[i] = _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(mat0.n[i], mat0.n[i], _MM_SHUFFLE(3, 3, 3, 3)), mat1.n[3]), n[i]);

#else
			n[i] =            _mm_mul_ps(_mm_broadcast_ss(mat0.m[i] + 0), mat1.n[0]);
			n[i] = _mm_add_ps(_mm_mul_ps(_mm_broadcast_ss(mat0.m[i] + 1), mat1.n[1]), n[i]);
			n[i] = _mm_add_ps(_mm_mul_ps(_mm_broadcast_ss(mat0.m[i] + 2), mat1.n[2]), n[i]);
			n[i] = _mm_add_ps(_mm_mul_ps(_mm_broadcast_ss(mat0.m[i] + 3), mat1.n[3]), n[i]);

#endif
		}

#elif SIMD_INTRINSICS == SIMD_NEON
#warning ok
		for (unsigned i = 0; i < 4; ++i) {

			// clang generates the same output from both versions; see SIMD_ETALON_ALT4 for desired output from the first version
#if 0
			n[i] = vmulq_lane_f32(      mat1.n[0], vget_low_f32 (mat0.n[i]), 0);
			n[i] = vmlaq_lane_f32(n[i], mat1.n[1], vget_low_f32 (mat0.n[i]), 1);
			n[i] = vmlaq_lane_f32(n[i], mat1.n[2], vget_high_f32(mat0.n[i]), 0);
			n[i] = vmlaq_lane_f32(n[i], mat1.n[3], vget_high_f32(mat0.n[i]), 1);

#else
			n[i] = vmulq_n_f32(      mat1.n[0], vgetq_lane_f32(mat0.n[i], 0));
			n[i] = vmlaq_n_f32(n[i], mat1.n[1], vgetq_lane_f32(mat0.n[i], 1));
			n[i] = vmlaq_n_f32(n[i], mat1.n[2], vgetq_lane_f32(mat0.n[i], 2));
			n[i] = vmlaq_n_f32(n[i], mat1.n[3], vgetq_lane_f32(mat0.n[i], 3));

#endif
		}

#elif SIMD_AUTOVECT == SIMD_2WAY
		for (unsigned i = 0; i < 4; ++i) {

			const vect2_float ai0 = (vect2_float) {
				mat0.m[i][0], mat0.m[i][0]
			};

			n[i * 2 + 0] = ai0 * mat1.n[0];
			n[i * 2 + 1] = ai0 * mat1.n[1];

			for (unsigned j = 1; j < 4; ++j) {

				const vect2_float aij = (vect2_float) {
					mat0.m[i][j], mat0.m[i][j]
				};

				n[i * 2 + 0] += aij * mat1.n[j * 2 + 0];
				n[i * 2 + 1] += aij * mat1.n[j * 2 + 1];
			}
		}

#elif SIMD_AUTOVECT == SIMD_4WAY
		for (unsigned i = 0; i < 4; ++i) {

			const vect4_float ai0 = (vect4_float) {
				mat0.m[i][0], mat0.m[i][0], mat0.m[i][0], mat0.m[i][0]
			};

			n[i] = ai0 * mat1.n[0];

			for (unsigned j = 1; j < 4; ++j) {

				const vect4_float aij = (vect4_float) {
					mat0.m[i][j], mat0.m[i][j], mat0.m[i][j], mat0.m[i][j]
				};

				n[i] += aij * mat1.n[j];
			}
		}

#elif SIMD_AUTOVECT == SIMD_8WAY
#warning ok
		const vect8_float a0 = mat0.n[0];
		const vect8_float a1 = mat0.n[1];
		const vect8_float b0 = mat1.n[0];
		const vect8_float b1 = mat1.n[1];

		// pass 0
		const vect8_float a0_0 = (vect8_float) {
			a0[0], a0[0], a0[0], a0[0],
			a0[4], a0[4], a0[4], a0[4]
		};
		const vect8_float a1_0 = (vect8_float) {
			a1[0], a1[0], a1[0], a1[0],
			a1[4], a1[4], a1[4], a1[4]
		};
		const vect8_float b_0 = (vect8_float) {
			b0[0], b0[1], b0[2], b0[3],
			b0[0], b0[1], b0[2], b0[3]
		};

		// pass 1
		const vect8_float a0_1 = (vect8_float) {
			a0[1], a0[1], a0[1], a0[1],
			a0[5], a0[5], a0[5], a0[5]
		};
		const vect8_float a1_1 = (vect8_float) {
			a1[1], a1[1], a1[1], a1[1],
			a1[5], a1[5], a1[5], a1[5]
		};
		const vect8_float b_1 = (vect8_float) {
			b0[4], b0[5], b0[6], b0[7],
			b0[4], b0[5], b0[6], b0[7]
		};

		// pass 2
		const vect8_float a0_2 = (vect8_float) {
			a0[2], a0[2], a0[2], a0[2],
			a0[6], a0[6], a0[6], a0[6]
		};
		const vect8_float a1_2 = (vect8_float) {
			a1[2], a1[2], a1[2], a1[2],
			a1[6], a1[6], a1[6], a1[6]
		};
		const vect8_float b_2 = (vect8_float) {
			b1[0], b1[1], b1[2], b1[3],
			b1[0], b1[1], b1[2], b1[3]
		};

		// pass 3
		const vect8_float a0_3 = (vect8_float) {
			a0[3], a0[3], a0[3], a0[3],
			a0[7], a0[7], a0[7], a0[7]
		};
		const vect8_float a1_3 = (vect8_float) {
			a1[3], a1[3], a1[3], a1[3],
			a1[7], a1[7], a1[7], a1[7]
		};
		const vect8_float b_3 = (vect8_float) {
			b1[4], b1[5], b1[6], b1[7],
			b1[4], b1[5], b1[6], b1[7]
		};

		n[0] =
			a0_0 * b_0 +
			a0_1 * b_1 +
			a0_2 * b_2 +
			a0_3 * b_3;

		n[1] =
			a1_0 * b_0 +
			a1_1 * b_1 +
			a1_2 * b_2 +
			a1_3 * b_3;

#elif SIMD_AUTOVECT == SIMD_16WAY
		const vect16_float ai0 = (vect16_float) {
			mat0.m[0][0], mat0.m[0][0], mat0.m[0][0], mat0.m[0][0],
			mat0.m[1][0], mat0.m[1][0], mat0.m[1][0], mat0.m[1][0],
			mat0.m[2][0], mat0.m[2][0], mat0.m[2][0], mat0.m[2][0],
			mat0.m[3][0], mat0.m[3][0], mat0.m[3][0], mat0.m[3][0]
		};
		const vect4_float b0 = (vect4_float) {
			mat1.m[0][0], mat1.m[0][1], mat1.m[0][2], mat1.m[0][3]
		};

		n = ai0 *
			(vect16_float) {
				b0[0], b0[1], b0[2], b0[3],
				b0[0], b0[1], b0[2], b0[3],
				b0[0], b0[1], b0[2], b0[3],
				b0[0], b0[1], b0[2], b0[3] };

		for (unsigned j = 1; j < 4; ++j) {

			const vect16_float aij = (vect16_float) {
				mat0.m[0][j], mat0.m[0][j], mat0.m[0][j], mat0.m[0][j],
				mat0.m[1][j], mat0.m[1][j], mat0.m[1][j], mat0.m[1][j],
				mat0.m[2][j], mat0.m[2][j], mat0.m[2][j], mat0.m[2][j],
				mat0.m[3][j], mat0.m[3][j], mat0.m[3][j], mat0.m[3][j]
			};
			const vect4_float bj = (vect4_float) {
				mat1.m[j][0], mat1.m[j][1], mat1.m[j][2], mat1.m[j][3]
			};

			n += aij *
				(vect16_float) {
					bj[0], bj[1], bj[2], bj[3],
					bj[0], bj[1], bj[2], bj[3],
					bj[0], bj[1], bj[2], bj[3],
					bj[0], bj[1], bj[2], bj[3] };
		}

#else // scalar (SIMD_AUTOVECT == SIMD_1WAY)
		for (unsigned i = 0; i < 4; ++i) {

			const float e0 = mat0.m[i][0];

#if SIMD_MANUAL_UNROLL == 0
			for (unsigned j = 0; j < 4; ++j)
				m[i][j] = e0 * mat1.m[0][j];

			for (unsigned j = 1; j < 4; ++j) {

				const float ej = mat0.m[i][j];

				for (unsigned k = 0; k < 4; ++k)
					m[i][k] += ej * mat1.m[j][k];
			}

#else
			const float e1 = mat0.m[i][1];
			const float e2 = mat0.m[i][2];
			const float e3 = mat0.m[i][3];

			float col0 = e0 * mat1.m[0][0];
			float col1 = e0 * mat1.m[0][1];
			float col2 = e0 * mat1.m[0][2];
			float col3 = e0 * mat1.m[0][3];

			col0 += e1 * mat1.m[1][0];
			col1 += e1 * mat1.m[1][1];
			col2 += e1 * mat1.m[1][2];
			col3 += e1 * mat1.m[1][3];

			col0 += e2 * mat1.m[2][0];
			col1 += e2 * mat1.m[2][1];
			col2 += e2 * mat1.m[2][2];
			col3 += e2 * mat1.m[2][3];

			col0 += e3 * mat1.m[3][0];
			col1 += e3 * mat1.m[3][1];
			col2 += e3 * mat1.m[3][2];
			col3 += e3 * mat1.m[3][3];

			m[i][0] = col0;
			m[i][1] = col1;
			m[i][2] = col2;
			m[i][3] = col3;

#endif
		}

#endif // SIMD_INTRINSICS == 0

		return *this;
	}

#endif // SIMD_ETALON_ALT*
};

} // namespace etal

#endif

#if SIMD_ETALON
namespace space = etal;

#elif SIMD_SCALAR
namespace space = scal;

namespace scal {

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
namespace space = simd;

#endif

std::istream& operator >> (
	std::istream& str,
	space::matx4& a) {

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

	a = space::matx4(t);

	return str;
}


class formatter {

	const float m;

public:
	formatter(const float& a)
	: m(a) {
	}

	float get() const {
		return m;
	}
};


std::ostream& operator << (
	std::ostream& str,
	const formatter& a) {

	return str << std::setw(12) << std::setfill('_') << a.get();
}


std::ostream& operator << (
	std::ostream& str,
	const space::matx4& a) {

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

space::matx4 ma[2] __attribute__ ((aligned(CACHELINE_SIZE))) = {
	space::matx4(
		 1,  0,  0,  0,
		 0,  1,  0,  0,
		 0,  0,  1,  0,
		 0,  0,  0,  1),
	space::matx4(
		 1,  2,  3,  4,
		 5,  6,  7,  8,
		 9, 10, 11, 12,
		13, 14, 15, 16)
};
space::matx4 ra[nthreads] __attribute__ ((aligned(CACHELINE_SIZE)));

template < bool >
struct compile_assert;

template <>
struct compile_assert< true > {
	compile_assert() {}
};

static compile_assert< sizeof(space::matx4) / CACHELINE_SIZE * CACHELINE_SIZE == sizeof(space::matx4) > assert_ra_element_size;

enum {
	BARRIER_START,
	BARRIER_FINISH,
	BARRIER_COUNT
};

static pthread_barrier_t barrier[BARRIER_COUNT];

struct compute_arg {

	pthread_t thread;
	size_t id;
	uint64_t dt;

	compute_arg()
	: thread(0)
	, id(0)
	, dt(0) {
	}

	compute_arg(
		const size_t arg_id)
	: thread(0)
	, id(arg_id)
	, dt(0) {
	}
};

size_t obfuscator; // invariance obfuscator

static inline __attribute__ ((always_inline)) void workload(
	const size_t id,
	const size_t count) {

	const size_t offset = obfuscator;

	for (size_t i = 0; i < count; ++i) {
		const size_t offs0 = i * offset + 0;
		const size_t offs1 = i * offset + 1;

#if SIMD_WORKLOAD_ITERATION_DENSITY_X4
		const size_t offs2 = (i + 1) * offset + 0;
		const size_t offs3 = (i + 1) * offset + 1;
		const size_t offs4 = (i + 2) * offset + 0;
		const size_t offs5 = (i + 2) * offset + 1;
		const size_t offs6 = (i + 3) * offset + 0;
		const size_t offs7 = (i + 3) * offset + 1;

#endif
		// issue with intel compiler (icpc 14.0.4 20140805) where it won't amortize the temporary;
		// it gets written on the stack first and only then moved to the destination
#if __MIC__ == 0
		ra[id + offs0] = space::matx4().mul(ma[offs0], ma[offs1]);

#if SIMD_WORKLOAD_ITERATION_DENSITY_X4
		ra[id + offs2] = space::matx4().mul(ma[offs2], ma[offs3]);
		ra[id + offs4] = space::matx4().mul(ma[offs4], ma[offs5]);
		ra[id + offs6] = space::matx4().mul(ma[offs6], ma[offs7]);

#endif
#else
		ra[id + offs0].mul(ma[offs0], ma[offs1]);

#if SIMD_WORKLOAD_ITERATION_DENSITY_X4
		ra[id + offs2].mul(ma[offs2], ma[offs3]);
		ra[id + offs4].mul(ma[offs4], ma[offs5]);
		ra[id + offs6].mul(ma[offs6], ma[offs7]);

#endif
#endif
	}
}

static void*
compute(
	void* arg) {

	compute_arg* const carg = reinterpret_cast< compute_arg* >(arg);
	const size_t id = carg->id;

	pthread_barrier_wait(barrier + BARRIER_START);

	// warm up the engines
	workload(id, reps / 10);

	const uint64_t t0 = timer_ns();
	workload(id, reps);

	const uint64_t dt = timer_ns() - t0;
	carg->dt = dt;

	pthread_barrier_wait(barrier + BARRIER_FINISH);

	return 0;
}


class workforce_t {

	compute_arg record[one_less];
	bool successfully_init;

public:
	workforce_t();
	~workforce_t();

	bool is_successfully_init() const {
		return successfully_init;
	}

	uint64_t get_dt(const size_t i) const {
		assert(one_less > i);
		return record[i].dt;
	}
};


static void
report_err(
	const char* const func,
	const int line,
	const size_t counter,
	const int err) {

	std::cerr << func << ':' << line << ", i: "
		<< counter << ", err: " << err << std::endl;
}


workforce_t::workforce_t()
: successfully_init(false) {

	for (size_t i = 0; i < sizeof(barrier) / sizeof(barrier[0]); ++i) {
		const int r = pthread_barrier_init(barrier + i, NULL, nthreads);

		if (0 != r) {
			report_err(__FUNCTION__, __LINE__, i, r);
			return;
		}
	}

	for (size_t i = 0; i < one_less; ++i) {
		const size_t id = i + 1;
		record[i] = compute_arg(id);

#if SIMD_THREAD_AFFINITY
		struct scoped_t {
			pthread_attr_t attr;
			bool successfully_init;

			scoped_t(
				const size_t i)
			: successfully_init(false) {

				const int r = pthread_attr_init(&attr);

				if (0 != r) {
					report_err(__FUNCTION__, __LINE__, i, r);
					return;
				}

				successfully_init = true;
			}

			~scoped_t() {
				if (successfully_init)
					pthread_attr_destroy(&attr);
			}
		};

		scoped_t scoped(i);

		if (!scoped.successfully_init)
			return;

		cpu_set_t affin;
		CPU_ZERO(&affin);
		CPU_SET(id * SIMD_THREAD_AFFINITY, &affin);

		const int ra = pthread_attr_setaffinity_np(&scoped.attr, sizeof(affin), &affin);

		if (0 != ra) {
			report_err(__FUNCTION__, __LINE__, i, ra);
			return;
		}

		const int r = pthread_create(&record[i].thread, &scoped.attr, compute, record + i);

#else // SIMD_THREAD_AFFINITY == 0
		const int r = pthread_create(&record[i].thread, 0, compute, record + i);

#endif // SIMD_THREAD_AFFINITY

		if (0 != r) {
			report_err(__FUNCTION__, __LINE__, i, r);
			return;
		}
	}

	successfully_init = true;
}


workforce_t::~workforce_t() {

	for (size_t i = 0; i < sizeof(barrier) / sizeof(barrier[0]); ++i) {
		const int r = pthread_barrier_destroy(barrier + i);

		if (0 != r)
			report_err(__FUNCTION__, __LINE__, i, r);
	}

	for (size_t i = 0; i < one_less && 0 != record[i].thread; ++i) {
		const int r = pthread_join(record[i].thread, NULL);

		if (0 != r)
			report_err(__FUNCTION__, __LINE__, i, r);
	}
}


class formatter_float {

	union {
		uint32_t u;
		float f;
	};

public:
	formatter_float(const float a)
	: f(a) {
	}

	uint32_t get() const {
		return u;
	}
};

static std::ostream& operator << (
	std::ostream& str,
	const formatter_float a) {

	str << 
		(a.get() >> 31) << ":" <<
		std::setw(2) << std::setfill('0') << (a.get() >> 23 & 0xff) << ":" <<
		std::setw(6) << std::setfill('0') << (a.get() >>  0 & 0x7fffff);
	return str;
}

#if SIMD_SCALAR
namespace conf = scal;

#else
namespace conf = simd;

#endif

#if SIMD_ETALON == 0
static std::ostream& operator << (
	std::ostream& str,
	const conf::vect2& a) {

	return str <<
		formatter_float(a.get(0)) << " " << formatter_float(a.get(1));
}

static std::ostream& operator << (
	std::ostream& str,
	const conf::vect3& a) {

	return str <<
		formatter_float(a.get(0)) << " " << formatter_float(a.get(1)) << " " << formatter_float(a.get(2));
}

static std::ostream& operator << (
	std::ostream& str,
	const conf::vect4& a) {

	return str <<
		formatter_float(a.get(0)) << " " << formatter_float(a.get(1)) << " " << formatter_float(a.get(2)) << " " << formatter_float(a.get(3));
}

static bool
conformance() {

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
		v4 != vect4(1.f, 2.f, 3.f, 4.f)) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false negative
	if (v2 != vect2(1.f, -2.f) &&
		v3 != vect3(1.f, 2.f, -3.f) &&
		v4 != vect4(1.f, 2.f, 3.f, -4.f)) {
	}
	else {
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false positive
	if (v2 == vect2(1.f, -2.f) ||
		v3 == vect3(1.f, 2.f, -3.f) ||
		v4 == vect4(1.f, 2.f, 3.f, -4.f)) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false negative
	if (v2 == vect2(1.f, 2.f) &&
		v3 == vect3(1.f, 2.f, 3.f) &&
		v4 == vect4(1.f, 2.f, 3.f, 4.f)) {
	}
	else {
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false positive
	if (iv2 != ivect2(1, 2) ||
		iv3 != ivect3(1, 2, 3) ||
		iv4 != ivect4(1, 2, 3, 4)) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false negative
	if (iv2 != ivect2(1, -2) &&
		iv3 != ivect3(1, 2, -3) &&
		iv4 != ivect4(1, 2, 3, -4)) {
	}
	else {
		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false positive
	if (iv2 == ivect2(1, -2) ||
		iv3 == ivect3(1, 2, -3) ||
		iv4 == ivect4(1, 2, 3, -4)) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// test for false negative
	if (iv2 == ivect2(1, 2) &&
		iv3 == ivect3(1, 2, 3) &&
		iv4 == ivect4(1, 2, 3, 4)) {
	}
	else {
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
		precision4[3] != -2.f - 2.f / (1 << 23)) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	const float f[] = { 42.f, 43.f, 44.f, 45.f };
	const int i[] = { 42, 43, 44, 45 };

	if (vect2(*reinterpret_cast< const float(*)[2] >(f)).negate() != vect2(-f[0], -f[1]) ||
		vect3(*reinterpret_cast< const float(*)[3] >(f)).negate() != vect3(-f[0], -f[1], -f[2]) ||
		vect4(*reinterpret_cast< const float(*)[4] >(f)).negate() != vect4(-f[0], -f[1], -f[2], -f[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2(*reinterpret_cast< const int(*)[2] >(i)).negate() != ivect2(-i[0], -i[1]) ||
		ivect3(*reinterpret_cast< const int(*)[3] >(i)).negate() != ivect3(-i[0], -i[1], -i[2]) ||
		ivect4(*reinterpret_cast< const int(*)[4] >(i)).negate() != ivect4(-i[0], -i[1], -i[2], -i[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().add(v2, vect2(*reinterpret_cast< const float(*)[2] >(f))) != vect2(v2[0] + f[0], v2[1] + f[1]) ||
		vect3().add(v3, vect3(*reinterpret_cast< const float(*)[3] >(f))) != vect3(v3[0] + f[0], v3[1] + f[1], v3[2] + f[2]) ||
		vect4().add(v4, vect4(*reinterpret_cast< const float(*)[4] >(f))) != vect4(v4[0] + f[0], v4[1] + f[1], v4[2] + f[2], v4[3] + f[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().sub(v2, vect2(*reinterpret_cast< const float(*)[2] >(f))) != vect2(v2[0] - f[0], v2[1] - f[1]) ||
		vect3().sub(v3, vect3(*reinterpret_cast< const float(*)[3] >(f))) != vect3(v3[0] - f[0], v3[1] - f[1], v3[2] - f[2]) ||
		vect4().sub(v4, vect4(*reinterpret_cast< const float(*)[4] >(f))) != vect4(v4[0] - f[0], v4[1] - f[1], v4[2] - f[2], v4[3] - f[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2().add(iv2, ivect2(*reinterpret_cast< const int(*)[2] >(i))) != ivect2(iv2[0] + i[0], iv2[1] + i[1]) ||
		ivect3().add(iv3, ivect3(*reinterpret_cast< const int(*)[3] >(i))) != ivect3(iv3[0] + i[0], iv3[1] + i[1], iv3[2] + i[2]) ||
		ivect4().add(iv4, ivect4(*reinterpret_cast< const int(*)[4] >(i))) != ivect4(iv4[0] + i[0], iv4[1] + i[1], iv4[2] + i[2], iv4[3] + i[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2().sub(iv2, ivect2(*reinterpret_cast< const int(*)[2] >(i))) != ivect2(iv2[0] - i[0], iv2[1] - i[1]) ||
		ivect3().sub(iv3, ivect3(*reinterpret_cast< const int(*)[3] >(i))) != ivect3(iv3[0] - i[0], iv3[1] - i[1], iv3[2] - i[2]) ||
		ivect4().sub(iv4, ivect4(*reinterpret_cast< const int(*)[4] >(i))) != ivect4(iv4[0] - i[0], iv4[1] - i[1], iv4[2] - i[2], iv4[3] - i[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().mul(v2, vect2(*reinterpret_cast< const float(*)[2] >(f))) != vect2(v2[0] * f[0], v2[1] * f[1]) ||
		vect3().mul(v3, vect3(*reinterpret_cast< const float(*)[3] >(f))) != vect3(v3[0] * f[0], v3[1] * f[1], v3[2] * f[2]) ||
		vect4().mul(v4, vect4(*reinterpret_cast< const float(*)[4] >(f))) != vect4(v4[0] * f[0], v4[1] * f[1], v4[2] * f[2], v4[3] * f[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().div(v2, vect2(*reinterpret_cast< const float(*)[2] >(f))) != vect2(v2[0] / f[0], v2[1] / f[1]) ||
		vect3().div(v3, vect3(*reinterpret_cast< const float(*)[3] >(f))) != vect3(v3[0] / f[0], v3[1] / f[1], v3[2] / f[2]) ||
		vect4().div(v4, vect4(*reinterpret_cast< const float(*)[4] >(f))) != vect4(v4[0] / f[0], v4[1] / f[1], v4[2] / f[2], v4[3] / f[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;

		std::cout <<
			vect2().div(v2, vect2(*reinterpret_cast< const float(*)[2] >(f))) << std::endl <<
			vect2(v2[0] / f[0], v2[1] / f[1]) << std::endl;

		std::cout <<
			vect3().div(v3, vect3(*reinterpret_cast< const float(*)[3] >(f))) << std::endl <<
			vect3(v3[0] / f[0], v3[1] / f[1], v3[2] / f[2]) << std::endl;

		std::cout <<
			vect4().div(v4, vect4(*reinterpret_cast< const float(*)[4] >(f))) << std::endl <<
			vect4(v4[0] / f[0], v4[1] / f[1], v4[2] / f[2], v4[3] / f[3]) << std::endl;
	}

	if (vect2().div(v2, precision2) != vect2(v2[0] / precision2[0], v2[1] / precision2[1]) ||
		vect3().div(v3, precision3) != vect3(v3[0] / precision3[0], v3[1] / precision3[1], v3[2] / precision3[2]) ||
		vect4().div(v4, precision4) != vect4(v4[0] / precision4[0], v4[1] / precision4[1], v4[2] / precision4[2], v4[3] / precision4[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;

		std::cout <<
			vect2().div(v2, precision2) << std::endl <<
			vect2(v2[0] / precision2[0], v2[1] / precision2[1]) << std::endl;

		std::cout <<
			vect3().div(v3, precision3) << std::endl <<
			vect3(v3[0] / precision3[0], v3[1] / precision3[1], v3[2] / precision3[2]) << std::endl;

		std::cout <<
			vect4().div(v4, precision4) << std::endl <<
			vect4(v4[0] / precision4[0], v4[1] / precision4[1], v4[2] / precision4[2], v4[3] / precision4[3]) << std::endl;
	}

	if (ivect2().mul(iv2, ivect2(*reinterpret_cast< const int(*)[2] >(i))) != ivect2(iv2[0] * i[0], iv2[1] * i[1]) ||
		ivect3().mul(iv3, ivect3(*reinterpret_cast< const int(*)[3] >(i))) != ivect3(iv3[0] * i[0], iv3[1] * i[1], iv3[2] * i[2]) ||
		ivect4().mul(iv4, ivect4(*reinterpret_cast< const int(*)[4] >(i))) != ivect4(iv4[0] * i[0], iv4[1] * i[1], iv4[2] * i[2], iv4[3] * i[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2().div(ivect2(*reinterpret_cast< const int(*)[2] >(i)), iv2) != ivect2(i[0] / iv2[0], i[1] / iv2[1]) ||
		ivect3().div(ivect3(*reinterpret_cast< const int(*)[3] >(i)), iv3) != ivect3(i[0] / iv3[0], i[1] / iv3[1], i[2] / iv3[2]) ||
		ivect4().div(ivect4(*reinterpret_cast< const int(*)[4] >(i)), iv4) != ivect4(i[0] / iv4[0], i[1] / iv4[1], i[2] / iv4[2], i[3] / iv4[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().mad(v2, v2, vect2(*reinterpret_cast< const float(*)[2] >(f))) != vect2(v2[0] * v2[0] + f[0], v2[1] * v2[1] + f[1]) ||
		vect3().mad(v3, v3, vect3(*reinterpret_cast< const float(*)[3] >(f))) != vect3(v3[0] * v3[0] + f[0], v3[1] * v3[1] + f[1], v3[2] * v3[2] + f[2]) ||
		vect4().mad(v4, v4, vect4(*reinterpret_cast< const float(*)[4] >(f))) != vect4(v4[0] * v4[0] + f[0], v4[1] * v4[1] + f[1], v4[2] * v4[2] + f[2], v4[3] * v4[3] + f[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2().mad(iv2, iv2, ivect2(*reinterpret_cast< const int(*)[2] >(i))) != ivect2(iv2[0] * iv2[0] + i[0], iv2[1] * iv2[1] + i[1]) ||
		ivect3().mad(iv3, iv3, ivect3(*reinterpret_cast< const int(*)[3] >(i))) != ivect3(iv3[0] * iv3[0] + i[0], iv3[1] * iv3[1] + i[1], iv3[2] * iv3[2] + i[2]) ||
		ivect4().mad(iv4, iv4, ivect4(*reinterpret_cast< const int(*)[4] >(i))) != ivect4(iv4[0] * iv4[0] + i[0], iv4[1] * iv4[1] + i[1], iv4[2] * iv4[2] + i[2], iv4[3] * iv4[3] + i[3])) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().wsum(v2, v2, .25f, -.25f) != vect2(0.f, 0.f) ||
		vect3().wsum(v3, v3, .25f, -.25f) != vect3(0.f, 0.f, 0.f) ||
		vect4().wsum(v4, v4, .25f, -.25f) != vect4(0.f, 0.f, 0.f, 0.f)) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (ivect2().wsum(iv2, iv2, 25, -25) != ivect2(0, 0) ||
		ivect3().wsum(iv3, iv3, 25, -25) != ivect3(0, 0, 0) ||
		ivect4().wsum(iv4, iv4, 25, -25) != ivect4(0, 0, 0, 0)) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	if (vect2().cross(vect2(1.f, 0.f), vect2(0.f, 1.f)) != vect2(1.f, 1.f) ||
		vect3().cross(vect3(1.f, 0.f, 0.f), vect3(0.f, 1.f, 0.f)) != vect3(0.f, 0.f, 1.f) ||
		vect4().cross(vect4(1.f, 0.f, 0.f, 0.f), vect4(0.f, 1.f, 0.f, 0.f)) != vect4(0.f, 0.f, 1.f, 0.f)) {

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
		 4.f,  8.f, 12.f, 16.f)) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	// angle = pi, axis = { 0, 0, 1 }
	if (vect3(1.f, 0.f, 0.f).mul(matx3(quat(0.f, 0.f, 1.f, 0.f))) != vect3(-1.f, 0.f, 0.f) ||
		vect4(1.f, 0.f, 0.f, 1.f).mul(matx4(quat(0.f, 0.f, 1.f, 0.f))) != vect4(-1.f, 0.f, 0.f, 1.f)) {

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
		0.f,  0.f, 0.f, 1.f)) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;
	}

	const vect3 nv3(1.f, 2.f, 4.f);
	const vect4 nv4(1.f, 2.f, 4.f, 8.f);

	if (vect3().normalise(nv3) != vect3(nv3[0] / nv3.norm(), nv3[1] / nv3.norm(), nv3[2] / nv3.norm()) ||
		vect4().normalise(nv4) != vect4(nv4[0] / nv4.norm(), nv4[1] / nv4.norm(), nv4[2] / nv4.norm(), nv4[3] / nv4.norm())) {

		std::cerr << "failed test at line " << __LINE__ << std::endl;
		success = false;

		std::cout <<
			vect3(nv3[0] / nv3.norm(), nv3[1] / nv3.norm(), nv3[2] / nv3.norm()).normalise() << std::endl <<
			vect3().normalise(nv3) << std::endl;

		std::cout <<
			vect4(nv4[0] / nv4.norm(), nv4[1] / nv4.norm(), nv4[2] / nv4.norm(), nv4[3] / nv4.norm()).normalise() << std::endl <<
			vect4().normalise(nv4) << std::endl;
	}

	return success;
}

#endif // SIMD_ETALON == 0

int main(
	int argc,
	char** argv) {

#if SIMD_ETALON == 0 && SIMD_TEST_CONFORMANCE != 0
	std::cout << "conformance test.." << std::endl;
	std::cout << (conformance() ? "passed" : "failed") << std::endl;

#endif
	std::cout << "performance test.." << std::endl;

	std::ifstream in("vect.input");

	if (in.is_open()) {
		in >> ma[0];
		in >> ma[1];
		in.close();
	}

	const workforce_t workforce;

	if (!workforce.is_successfully_init()) {
		std::cerr << "failed to raise workforce; bailing out" << std::endl;
		return -1;
	}

	compute_arg carg(0);

	compute(&carg);

	double sec = carg.dt * 1e-9;

	std::cout << "elapsed time: " << sec << " s";

	for (size_t i = 0; i < one_less; ++i) {
		const double isec = workforce.get_dt(i) * 1e-9;
		std::cout << ", " << isec << " s";
		sec += isec;
	}

	std::cout << " (" << sec / nthreads << " s)" << std::endl;

	for (size_t i = 0; i < sizeof(ra) / sizeof(ra[0]); ++i)
		std::cout << ra[i];

	return 0;
}
