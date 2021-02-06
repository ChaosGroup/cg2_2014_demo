////////////////////////////////////////////////////////////////////////////////
// AVX implementation of sin, cos, exp and log
//
// A direct translation of Julien Pommier's cephes/SSE2 to AVX
// Original copyright notice follows.
//
////////////////////////////////////////////////////////////////////////////////

/* SIMD (SSE2) implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/

/* Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#ifndef cephes_math_avx_H__
#define cephes_math_avx_H__

#include <immintrin.h>

////////////////////////////////////////////////////////////////////////////////
// mk: remark on constant declarations:
//
// to keep the following constants in .rodata section, declare them as const
// arrays of POD; declaring them as const SSE types would place them in .bss,
// where they would be initialized at 'early construction'; .rodata is a better
// location as it minimizes the chances for adjacency to vars, and thus for
// false sharing (scenario was hit under clang 3.4 - 3.5)
//
////////////////////////////////////////////////////////////////////////////////

#if defined(__GNUC__)
#define _PS_CONST(Name, Val) \
	static const float pf32_##Name[8] __attribute__ ((aligned(32))) = { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PI32_CONST(Name, Val) \
	static const int pi32_##Name[8] __attribute__ ((aligned(32))) = { (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val) }
#define _PS_CONST_INT(Name, Val) \
	static const int pf32_##Name[8] __attribute__ ((aligned(32))) = { (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val) }

#else
#define _PS_CONST(Name, Val) \
	static const float __declspec (align(32)) pf32_##Name[8] = { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PI32_CONST(Name, Val) \
	static const int __declspec (align(32)) pi32_##Name[8] = { (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val) }
#define _PS_CONST_INT(Name, Val) \
	static const int __declspec (align(32)) pf32_##Name[8] = { (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val), (int)(Val) }

#endif

_PS_CONST(1  , 1.0f);
_PS_CONST(0p5, 0.5f);

/* the smallest non denormalized float number */
_PS_CONST_INT(min_norm_pos, 0x00800000);
_PS_CONST_INT(mant_mask, 0x7f800000);
_PS_CONST_INT(inv_mant_mask, ~0x7f800000);
_PS_CONST_INT(sign_mask, 0x80000000);
_PS_CONST_INT(inv_sign_mask, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS_CONST(cephes_log_p0, 7.0376836292E-2);
_PS_CONST(cephes_log_p1, -1.1514610310E-1);
_PS_CONST(cephes_log_p2, 1.1676998740E-1);
_PS_CONST(cephes_log_p3, -1.2420140846E-1);
_PS_CONST(cephes_log_p4, +1.4249322787E-1);
_PS_CONST(cephes_log_p5, -1.6668057665E-1);
_PS_CONST(cephes_log_p6, +2.0000714765E-1);
_PS_CONST(cephes_log_p7, -2.4999993993E-1);
_PS_CONST(cephes_log_p8, +3.3333331174E-1);
_PS_CONST(cephes_log_q1, -2.12194440e-4);
_PS_CONST(cephes_log_q2, 0.693359375);

_PS_CONST(exp_hi, 88.3762626647949f);
_PS_CONST(exp_lo, -88.3762626647949f);
_PS_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS_CONST(cephes_exp_C1, 0.693359375);
_PS_CONST(cephes_exp_C2, -2.12194440e-4);
_PS_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1);

_PS_CONST(minus_cephes_DP1, -0.78515625);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS_CONST(sincof_p0, -1.9515295891E-4);
_PS_CONST(sincof_p1,  8.3321608736E-3);
_PS_CONST(sincof_p2, -1.6666654611E-1);
_PS_CONST(coscof_p0,  2.443315711809948E-005);
_PS_CONST(coscof_p1, -1.388731625493765E-003);
_PS_CONST(coscof_p2,  4.166664568298827E-002);
_PS_CONST(cephes_FOPI, 1.27323954473516);

typedef __m256 v8sf;
typedef __m256i v8si;

/* natural logarithm computed for 4 simultaneous float 
   return NaN for x <= 0
*/
inline v8sf cephes_log(v8sf x) {
	v8si emm0;
	v8sf one = *(v8sf*)pf32_1;

	v8sf invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ);

	/* cut off denormalized stuff */
	x = _mm256_max_ps(x, *(v8sf*)pf32_min_norm_pos);

#if __AVX2__ != 0
	emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
	emm0 = _mm256_sub_epi32(emm0, *(v8si*)pi32_0x7f);

#else
	const __m128i tmm0 = _mm256_castsi256_si128(_mm256_castps_si256(x));
	const __m128i tmm1 = _mm256_extractf128_si256(_mm256_castps_si256(x), 1);
	const __m128i tmm2 = _mm_srli_epi32(tmm0, 23);
	const __m128i tmm3 = _mm_srli_epi32(tmm1, 23);
	const __m128i tmm4 = _mm_sub_epi32(tmm2, *(const __m128i*)pi32_0x7f);
	const __m128i tmm5 = _mm_sub_epi32(tmm3, *(const __m128i*)pi32_0x7f);
	emm0 = _mm256_insertf128_si256(_mm256_castsi128_si256(tmm4), tmm5, 1);

#endif
	/* keep only the fractional part */
	x = _mm256_and_ps(x, *(v8sf*)pf32_inv_mant_mask);
	x = _mm256_or_ps(x, *(v8sf*)pf32_0p5);

	v8sf e = _mm256_cvtepi32_ps(emm0);

	e = _mm256_add_ps(e, one);

	/* part2: 
	   if( x < SQRTHF ) {
		 e -= 1;
		 x = x + x - 1.0;
	   } else { x = x - 1.0; }
	*/
	v8sf mask = _mm256_cmp_ps(x, *(v8sf*)pf32_cephes_SQRTHF, _CMP_LT_OQ);
	v8sf tmp = _mm256_and_ps(x, mask);
	x = _mm256_sub_ps(x, one);
	e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
	x = _mm256_add_ps(x, tmp);

	v8sf z = _mm256_mul_ps(x, x);

	v8sf y = *(v8sf*)pf32_cephes_log_p0;
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_log_p1);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_log_p2);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_log_p3);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_log_p4);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_log_p5);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_log_p6);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_log_p7);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_log_p8);
	y = _mm256_mul_ps(y, x);

	y = _mm256_mul_ps(y, z);

	tmp = _mm256_mul_ps(e, *(v8sf*)pf32_cephes_log_q1);
	y = _mm256_add_ps(y, tmp);

	tmp = _mm256_mul_ps(z, *(v8sf*)pf32_0p5);
	y = _mm256_sub_ps(y, tmp);

	tmp = _mm256_mul_ps(e, *(v8sf*)pf32_cephes_log_q2);
	x = _mm256_add_ps(x, y);
	x = _mm256_add_ps(x, tmp);
	x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
	return x;
}

inline v8sf cephes_exp(v8sf x) {
	v8sf tmp = _mm256_setzero_ps(), fx;
	v8si emm0;
	v8sf one = *(v8sf*)pf32_1;

	x = _mm256_min_ps(x, *(v8sf*)pf32_exp_hi);
	x = _mm256_max_ps(x, *(v8sf*)pf32_exp_lo);

	/* express exp(x) as exp(g + n*log(2)) */
	fx = _mm256_mul_ps(x, *(v8sf*)pf32_cephes_LOG2EF);
	fx = _mm256_add_ps(fx, *(v8sf*)pf32_0p5);

	/* how to perform a floorf with SSE2: just below */
	emm0 = _mm256_cvttps_epi32(fx);
	tmp  = _mm256_cvtepi32_ps(emm0);

	/* if greater, substract 1 */
	v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OQ);
	mask = _mm256_and_ps(mask, one);
	fx = _mm256_sub_ps(tmp, mask);

	tmp = _mm256_mul_ps(fx, *(v8sf*)pf32_cephes_exp_C1);
	v8sf z = _mm256_mul_ps(fx, *(v8sf*)pf32_cephes_exp_C2);
	x = _mm256_sub_ps(x, tmp);
	x = _mm256_sub_ps(x, z);

	z = _mm256_mul_ps(x,x);

	v8sf y = *(v8sf*)pf32_cephes_exp_p0;
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_exp_p1);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_exp_p2);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_exp_p3);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_exp_p4);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)pf32_cephes_exp_p5);
	y = _mm256_mul_ps(y, z);
	y = _mm256_add_ps(y, x);
	y = _mm256_add_ps(y, one);

	/* build 2^n */
	emm0 = _mm256_cvttps_epi32(fx);

#if __AVX2__ != 0
	emm0 = _mm256_add_epi32(emm0, *(v8si*)pi32_0x7f);
	emm0 = _mm256_slli_epi32(emm0, 23);

#else
	const __m128i tmm0 = _mm256_castsi256_si128(emm0);
	const __m128i tmm1 = _mm256_extractf128_si256(emm0, 1);
	const __m128i tmm2 = _mm_add_epi32(tmm0, *(const __m128i*)pi32_0x7f);
	const __m128i tmm3 = _mm_add_epi32(tmm1, *(const __m128i*)pi32_0x7f);
	const __m128i tmm4 = _mm_slli_epi32(tmm2, 23);
	const __m128i tmm5 = _mm_slli_epi32(tmm3, 23);
	emm0 = _mm256_insertf128_si256(_mm256_castsi128_si256(tmm4), tmm5, 1);

#endif
	v8sf pow2n = _mm256_castsi256_ps(emm0);

	y = _mm256_mul_ps(y, pow2n);
	return y;
}

////////////////////////////////////////////////////////////////////////////////
// mk: raise four non-negative bases to the four exponents
//
////////////////////////////////////////////////////////////////////////////////

inline v8sf cephes_pow(const v8sf x, const v8sf y) {
	const v8sf zero_filter = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ);
	return _mm256_andnot_ps(zero_filter, cephes_exp(_mm256_mul_ps(cephes_log(x), y)));
}

/* evaluation of 4 sines at onces, using only SSE1+MMX intrinsics so
   it runs also on old athlons XPs and the pentium III of your grand
   mother.

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

   Performance is also surprisingly good, 1.33 times faster than the
   macos vsinf SSE2 function, and 1.5 times faster than the
   __vrs4_sinf of amd's ACML (which is only available in 64 bits). Not
   too bad for an SSE1 function (with no special tuning) !
   However the latter libraries probably have a much better handling of NaN,
   Inf, denormalized and other special arguments..

   On my core 1 duo, the execution of this function takes approximately 95 cycles.

   From what I have observed on the experiments with Intel AMath lib, switching to an
   SSE2 version would improve the perf by only 10%.

   Since it is based on SSE intrinsics, it has to be compiled at -O2 to
   deliver full speed.
*/

inline v8sf cephes_sin(v8sf x) {
	v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, sign_bit, y;
	v8si emm0, emm2;
	sign_bit = x;

	/* take the absolute value */
	x = _mm256_and_ps(x, *(v8sf*)pf32_inv_sign_mask);

	/* extract the sign bit (upper one) */
	sign_bit = _mm256_and_ps(sign_bit, *(v8sf*)pf32_sign_mask);

	/* scale by 4/Pi */
	y = _mm256_mul_ps(x, *(v8sf*)pf32_cephes_FOPI);

	/* store the integer part of y in mm0 */
	emm2 = _mm256_cvttps_epi32(y);

	/* j=(j+1) & (~1) (see the cephes sources) */
#if __AVX2__ != 0
	emm2 = _mm256_add_epi32(emm2, *(v8si*)pi32_1);
	emm2 = _mm256_and_si256(emm2, *(v8si*)pi32_inv1);

#else
	const __m128i tmm0 = _mm256_castsi256_si128(emm2);
	const __m128i tmm1 = _mm256_extractf128_si256(emm2, 1);
	const __m128i tmm2 = _mm_add_epi32(tmm0, *(const __m128i*)pi32_1);
	const __m128i tmm3 = _mm_add_epi32(tmm1, *(const __m128i*)pi32_1);
	const __m128i tmm4 = _mm_and_si128(tmm2, *(const __m128i*)pi32_inv1);
	const __m128i tmm5 = _mm_and_si128(tmm3, *(const __m128i*)pi32_inv1);
	emm2 = _mm256_insertf128_si256(_mm256_castsi128_si256(tmm4), tmm5, 1);

#endif
	y = _mm256_cvtepi32_ps(emm2);

	/* get the swap sign flag */

#if __AVX2__ != 0
	emm0 = _mm256_and_si256(emm2, *(v8si*)pi32_4);
	emm0 = _mm256_slli_epi32(emm0, 29);

#else
	const __m128i tmm6 = _mm_and_si128(tmm4, *(const __m128i*)pi32_4);
	const __m128i tmm7 = _mm_and_si128(tmm5, *(const __m128i*)pi32_4);
	const __m128i tmm8 = _mm_slli_epi32(tmm6, 29);
	const __m128i tmm9 = _mm_slli_epi32(tmm7, 29);
	emm0 = _mm256_insertf128_si256(_mm256_castsi128_si256(tmm8), tmm9, 1);

#endif
	/* get the polynom selection mask 
	   there is one polynom for 0 <= x <= Pi/4
	   and another one for Pi/4<x<=Pi/2

	   Both branches will be computed.
	*/

#if __AVX2__ != 0
	emm2 = _mm256_and_si256(emm2, *(v8si*)pi32_2);
	emm2 = _mm256_cmpeq_epi32(emm2, _mm256_setzero_si256());

#else
	const __m128i tmm10 = _mm_and_si128(tmm4, *(const __m128i*)pi32_2);
	const __m128i tmm11 = _mm_and_si128(tmm5, *(const __m128i*)pi32_2);
	const __m128i tmm12 = _mm_cmpeq_epi32(tmm10, _mm_setzero_si128());
	const __m128i tmm13 = _mm_cmpeq_epi32(tmm11, _mm_setzero_si128());
	emm2 = _mm256_insertf128_si256(_mm256_castsi128_si256(tmm12), tmm13, 1);

#endif
	v8sf swap_sign_bit = _mm256_castsi256_ps(emm0);
	v8sf poly_mask = _mm256_castsi256_ps(emm2);
	sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v8sf*)pf32_minus_cephes_DP1;
	xmm2 = *(v8sf*)pf32_minus_cephes_DP2;
	xmm3 = *(v8sf*)pf32_minus_cephes_DP3;
	xmm1 = _mm256_mul_ps(y, xmm1);
	xmm2 = _mm256_mul_ps(y, xmm2);
	xmm3 = _mm256_mul_ps(y, xmm3);
	x = _mm256_add_ps(x, xmm1);
	x = _mm256_add_ps(x, xmm2);
	x = _mm256_add_ps(x, xmm3);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	y = *(v8sf*)pf32_coscof_p0;
	v8sf z = _mm256_mul_ps(x,x);

	y = _mm256_mul_ps(y, z);
	y = _mm256_add_ps(y, *(v8sf*)pf32_coscof_p1);
	y = _mm256_mul_ps(y, z);
	y = _mm256_add_ps(y, *(v8sf*)pf32_coscof_p2);
	y = _mm256_mul_ps(y, z);
	y = _mm256_mul_ps(y, z);
	v8sf tmp = _mm256_mul_ps(z, *(v8sf*)pf32_0p5);
	y = _mm256_sub_ps(y, tmp);
	y = _mm256_add_ps(y, *(v8sf*)pf32_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */
	v8sf y2 = *(v8sf*)pf32_sincof_p0;
	y2 = _mm256_mul_ps(y2, z);
	y2 = _mm256_add_ps(y2, *(v8sf*)pf32_sincof_p1);
	y2 = _mm256_mul_ps(y2, z);
	y2 = _mm256_add_ps(y2, *(v8sf*)pf32_sincof_p2);
	y2 = _mm256_mul_ps(y2, z);
	y2 = _mm256_mul_ps(y2, x);
	y2 = _mm256_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	y = _mm256_blendv_ps(y, y2, poly_mask);

	/* update the sign */
	y = _mm256_xor_ps(y, sign_bit);
	return y;
}

/* almost the same as cephes_sin */
inline v8sf cephes_cos(v8sf x) {
	v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, y;
	v8si emm0, emm2;

	/* take the absolute value */
	x = _mm256_and_ps(x, *(v8sf*)pf32_inv_sign_mask);

	/* scale by 4/Pi */
	y = _mm256_mul_ps(x, *(v8sf*)pf32_cephes_FOPI);

	/* store the integer part of y in mm0 */
	emm2 = _mm256_cvttps_epi32(y);

	/* j=(j+1) & (~1) (see the cephes sources) */
#if __AVX2__ != 0
	emm2 = _mm256_add_epi32(emm2, *(v8si*)pi32_1);
	emm2 = _mm256_and_si256(emm2, *(v8si*)pi32_inv1);

#else
	const __m128i tmm0 = _mm256_castsi256_si128(emm2);
	const __m128i tmm1 = _mm256_extractf128_si256(emm2, 1);
	const __m128i tmm2 = _mm_add_epi32(tmm0, *(const __m128i*)pi32_1);
	const __m128i tmm3 = _mm_add_epi32(tmm1, *(const __m128i*)pi32_1);
	const __m128i tmm4 = _mm_and_si128(tmm2, *(const __m128i*)pi32_inv1);
	const __m128i tmm5 = _mm_and_si128(tmm3, *(const __m128i*)pi32_inv1);
	emm2 = _mm256_insertf128_si256(_mm256_castsi128_si256(tmm4), tmm5, 1);

#endif
	y = _mm256_cvtepi32_ps(emm2);

#if __AVX2__ != 0
	emm2 = _mm256_sub_epi32(emm2, *(v8si*)pi32_2);

	/* get the swap sign flag */
	emm0 = _mm256_andnot_si256(emm2, *(v8si*)pi32_4);
	emm0 = _mm256_slli_epi32(emm0, 29);

	/* get the polynom selection mask */
	emm2 = _mm256_and_si256(emm2, *(v8si*)pi32_2);
	emm2 = _mm256_cmpeq_epi32(emm2, _mm256_setzero_si256());

#else
	const __m128i tmm6 = _mm_sub_epi32(tmm4, *(const __m128i*)pi32_2);
	const __m128i tmm7 = _mm_sub_epi32(tmm5, *(const __m128i*)pi32_2);

	const __m128i tmm8 = _mm_andnot_si128(tmm6, *(const __m128i*)pi32_4);
	const __m128i tmm9 = _mm_andnot_si128(tmm7, *(const __m128i*)pi32_4);
	const __m128i tmm10 = _mm_slli_epi32(tmm8, 29);
	const __m128i tmm11 = _mm_slli_epi32(tmm9, 29);
	emm0 = _mm256_insertf128_si256(_mm256_castsi128_si256(tmm10), tmm11, 1);

	const __m128i tmm12 = _mm_and_si128(tmm6, *(const __m128i*)pi32_2);
	const __m128i tmm13 = _mm_and_si128(tmm7, *(const __m128i*)pi32_2);
	const __m128i tmm14 = _mm_cmpeq_epi32(tmm12, _mm_setzero_si128());
	const __m128i tmm15 = _mm_cmpeq_epi32(tmm13, _mm_setzero_si128());
	emm2 = _mm256_insertf128_si256(_mm256_castsi128_si256(tmm14), tmm15, 1);

#endif
	v8sf sign_bit = _mm256_castsi256_ps(emm0);
	v8sf poly_mask = _mm256_castsi256_ps(emm2);

	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v8sf*)pf32_minus_cephes_DP1;
	xmm2 = *(v8sf*)pf32_minus_cephes_DP2;
	xmm3 = *(v8sf*)pf32_minus_cephes_DP3;
	xmm1 = _mm256_mul_ps(y, xmm1);
	xmm2 = _mm256_mul_ps(y, xmm2);
	xmm3 = _mm256_mul_ps(y, xmm3);
	x = _mm256_add_ps(x, xmm1);
	x = _mm256_add_ps(x, xmm2);
	x = _mm256_add_ps(x, xmm3);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	y = *(v8sf*)pf32_coscof_p0;
	v8sf z = _mm256_mul_ps(x,x);

	y = _mm256_mul_ps(y, z);
	y = _mm256_add_ps(y, *(v8sf*)pf32_coscof_p1);
	y = _mm256_mul_ps(y, z);
	y = _mm256_add_ps(y, *(v8sf*)pf32_coscof_p2);
	y = _mm256_mul_ps(y, z);
	y = _mm256_mul_ps(y, z);
	v8sf tmp = _mm256_mul_ps(z, *(v8sf*)pf32_0p5);
	y = _mm256_sub_ps(y, tmp);
	y = _mm256_add_ps(y, *(v8sf*)pf32_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */
	v8sf y2 = *(v8sf*)pf32_sincof_p0;
	y2 = _mm256_mul_ps(y2, z);
	y2 = _mm256_add_ps(y2, *(v8sf*)pf32_sincof_p1);
	y2 = _mm256_mul_ps(y2, z);
	y2 = _mm256_add_ps(y2, *(v8sf*)pf32_sincof_p2);
	y2 = _mm256_mul_ps(y2, z);
	y2 = _mm256_mul_ps(y2, x);
	y2 = _mm256_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	y = _mm256_blendv_ps(y, y2, poly_mask);

	/* update the sign */
	y = _mm256_xor_ps(y, sign_bit);
	return y;
}

/* since cephes_sin and cephes_cos are almost identical, cephes_sincos could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
inline void cephes_sincos(v8sf x, v8sf *s, v8sf *c) {
	v8sf xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
	v8si emm0, emm2, emm4;
	sign_bit_sin = x;

	/* take the absolute value */
	x = _mm256_and_ps(x, *(v8sf*)pf32_inv_sign_mask);

	/* extract the sign bit (upper one) */
	sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(v8sf*)pf32_sign_mask);

	/* scale by 4/Pi */
	y = _mm256_mul_ps(x, *(v8sf*)pf32_cephes_FOPI);
	  
	/* store the integer part of y in emm2 */
	emm2 = _mm256_cvttps_epi32(y);

	/* j=(j+1) & (~1) (see the cephes sources) */
#if __AVX2__ != 0
	emm2 = _mm256_add_epi32(emm2, *(v8si*)pi32_1);
	emm2 = _mm256_and_si256(emm2, *(v8si*)pi32_inv1);

#else
	const __m128i tmm0 = _mm256_castsi256_si128(emm2);
	const __m128i tmm1 = _mm256_extractf128_si256(emm2, 1);
	const __m128i tmm2 = _mm_add_epi32(tmm0, *(const __m128i*)pi32_1);
	const __m128i tmm3 = _mm_add_epi32(tmm1, *(const __m128i*)pi32_1);
	const __m128i tmm4 = _mm_and_si128(tmm2, *(const __m128i*)pi32_inv1);
	const __m128i tmm5 = _mm_and_si128(tmm3, *(const __m128i*)pi32_inv1);
	emm2 = _mm256_insertf128_si256(_mm256_castsi128_si256(tmm4), tmm5, 1);

#endif
	y = _mm256_cvtepi32_ps(emm2);

	emm4 = emm2;

	/* get the swap sign flag for the sine */
#if __AVX2__ != 0
	emm0 = _mm256_and_si256(emm2, *(v8si*)pi32_4);
	emm0 = _mm256_slli_epi32(emm0, 29);

#else
	const __m128i tmm6 = _mm_and_si128(tmm4, *(const __m128i*)pi32_4);
	const __m128i tmm7 = _mm_and_si128(tmm5, *(const __m128i*)pi32_4);
	const __m128i tmm8 = _mm_slli_epi32(tmm6, 29);
	const __m128i tmm9 = _mm_slli_epi32(tmm7, 29);
	emm0 = _mm256_insertf128_si256(_mm256_castsi128_si256(tmm8), tmm9, 1);

#endif
	v8sf swap_sign_bit_sin = _mm256_castsi256_ps(emm0);

	/* get the polynom selection mask for the sine*/
#if __AVX2__ != 0
	emm2 = _mm256_and_si256(emm2, *(v8si*)pi32_2);
	emm2 = _mm256_cmpeq_epi32(emm2, _mm256_setzero_si256());

#else
	const __m128i tmm10 = _mm_and_si128(tmm4, *(const __m128i*)pi32_2);
	const __m128i tmm11 = _mm_and_si128(tmm5, *(const __m128i*)pi32_2);
	const __m128i tmm12 = _mm_cmpeq_epi32(tmm10, _mm_setzero_si128());
	const __m128i tmm13 = _mm_cmpeq_epi32(tmm11, _mm_setzero_si128());
	emm2 = _mm256_insertf128_si256(_mm256_castsi128_si256(tmm12), tmm13, 1);

#endif
	v8sf poly_mask = _mm256_castsi256_ps(emm2);

	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v8sf*)pf32_minus_cephes_DP1;
	xmm2 = *(v8sf*)pf32_minus_cephes_DP2;
	xmm3 = *(v8sf*)pf32_minus_cephes_DP3;
	xmm1 = _mm256_mul_ps(y, xmm1);
	xmm2 = _mm256_mul_ps(y, xmm2);
	xmm3 = _mm256_mul_ps(y, xmm3);
	x = _mm256_add_ps(x, xmm1);
	x = _mm256_add_ps(x, xmm2);
	x = _mm256_add_ps(x, xmm3);

#if __AVX2__ != 0
	emm4 = _mm256_sub_epi32(emm4, *(v8si*)pi32_2);
	emm4 = _mm256_andnot_si256(emm4, *(v8si*)pi32_4);
	emm4 = _mm256_slli_epi32(emm4, 29);

#else
	const __m128i tmm14 = _mm_sub_epi32(tmm4, *(const __m128i*)pi32_2);
	const __m128i tmm15 = _mm_sub_epi32(tmm5, *(const __m128i*)pi32_2);
	const __m128i tmm16 = _mm_andnot_si128(tmm14, *(const __m128i*)pi32_4);
	const __m128i tmm17 = _mm_andnot_si128(tmm15, *(const __m128i*)pi32_4);
	const __m128i tmm18 = _mm_slli_epi32(tmm16, 29);
	const __m128i tmm19 = _mm_slli_epi32(tmm17, 29);
	emm4 = _mm256_insertf128_si256(_mm256_castsi128_si256(tmm18), tmm19, 1);

#endif
	v8sf sign_bit_cos = _mm256_castsi256_ps(emm4);

	sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	v8sf z = _mm256_mul_ps(x,x);
	y = *(v8sf*)pf32_coscof_p0;

	y = _mm256_mul_ps(y, z);
	y = _mm256_add_ps(y, *(v8sf*)pf32_coscof_p1);
	y = _mm256_mul_ps(y, z);
	y = _mm256_add_ps(y, *(v8sf*)pf32_coscof_p2);
	y = _mm256_mul_ps(y, z);
	y = _mm256_mul_ps(y, z);
	v8sf tmp = _mm256_mul_ps(z, *(v8sf*)pf32_0p5);
	y = _mm256_sub_ps(y, tmp);
	y = _mm256_add_ps(y, *(v8sf*)pf32_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */
	v8sf y2 = *(v8sf*)pf32_sincof_p0;
	y2 = _mm256_mul_ps(y2, z);
	y2 = _mm256_add_ps(y2, *(v8sf*)pf32_sincof_p1);
	y2 = _mm256_mul_ps(y2, z);
	y2 = _mm256_add_ps(y2, *(v8sf*)pf32_sincof_p2);
	y2 = _mm256_mul_ps(y2, z);
	y2 = _mm256_mul_ps(y2, x);
	y2 = _mm256_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	xmm1 = _mm256_blendv_ps(y, y2, poly_mask);
	xmm2 = _mm256_blendv_ps(y2, y, poly_mask);

	/* update the sign */
	*s = _mm256_xor_ps(xmm1, sign_bit_sin);
	*c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

#undef _PS_CONST
#undef _PI32_CONST
#undef _PS_CONST_INT

#endif // cephes_math_avx_H__
