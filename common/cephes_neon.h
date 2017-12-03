////////////////////////////////////////////////////////////////////////////////
// ARM ASIMD (NEON) implementation of sin, cos, exp and log
//
// A direct translation of Julien Pommier's cephes/SSE2 to ASIMD
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

#ifndef cephes_math_asimd_H__
#define cephes_math_asimd_H__

#include <arm_neon.h>

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

#define _PS_CONST(Name, Val) \
	static const float pf32_##Name[4] __attribute__ ((aligned(16))) = { Val, Val, Val, Val }
#define _PI32_CONST(Name, Val) \
	static const int pi32_##Name[4] __attribute__ ((aligned(16))) = { (int)(Val), (int)(Val), (int)(Val), (int)(Val) }
#define _PS_CONST_INT(Name, Val) \
	static const int pf32_##Name[4] __attribute__ ((aligned(16))) = { (int)(Val), (int)(Val), (int)(Val), (int)(Val) }

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

typedef float32x4_t v4sf;
typedef int32x4_t v4si;

/* natural logarithm computed for 4 simultaneous float 
   return NaN for x <= 0
*/
inline v4sf cephes_log(v4sf x) {
	v4si emm0;
	v4sf one = *(v4sf*)pf32_1;

	v4si invalid_mask = vreinterpretq_s32_u32(vcleq_f32(x, vdupq_n_f32(0)));

	/* cut off denormalized stuff */
	x = vmaxq_f32(x, *(v4sf*)pf32_min_norm_pos);

	emm0 = vshrq_n_s32(vreinterpretq_s32_f32(x), 23);

	/* keep only the fractional part */
	x = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(x), *(v4si*)pf32_inv_mant_mask));
	x = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(x), *(v4si*)pf32_0p5));

	emm0 = vsubq_s32(emm0, *(v4si*)pi32_0x7f);
	v4sf e = vcvtq_f32_s32(emm0);

	e = vaddq_f32(e, one);

	/* part2: 
	   if( x < SQRTHF ) {
		 e -= 1;
		 x = x + x - 1.0;
	   } else { x = x - 1.0; }
	*/
	v4si mask = vreinterpretq_s32_u32(vcltq_f32(x, *(v4sf*)pf32_cephes_SQRTHF));
	v4sf tmp = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(x), mask));
	x = vsubq_f32(x, one);
	e = vsubq_f32(e, vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(one), mask)));
	x = vaddq_f32(x, tmp);

	v4sf z = vmulq_f32(x, x);

	v4sf y = *(v4sf*)pf32_cephes_log_p0;
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_log_p1);
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_log_p2);
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_log_p3);
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_log_p4);
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_log_p5);
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_log_p6);
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_log_p7);
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_log_p8);
	y = vmulq_f32(y, x);

	y = vmulq_f32(y, z);

	tmp = vmulq_f32(e, *(v4sf*)pf32_cephes_log_q1);
	y = vaddq_f32(y, tmp);

	tmp = vmulq_f32(z, *(v4sf*)pf32_0p5);
	y = vsubq_f32(y, tmp);

	tmp = vmulq_f32(e, *(v4sf*)pf32_cephes_log_q2);
	x = vaddq_f32(x, y);
	x = vaddq_f32(x, tmp);
	x = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(x), invalid_mask)); // negative arg will be NAN
	return x;
}

inline v4sf cephes_exp(v4sf x) {
	v4sf tmp = vdupq_n_f32(0), fx;
	v4si emm0;
	v4sf one = *(v4sf*)pf32_1;

	x = vminq_f32(x, *(v4sf*)pf32_exp_hi);
	x = vmaxq_f32(x, *(v4sf*)pf32_exp_lo);

	/* express exp(x) as exp(g + n*log(2)) */
	fx = vmulq_f32(x, *(v4sf*)pf32_cephes_LOG2EF);
	fx = vaddq_f32(fx, *(v4sf*)pf32_0p5);

	/* how to perform a floorf with NEON: just below */
	tmp = vrndq_f32(fx);

	/* if greater, substract 1 */
	v4si mask = vreinterpretq_s32_u32(vcgtq_f32(tmp, fx));
	mask = vandq_s32(mask, vreinterpretq_s32_f32(one));
	fx = vsubq_f32(tmp, vreinterpretq_f32_s32(mask));

	tmp = vmulq_f32(fx, *(v4sf*)pf32_cephes_exp_C1);
	v4sf z = vmulq_f32(fx, *(v4sf*)pf32_cephes_exp_C2);
	x = vsubq_f32(x, tmp);
	x = vsubq_f32(x, z);

	z = vmulq_f32(x,x);

	v4sf y = *(v4sf*)pf32_cephes_exp_p0;
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_exp_p1);
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_exp_p2);
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_exp_p3);
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_exp_p4);
	y = vmulq_f32(y, x);
	y = vaddq_f32(y, *(v4sf*)pf32_cephes_exp_p5);
	y = vmulq_f32(y, z);
	y = vaddq_f32(y, x);
	y = vaddq_f32(y, one);

	/* build 2^n */
	emm0 = vcvtq_s32_f32(fx);
	emm0 = vaddq_s32(emm0, *(v4si*)pi32_0x7f);
	emm0 = vshlq_n_s32(emm0, 23);
	v4sf pow2n = vreinterpretq_f32_s32(emm0);

	y = vmulq_f32(y, pow2n);
	return y;
}

////////////////////////////////////////////////////////////////////////////////
// mk: raise four non-negative bases to the four exponents
//
////////////////////////////////////////////////////////////////////////////////

inline v4sf cephes_pow(const v4sf x, const v4sf y) {
	const v4si zero_filter = vreinterpretq_s32_u32(vceqq_f32(x, vdupq_n_f32(0)));
	return vreinterpretq_f32_s32(vbicq_s32(vreinterpretq_s32_f32(cephes_exp(vmulq_f32(cephes_log(x), y))), zero_filter));
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

inline v4sf cephes_sin(v4sf x) {
	v4sf xmm1, xmm2 = vdupq_n_f32(0), xmm3, y;
	v4si emm0, emm2, sign_bit;
	sign_bit = vreinterpretq_s32_f32(x);

	/* take the absolute value */
	x = vabsq_f32(x);

	/* extract the sign bit (upper one) */
	sign_bit = vandq_s32(sign_bit, *(v4si*)pf32_sign_mask);

	/* scale by 4/Pi */
	y = vmulq_f32(x, *(v4sf*)pf32_cephes_FOPI);

	/* store the integer part of y in mm0 */
	emm2 = vcvtq_s32_f32(y);

	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = vaddq_s32(emm2, *(v4si*)pi32_1);
	emm2 = vandq_s32(emm2, *(v4si*)pi32_inv1);
	y = vcvtq_f32_s32(emm2);

	/* get the swap sign flag */
	emm0 = vandq_s32(emm2, *(v4si*)pi32_4);
	emm0 = vshlq_n_s32(emm0, 29);
	/* get the polynom selection mask 
	   there is one polynom for 0 <= x <= Pi/4
	   and another one for Pi/4<x<=Pi/2

	   Both branches will be computed.
	*/
	emm2 = vandq_s32(emm2, *(v4si*)pi32_2);
	emm2 = vreinterpretq_s32_u32(vceqzq_s32(emm2));

	v4si swap_sign_bit = emm0;
	v4si poly_mask = emm2;
	sign_bit = veorq_s32(sign_bit, swap_sign_bit);

	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v4sf*)pf32_minus_cephes_DP1;
	xmm2 = *(v4sf*)pf32_minus_cephes_DP2;
	xmm3 = *(v4sf*)pf32_minus_cephes_DP3;
	xmm1 = vmulq_f32(y, xmm1);
	xmm2 = vmulq_f32(y, xmm2);
	xmm3 = vmulq_f32(y, xmm3);
	x = vaddq_f32(x, xmm1);
	x = vaddq_f32(x, xmm2);
	x = vaddq_f32(x, xmm3);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	y = *(v4sf*)pf32_coscof_p0;
	v4sf z = vmulq_f32(x,x);

	y = vmulq_f32(y, z);
	y = vaddq_f32(y, *(v4sf*)pf32_coscof_p1);
	y = vmulq_f32(y, z);
	y = vaddq_f32(y, *(v4sf*)pf32_coscof_p2);
	y = vmulq_f32(y, z);
	y = vmulq_f32(y, z);
	v4sf tmp = vmulq_f32(z, *(v4sf*)pf32_0p5);
	y = vsubq_f32(y, tmp);
	y = vaddq_f32(y, *(v4sf*)pf32_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */
	v4sf y2 = *(v4sf*)pf32_sincof_p0;
	y2 = vmulq_f32(y2, z);
	y2 = vaddq_f32(y2, *(v4sf*)pf32_sincof_p1);
	y2 = vmulq_f32(y2, z);
	y2 = vaddq_f32(y2, *(v4sf*)pf32_sincof_p2);
	y2 = vmulq_f32(y2, z);
	y2 = vmulq_f32(y2, x);
	y2 = vaddq_f32(y2, x);

	/* select the correct result from the two polynoms */
	y = vbslq_f32(vreinterpretq_u32_s32(poly_mask), y2, y);

	/* update the sign */
	y = vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(y), sign_bit));
	return y;
}

/* almost the same as cephes_sin */
inline v4sf cephes_cos(v4sf x) {
	v4sf xmm1, xmm2 = vdupq_n_f32(0), xmm3, y;
	v4si emm0, emm2;

	/* take the absolute value */
	x = vabsq_f32(x);

	/* scale by 4/Pi */
	y = vmulq_f32(x, *(v4sf*)pf32_cephes_FOPI);

	/* store the integer part of y in mm0 */
	emm2 = vcvtq_s32_f32(y);

	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = vaddq_s32(emm2, *(v4si*)pi32_1);
	emm2 = vandq_s32(emm2, *(v4si*)pi32_inv1);
	y = vcvtq_f32_s32(emm2);

	emm2 = vsubq_s32(emm2, *(v4si*)pi32_2);

	/* get the swap sign flag */
	emm0 = vbicq_s32(*(v4si*)pi32_4, emm2);
	emm0 = vshlq_n_s32(emm0, 29);

	/* get the polynom selection mask */
	emm2 = vandq_s32(emm2, *(v4si*)pi32_2);
	emm2 = vreinterpretq_s32_u32(vceqzq_s32(emm2));

	v4si sign_bit = emm0;
	v4si poly_mask = emm2;

	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v4sf*)pf32_minus_cephes_DP1;
	xmm2 = *(v4sf*)pf32_minus_cephes_DP2;
	xmm3 = *(v4sf*)pf32_minus_cephes_DP3;
	xmm1 = vmulq_f32(y, xmm1);
	xmm2 = vmulq_f32(y, xmm2);
	xmm3 = vmulq_f32(y, xmm3);
	x = vaddq_f32(x, xmm1);
	x = vaddq_f32(x, xmm2);
	x = vaddq_f32(x, xmm3);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	y = *(v4sf*)pf32_coscof_p0;
	v4sf z = vmulq_f32(x,x);

	y = vmulq_f32(y, z);
	y = vaddq_f32(y, *(v4sf*)pf32_coscof_p1);
	y = vmulq_f32(y, z);
	y = vaddq_f32(y, *(v4sf*)pf32_coscof_p2);
	y = vmulq_f32(y, z);
	y = vmulq_f32(y, z);
	v4sf tmp = vmulq_f32(z, *(v4sf*)pf32_0p5);
	y = vsubq_f32(y, tmp);
	y = vaddq_f32(y, *(v4sf*)pf32_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */
	v4sf y2 = *(v4sf*)pf32_sincof_p0;
	y2 = vmulq_f32(y2, z);
	y2 = vaddq_f32(y2, *(v4sf*)pf32_sincof_p1);
	y2 = vmulq_f32(y2, z);
	y2 = vaddq_f32(y2, *(v4sf*)pf32_sincof_p2);
	y2 = vmulq_f32(y2, z);
	y2 = vmulq_f32(y2, x);
	y2 = vaddq_f32(y2, x);

	/* select the correct result from the two polynoms */
	y = vbslq_f32(vreinterpretq_u32_s32(poly_mask), y2, y);

	/* update the sign */
	y = vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(y), sign_bit));
	return y;
}

/* since cephes_sin and cephes_cos are almost identical, cephes_sincos could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
inline void cephes_sincos(v4sf x, v4sf *s, v4sf *c) {
	v4sf xmm1, xmm2, xmm3 = vdupq_n_f32(0), y;
	v4si emm0, emm2, emm4, sign_bit_sin;
	sign_bit_sin = vreinterpretq_s32_f32(x);

	/* take the absolute value */
	x = vabsq_f32(x);

	/* extract the sign bit (upper one) */
	sign_bit_sin = vandq_s32(sign_bit_sin, *(v4si*)pf32_sign_mask);

	/* scale by 4/Pi */
	y = vmulq_f32(x, *(v4sf*)pf32_cephes_FOPI);

	/* store the integer part of y in emm2 */
	emm2 = vcvtq_s32_f32(y);

	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = vaddq_s32(emm2, *(v4si*)pi32_1);
	emm2 = vandq_s32(emm2, *(v4si*)pi32_inv1);
	y = vcvtq_f32_s32(emm2);

	emm4 = emm2;

	/* get the swap sign flag for the sine */
	emm0 = vandq_s32(emm2, *(v4si*)pi32_4);
	emm0 = vshlq_n_s32(emm0, 29);
	v4si swap_sign_bit_sin = emm0;

	/* get the polynom selection mask for the sine*/
	emm2 = vandq_s32(emm2, *(v4si*)pi32_2);
	emm2 = vreinterpretq_s32_u32(vceqzq_s32(emm2));
	v4si poly_mask = emm2;

	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v4sf*)pf32_minus_cephes_DP1;
	xmm2 = *(v4sf*)pf32_minus_cephes_DP2;
	xmm3 = *(v4sf*)pf32_minus_cephes_DP3;
	xmm1 = vmulq_f32(y, xmm1);
	xmm2 = vmulq_f32(y, xmm2);
	xmm3 = vmulq_f32(y, xmm3);
	x = vaddq_f32(x, xmm1);
	x = vaddq_f32(x, xmm2);
	x = vaddq_f32(x, xmm3);

	emm4 = vsubq_s32(emm4, *(v4si*)pi32_2);
	emm4 = vbicq_s32(*(v4si*)pi32_4, emm4);
	emm4 = vshlq_n_s32(emm4, 29);
	v4si sign_bit_cos = emm4;

	sign_bit_sin = veorq_s32(sign_bit_sin, swap_sign_bit_sin);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	v4sf z = vmulq_f32(x,x);
	y = *(v4sf*)pf32_coscof_p0;

	y = vmulq_f32(y, z);
	y = vaddq_f32(y, *(v4sf*)pf32_coscof_p1);
	y = vmulq_f32(y, z);
	y = vaddq_f32(y, *(v4sf*)pf32_coscof_p2);
	y = vmulq_f32(y, z);
	y = vmulq_f32(y, z);
	v4sf tmp = vmulq_f32(z, *(v4sf*)pf32_0p5);
	y = vsubq_f32(y, tmp);
	y = vaddq_f32(y, *(v4sf*)pf32_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */
	v4sf y2 = *(v4sf*)pf32_sincof_p0;
	y2 = vmulq_f32(y2, z);
	y2 = vaddq_f32(y2, *(v4sf*)pf32_sincof_p1);
	y2 = vmulq_f32(y2, z);
	y2 = vaddq_f32(y2, *(v4sf*)pf32_sincof_p2);
	y2 = vmulq_f32(y2, z);
	y2 = vmulq_f32(y2, x);
	y2 = vaddq_f32(y2, x);

	/* select the correct result from the two polynoms */
	v4sf ysin2 = vreinterpretq_f32_s32(vandq_s32(poly_mask, vreinterpretq_s32_f32(y2)));
	v4sf ysin1 = vreinterpretq_f32_s32(vbicq_s32(vreinterpretq_s32_f32(y), poly_mask));
	y2 = vsubq_f32(y2, ysin2);
	y = vsubq_f32(y, ysin1);

	xmm1 = vaddq_f32(ysin1, ysin2);
	xmm2 = vaddq_f32(y, y2);

	/* update the sign */
	*s = vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(xmm1), sign_bit_sin));
	*c = vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(xmm2), sign_bit_cos));
}

#undef _PS_CONST
#undef _PI32_CONST
#undef _PS_CONST_INT

#endif // cephes_math_asimd_H__
