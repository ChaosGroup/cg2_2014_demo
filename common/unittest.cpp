#define _USE_MATH_DEFINES
#include <cmath>
#include <cassert>
#include <limits>
#include <iomanip>
#include "stream.hpp"
#include "vectnative.hpp"

#ifdef _WIN32
#include <SDKDDKVer.h>
#include <stdio.h>
#else
// verify iostream-free status (see further down)
#if _GLIBCXX_IOSTREAM != 0
#error rogue iostream acquired
#endif
#endif

// use a custom minimalistic implementation of std::istream/ostream to avoid the highly-harmful-for-the-optimisations
// static initialization of std::cin/cout/cerr; our mini-implementation is initialized at the start of main()
namespace stream {
in cin;
out cout;
out cerr;
}

#if _MSC_VER != 0
using namespace simd; // oldest msvc we target does not support using-declarations, so we go full monty there
#else
#if NATIVE_F64X4 != 0
using simd::f64x4;
using simd::as_f64x4;
using simd::get_f64x2;
#endif
#if NATIVE_S64X4 != 0
using simd::s64x4;
using simd::as_s64x4;
using simd::get_s64x2;
#endif
#if NATIVE_U64X4 != 0
using simd::u64x4;
using simd::as_u64x4;
using simd::get_u64x2;
#endif
#if NATIVE_F32X8 != 0
using simd::f32x8;
using simd::as_f32x8;
using simd::get_f32x4;
#endif
#if NATIVE_S32X8 != 0
using simd::s32x8;
using simd::as_s32x8;
using simd::get_s32x4;
#endif
#if NATIVE_U32X8 != 0
using simd::u32x8;
using simd::as_u32x8;
using simd::get_u32x4;
#endif
#if NATIVE_F32X2 != 0
using simd::f32x2;
using simd::as_f32x2;
#endif
#if NATIVE_S32X2 != 0
using simd::s32x2;
using simd::as_s32x2;
#endif
#if NATIVE_U32X2 != 0
using simd::u32x2;
using simd::as_u32x2;
#endif
#if NATIVE_S16X16 != 0
using simd::s16x16;
using simd::as_s16x16;
using simd::get_s16x8;
#endif
#if NATIVE_U16X16 != 0
using simd::u16x16;
using simd::as_u16x16;
using simd::get_u16x8;
#endif
#if NATIVE_S16X4 != 0
using simd::s16x4;
using simd::as_s16x4;
#endif
#if NATIVE_U16X4 != 0
using simd::u16x4;
using simd::as_u16x4;
#endif
using simd::f64x2;
using simd::s64x2;
using simd::u64x2;
using simd::f32x4;
using simd::s32x4;
using simd::u32x4;
using simd::s16x8;
using simd::u16x8;
using simd::as_f64x2;
using simd::as_s64x2;
using simd::as_u64x2;
using simd::as_f32x4;
using simd::as_s32x4;
using simd::as_u32x4;
using simd::as_s16x8;
using simd::as_u16x8;
using simd::all;
using simd::any;
using simd::none;
using simd::min;
using simd::max;
using simd::abs;
using simd::mask;
using simd::select;
using simd::sqrt;
using simd::log;
using simd::exp;
using simd::pow;
using simd::sin;
using simd::cos;
using simd::sincos;
using simd::transpose4x4;
using simd::operator ==;
using simd::operator !=;
using simd::operator <;
using simd::operator >;
using simd::operator <=;
using simd::operator >=;
using simd::flag_zero;
using simd::flag_native;
#endif

template < typename T >
const T& min(const T& a, const T& b) {
	return (a < b) ? a : b;
}

template < typename T >
const T& max(const T& a, const T& b) {
	return (a > b) ? a : b;
}

class formatter_f64 {
	double f;

public:
	formatter_f64(const double a) : f(a) {}

	uint64_t get() const {
		return reinterpret_cast< const uint64_t& >(f);
	}
};

class formatter_f32 {
	float f;

public:
	formatter_f32(const float a) : f(a) {}

	uint32_t get() const {
		return reinterpret_cast< const uint32_t& >(f);
	}
};

static stream::out& operator << (
	stream::out& str,
	const formatter_f64 a) {

	str << 
		(a.get() >> 63) << ':' <<
		stream::setw(3)  << stream::setfill('0') << (a.get() >> 52 & 0x7ff) << ':' <<
		stream::setw(13) << stream::setfill('0') << (a.get() >>  0 & 0xfffffffffffff);
	return str;
}

static stream::out& operator << (
	stream::out& str,
	const formatter_f32 a) {

	str << 
		(a.get() >> 31) << ':' <<
		stream::setw(2) << stream::setfill('0') << (a.get() >> 23 & 0xff) << ':' <<
		stream::setw(6) << stream::setfill('0') << (a.get() >>  0 & 0x7fffff);
	return str;
}

#if NATIVE_F64X4 != 0
static stream::out& operator << (
	stream::out& str,
	const f64x4& a) {

	return str <<
		formatter_f64(a[0]) << ' ' << formatter_f64(a[1]) << ' ' << formatter_f64(a[2]) << ' ' << formatter_f64(a[3]);
}

#endif
#if NATIVE_S64X4 != 0
static stream::out& operator << (
	stream::out& str,
	const s64x4& a) {

	return str <<
		a[0] << ' ' << a[1] << ' ' << a[2] << ' ' << a[3];
}

#endif
#if NATIVE_U64X4 != 0
static stream::out& operator << (
	stream::out& str,
	const u64x4& a) {

	return str <<
		a[0] << ' ' << a[1] << ' ' << a[2] << ' ' << a[3];
}

#endif
static stream::out& operator << (
	stream::out& str,
	const f64x2& a) {

	return str <<
		formatter_f64(a[0]) << ' ' << formatter_f64(a[1]);
}

static stream::out& operator << (
	stream::out& str,
	const s64x2& a) {

	return str <<
		a[0] << ' ' << a[1];
}

static stream::out& operator << (
	stream::out& str,
	const u64x2& a) {

	return str <<
		a[0] << ' ' << a[1];
}

#if NATIVE_F32X8 != 0
static stream::out& operator << (
	stream::out& str,
	const f32x8& a) {

	return str <<
		formatter_f32(a[0]) << ' ' << formatter_f32(a[1]) << ' ' << formatter_f32(a[2]) << ' ' << formatter_f32(a[3]) << ' ' <<
		formatter_f32(a[4]) << ' ' << formatter_f32(a[5]) << ' ' << formatter_f32(a[6]) << ' ' << formatter_f32(a[7]);
}

#endif
#if NATIVE_S32X8 != 0
static stream::out& operator << (
	stream::out& str,
	const s32x8& a) {

	return str <<
		a[0] << ' ' << a[1] << ' ' << a[2] << ' ' << a[3] << ' ' <<
		a[4] << ' ' << a[5] << ' ' << a[6] << ' ' << a[7];
}

#endif
#if NATIVE_U32X8 != 0
static stream::out& operator << (
	stream::out& str,
	const u32x8& a) {

	return str <<
		a[0] << ' ' << a[1] << ' ' << a[2] << ' ' << a[3] << ' ' <<
		a[4] << ' ' << a[5] << ' ' << a[6] << ' ' << a[7];
}

#endif
static stream::out& operator << (
	stream::out& str,
	const f32x4& a) {

	return str <<
		formatter_f32(a[0]) << ' ' << formatter_f32(a[1]) << ' ' << formatter_f32(a[2]) << ' ' << formatter_f32(a[3]);
}

static stream::out& operator << (
	stream::out& str,
	const s32x4& a) {

	return str <<
		a[0] << ' ' << a[1] << ' ' << a[2] << ' ' << a[3];
}

static stream::out& operator << (
	stream::out& str,
	const u32x4& a) {

	return str <<
		a[0] << ' ' << a[1] << ' ' << a[2] << ' ' << a[3];
}

#if NATIVE_F32X2 != 0
static stream::out& operator << (
	stream::out& str,
	const f32x2& a) {

	return str <<
		formatter_f32(a[0]) << ' ' << formatter_f32(a[1]);
}

#endif
#if NATIVE_S32X2 != 0
static stream::out& operator << (
	stream::out& str,
	const s32x2& a) {

	return str <<
		a[0] << ' ' << a[1];
}

#endif
#if NATIVE_U32X2 != 0
static stream::out& operator << (
	stream::out& str,
	const u32x2& a) {

	return str <<
		a[0] << ' ' << a[1];
}

#endif
#if NATIVE_S16X16 != 0
static stream::out& operator << (
	stream::out& str,
	const s16x16& a) {

	return str <<
		a[ 0] << ' ' << a[ 1] << ' ' << a[ 2] << ' ' << a[ 3] << ' ' <<
		a[ 4] << ' ' << a[ 5] << ' ' << a[ 6] << ' ' << a[ 7] << ' ' <<
		a[ 8] << ' ' << a[ 9] << ' ' << a[10] << ' ' << a[11] << ' ' <<
		a[12] << ' ' << a[13] << ' ' << a[14] << ' ' << a[15];
}

#endif
#if NATIVE_U16X16 != 0
static stream::out& operator << (
	stream::out& str,
	const u16x16& a) {

	return str <<
		a[ 0] << ' ' << a[ 1] << ' ' << a[ 2] << ' ' << a[ 3] << ' ' <<
		a[ 4] << ' ' << a[ 5] << ' ' << a[ 6] << ' ' << a[ 7] << ' ' <<
		a[ 8] << ' ' << a[ 9] << ' ' << a[10] << ' ' << a[11] << ' ' <<
		a[12] << ' ' << a[13] << ' ' << a[14] << ' ' << a[15];
}

#endif
static stream::out& operator << (
	stream::out& str,
	const s16x8& a) {

	return str <<
		a[0] << ' ' << a[1] << ' ' << a[2] << ' ' << a[3] << ' ' <<
		a[4] << ' ' << a[5] << ' ' << a[6] << ' ' << a[7];
}

static stream::out& operator << (
	stream::out& str,
	const u16x8& a) {

	return str <<
		a[0] << ' ' << a[1] << ' ' << a[2] << ' ' << a[3] << ' ' <<
		a[4] << ' ' << a[5] << ' ' << a[6] << ' ' << a[7];
}

#if NATIVE_S16X4 != 0
static stream::out& operator << (
	stream::out& str,
	const s16x4& a) {

	return str <<
		a[0] << ' ' << a[1] << ' ' << a[2] << ' ' << a[3];
}

#endif
#if NATIVE_U16X4 != 0
static stream::out& operator << (
	stream::out& str,
	const u16x4& a) {

	return str <<
		a[0] << ' ' << a[1] << ' ' << a[2] << ' ' << a[3];
}

#endif
// compute the ULP difference between two finite f32's in FTZ mode, up to 24 orders of magnitude
// (larger difference is clamped to 24 orders)
int32_t diff_ulp_ftz(const float a, const float b) {
	const uint32_t ia = reinterpret_cast< const uint32_t& >(a);
	const uint32_t ib = reinterpret_cast< const uint32_t& >(b);
	const uint32_t sign_mask = 0x80000000;
	const uint32_t exp_mask = 0x7f800000;
	const uint32_t mant_mask = ~(sign_mask | exp_mask);

	// early out at identical args or different-sign zeros
	if (ia == ib || sign_mask == (ia | ib))
		return 0;

	// don't bother with a single zero or different-sign non-zero args
	if (0 == (ia & ~sign_mask) || 0 == (ib & ~sign_mask) || (ia & sign_mask) != (ib & sign_mask))
		return -1;

	const uint32_t maxar = max(ia, ib);
	const uint32_t minar = min(ia, ib);
	const uint32_t maxar_exp = maxar >> 23 & 0xff;
	const uint32_t minar_exp = minar >> 23 & 0xff;
	const uint32_t exp_diff = maxar_exp - minar_exp;

	// make sure our same-scale difference fits 64 bits
	if (40 < exp_diff)
		return 1 << 24;

	const uint64_t same_scale_maxar = uint64_t(0x800000 | maxar & mant_mask) << exp_diff;
	const uint64_t same_scale_minar = uint64_t(0x800000 | minar & mant_mask);
	const uint64_t diff = same_scale_maxar - same_scale_minar;

	// clamp differences to 24 orders of magnitude
	return uint32_t(min(diff, uint64_t(1 << 24)));
}

// compute the ULP difference between two finite f32's, up to 24 orders of magnitude
// (larger difference is clamped to 24 orders)
int32_t diff_ulp(const float a, const float b) {
	const uint32_t ia = reinterpret_cast< const uint32_t& >(a);
	const uint32_t ib = reinterpret_cast< const uint32_t& >(b);
	const uint32_t sign_mask = 0x80000000;
	const uint32_t exp_mask = 0x7f800000;
	const uint32_t mant_mask = ~(sign_mask | exp_mask);

	// early out at identical args or different-sign zeros
	if (ia == ib || sign_mask == (ia | ib))
		return 0;

	// don't bother with a single zero or different-sign non-zero args
	if (0 == (ia & ~sign_mask) || 0 == (ib & ~sign_mask) || (ia & sign_mask) != (ib & sign_mask))
		return -1;

	const uint32_t maxar = max(ia, ib);
	const uint32_t minar = min(ia, ib);
	const uint32_t maxar_exp = maxar >> 23 & 0xff;
	const uint32_t minar_exp = minar >> 23 & 0xff;
	const uint32_t maxar_msb = maxar_exp ? 0x800000 : 0;
	const uint32_t minar_msb = minar_exp ? 0x800000 : 0;
	const uint32_t exp_diff = (maxar_exp ? maxar_exp : 1) - (minar_exp ? minar_exp : 1);

	// make sure our same-scale difference fits 64 bits
	if (40 < exp_diff)
		return 1 << 24;

	const uint64_t same_scale_maxar = uint64_t(maxar_msb | maxar & mant_mask) << exp_diff;
	const uint64_t same_scale_minar = uint64_t(minar_msb | minar & mant_mask);
	const uint64_t diff = same_scale_maxar - same_scale_minar;

	// clamp differences to 24 orders of magnitude
	return uint32_t(min(diff, uint64_t(1 << 24)));
}

float compose_f32(const uint32_t sign, const uint32_t exp, const uint32_t mant) {
	const uint32_t ret = (sign << 31) | ((exp & 0xff) << 23) | (mant & 0x7fffff);
	return reinterpret_cast< const float& >(ret);
}

float round_bin(float x, const unsigned dropBits) {
	if (23 < dropBits)
		return compose_f32(0, 0xff, 0x400000);

	if (0 == dropBits)
		return x;

	uint32_t ix = reinterpret_cast< const uint32_t& >(x);
	const uint32_t sign_mask = 0x80000000;
	const uint32_t exp_mask = 0x7f800000;
	const uint32_t mant_mask = ~(sign_mask | exp_mask);

	const uint32_t exp = ix >> 23 & 0xff;
	const bool overflow = exp >= 0xff - dropBits;

	if (overflow) {
		ix -= dropBits << 23;
		x = reinterpret_cast< const float& >(ix);
	}

	const uint32_t ix_clean = ix & ~mant_mask;
	const float x_clean = reinterpret_cast< const float& >(ix_clean);
	const volatile float larger = x_clean * (1 << dropBits);
	const float r = (larger + x) - larger;

	if (overflow)
		return r * (1 << dropBits);

	return r;
}

float round_bin_unsafe(const float x, const unsigned dropBits) {
	if (23 < dropBits)
		return compose_f32(0, 0xff, 0x400000);

	if (0 == dropBits)
		return x;

	const uint32_t ix = reinterpret_cast< const uint32_t& >(x);
	const uint32_t sign_mask = 0x80000000;
	const uint32_t exp_mask = 0x7f800000;
	const uint32_t mant_mask = ~(sign_mask | exp_mask);

	const uint32_t ix_clean = ix & ~mant_mask;
	const float x_clean = reinterpret_cast< const float& >(ix_clean);
	const volatile float larger = x_clean * (1 << dropBits);
	return (larger + x) - larger;
}

#if _MSC_VER != 0
__declspec(align(32))

#else
__attribute__ ((aligned(32)))

#endif
double ff64[] = {
	1,
	2,
	3,
	4
};

#if _MSC_VER != 0
__declspec(align(32))

#else
__attribute__ ((aligned(32)))

#endif
float ff32[] = {
	1,
	2,
	3,
	4,
	5,
	6,
	7,
	8,
	9,
	10,
	11,
	12,
	13,
	14,
	15,
	16,
	compose_f32(0, 0, 1), // the smallest denorm f32
	compose_f32(0, 1, 0), // the smallest normal f32
	compose_f32(0, 1, 1)  // the smallest normal f32 + 1 ULP
};

enum {
	idenorm = 16,
	inormal,
	inext
};

int main(int argc, char**) {
	const size_t obfuscate = size_t(argc - 1);

	stream::cin.open(stdin);
	stream::cout.open(stdout);
	stream::cerr.open(stderr);

	stream::cout << stream::hex;
	stream::cerr << stream::hex;
	bool success = true;

	// report if Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) are in effect
	const float normal = ff32[inormal];
	const float denorm = ff32[idenorm];
	stream::cout << formatter_f32(normal) << ", " << formatter_f32(.5f * normal) << '\n';
	stream::cout << formatter_f32(denorm) << ", " << formatter_f32((1 << 23) * denorm) << '\n';

	// assuming absence of both FTZ and DAZ, see if we can detect the smallest possible difference -- the smallest denormal number
	const float next = ff32[inext];
	stream::cout << formatter_f32(normal) << '\n';
	stream::cout << formatter_f32(next) << '\n';
	stream::cout << formatter_f32(next - normal) << ", " << diff_ulp(next, normal) << '\n';

	// report delta between smallest normal and smallest denormal numbers
	{
		const float x = denorm;
		const float y = normal;
		stream::cout << formatter_f32(y) << '\n';
		stream::cout << formatter_f32(x) << '\n';
		stream::cout << formatter_f32(y - x) << ", " << diff_ulp(x, y) << '\n';
	}
	// report several normal deltas
	{
		const float x = compose_f32(0, 0x6, 0x4);
		const float y = compose_f32(0, 0x5, 0xe);
		stream::cout << formatter_f32(y) << '\n';
		stream::cout << formatter_f32(x) << '\n';
		stream::cout << formatter_f32(y - x) << ", " << diff_ulp(x, y) << '\n';
	}
	{
		const float x = compose_f32(0, 0x7e, 1);
		const float y = compose_f32(0, 0x7f, 0);
		stream::cout << formatter_f32(y) << '\n';
		stream::cout << formatter_f32(x) << '\n';
		stream::cout << formatter_f32(y - x) << ", " << diff_ulp(x, y) << '\n';
	}
	{
		const float x = compose_f32(0, 0x7f, 0x400000);
		const float y = compose_f32(0, 0x80, 0);
		stream::cout << formatter_f32(y) << '\n';
		stream::cout << formatter_f32(x) << '\n';
		stream::cout << formatter_f32(y - x) << ", " << diff_ulp(x, y) << '\n';
	}
	{
		const float x = compose_f32(0, 0x7f, 0);
		const float y = compose_f32(0, 0x80, 0);
		stream::cout << formatter_f32(y) << '\n';
		stream::cout << formatter_f32(x) << '\n';
		stream::cout << formatter_f32(y - x) << ", " << diff_ulp(x, y) << '\n';
	}
	{
		const float x = compose_f32(0, 0x7f, 0);
		const float y = compose_f32(0, 0x81, 0);
		stream::cout << formatter_f32(y) << '\n';
		stream::cout << formatter_f32(x) << '\n';
		stream::cout << formatter_f32(y - x) << ", " << diff_ulp(x, y) << '\n';
	}
	{
		const float x = compose_f32(0, 0x7e, 0x7fffff);
		const float y = compose_f32(0, 0x7f, 0);
		stream::cout << formatter_f32(y) << '\n';
		stream::cout << formatter_f32(x) << '\n';
		stream::cout << formatter_f32(y - x) << ", " << diff_ulp(x, y) << '\n';
	}
	{
		const float x = compose_f32(0, 0x7f, 0);
		const float y = compose_f32(0, 0x7f, 1);
		stream::cout << formatter_f32(y) << '\n';
		stream::cout << formatter_f32(x) << '\n';
		stream::cout << formatter_f32(y - x) << ", " << diff_ulp(x, y) << '\n';
	}
	{
		// 1 ULP
		const float x = 2.f;
		const float y = 2.f - 1.f / (1 << 23);
		stream::cout << formatter_f32(x) << '\n';
		stream::cout << formatter_f32(y) << '\n';
		stream::cout << formatter_f32(x - y) << ", " << diff_ulp(x, y) << '\n';
	}
	{
		// 2 ULPs
		const float x = 2.f;
		const float y = 2.f - 2.f / (1 << 23);
		stream::cout << formatter_f32(x) << '\n';
		stream::cout << formatter_f32(y) << '\n';
		stream::cout << formatter_f32(x - y) << ", " << diff_ulp(x, y) << '\n';
	}
	{
		// 3 ULPs
		const float x = 2.f;
		const float y = 2.f - 3.f / (1 << 23);
		stream::cout << formatter_f32(x) << '\n';
		stream::cout << formatter_f32(y) << '\n';
		stream::cout << formatter_f32(x - y) << ", " << diff_ulp(x, y) << '\n';
	}
	{
		// 4 ULPs
		const float x = 2.f;
		const float y = 2.f - 4.f / (1 << 23);
		stream::cout << formatter_f32(x) << '\n';
		stream::cout << formatter_f32(y) << '\n';
		stream::cout << formatter_f32(x - y) << ", " << diff_ulp(x, y) << '\n';
	}

	stream::cout << formatter_f32(compose_f32(1, 127, 0x7fffff)) << ", "
	<< formatter_f32(round_bin(compose_f32(1, 127, 0x7fffff), 2)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 127, 0x7ffffe)) << ", "
	<< formatter_f32(round_bin(compose_f32(1, 127, 0x7ffffe), 2)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 127, 0x7ffffd)) << ", "
	<< formatter_f32(round_bin(compose_f32(1, 127, 0x7ffffd), 2)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 127, 0x7ffffc)) << ", "
	<< formatter_f32(round_bin(compose_f32(1, 127, 0x7ffffc), 2)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 254, 0x6fffff)) << ", "
	<< formatter_f32(round_bin(compose_f32(1, 254, 0x6fffff), 2)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 254, 0x7fffff)) << ", "
	<< formatter_f32(round_bin(compose_f32(1, 254, 0x7fffff), 2)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 127, 0x7fffff)) << ", "
	<< formatter_f32(round_bin(compose_f32(1, 127, 0x7fffff), 0)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 127, 0x7fffff)) << ", "
	<< formatter_f32(round_bin_unsafe(compose_f32(1, 127, 0x7fffff), 2)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 127, 0x7ffffe)) << ", "
	<< formatter_f32(round_bin_unsafe(compose_f32(1, 127, 0x7ffffe), 2)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 127, 0x7ffffd)) << ", "
	<< formatter_f32(round_bin_unsafe(compose_f32(1, 127, 0x7ffffd), 2)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 127, 0x7ffffc)) << ", "
	<< formatter_f32(round_bin_unsafe(compose_f32(1, 127, 0x7ffffc), 2)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 254, 0x6fffff)) << ", "
	<< formatter_f32(round_bin_unsafe(compose_f32(1, 254, 0x6fffff), 2)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 254, 0x7fffff)) << ", "
	<< formatter_f32(round_bin_unsafe(compose_f32(1, 254, 0x7fffff), 2)) << '\n';

	stream::cout << formatter_f32(compose_f32(1, 127, 0x7fffff)) << ", "
	<< formatter_f32(round_bin_unsafe(compose_f32(1, 127, 0x7fffff), 0)) << '\n';

	// addition & predicates ///////////////////////////////////////////////////////////////////////

#if NATIVE_F64X4 != 0
	{
		const f64x4 x = f64x4(1, 2, 3, 4) + f64x4(4, 3, 2, 1);

		if (!all(x == f64x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != f64x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != f64x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_S64X4 != 0
	{
		const s64x4 x = s64x4(1, 2, 3, 4) + s64x4(4, 3, 2, 1);

		if (!all(x == s64x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != s64x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != s64x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U64X4 != 0
	{
		const u64x4 x = u64x4(1, 2, 3, 4) + u64x4(4, 3, 2, 1);

		if (!all(x == u64x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != u64x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != u64x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const f64x2 x = f64x2(1, 2) + f64x2(4, 3);

		if (!all(x == f64x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != f64x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != f64x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s64x2 x = s64x2(1, 2) + s64x2(4, 3);

		if (!all(x == s64x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != s64x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != s64x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u64x2 x = u64x2(1, 2) + u64x2(4, 3);

		if (!all(x == u64x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != u64x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != u64x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#if NATIVE_F32X8 != 0
	{
		const f32x8 x = f32x8(1, 2, 3, 4, 5, 6, 7, 8) + f32x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == f32x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != f32x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != f32x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_S32X8 != 0
	{
		const s32x8 x = s32x8(1, 2, 3, 4, 5, 6, 7, 8) + s32x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == s32x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != s32x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != s32x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U32X8 != 0
	{
		const u32x8 x = u32x8(1, 2, 3, 4, 5, 6, 7, 8) + u32x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u32x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != u32x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != u32x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const f32x4 x = f32x4(1, 2, 3, 4) + f32x4(4, 3, 2, 1);

		if (!all(x == f32x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != f32x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != f32x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s32x4 x = s32x4(1, 2, 3, 4) + s32x4(4, 3, 2, 1);

		if (!all(x == s32x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != s32x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != s32x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = u32x4(1, 2, 3, 4) + u32x4(4, 3, 2, 1);

		if (!all(x == u32x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != u32x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != u32x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_F32X2 != 0
	{
		const f32x2 x = f32x2(1, 2) + f32x2(4, 3);

		if (!all(x == f32x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != f32x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != f32x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_S32X2 != 0
	{
		const s32x2 x = s32x2(1, 2) + s32x2(4, 3);

		if (!all(x == s32x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != s32x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != s32x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U32X2 != 0
	{
		const u32x2 x = u32x2(1, 2) + u32x2(4, 3);

		if (!all(x == u32x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != u32x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != u32x2(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_S16X16 != 0
	{
		const s16x16 x = s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) +
			s16x16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == s16x16(17))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != s16x16(17))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != s16x16(17))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U16X16 != 0
	{
		const u16x16 x = u16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) +
			u16x16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u16x16(17))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != u16x16(17))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != u16x16(17))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const s16x8 x = s16x8(1, 2, 3, 4, 5, 6, 7, 8) + s16x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == s16x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != s16x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != s16x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x8 x = u16x8(1, 2, 3, 4, 5, 6, 7, 8) + u16x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u16x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != u16x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != u16x8(9))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_S16X4 != 0
	{
		const s16x4 x = s16x4(1, 2, 3, 4) + s16x4(4, 3, 2, 1);

		if (!all(x == s16x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != s16x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != s16x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U16X4 != 0
	{
		const u16x4 x = u16x4(1, 2, 3, 4) + u16x4(4, 3, 2, 1);

		if (!all(x == u16x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (any(x != u16x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}

		if (!none(x != u16x4(5))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	// subtraction /////////////////////////////////////////////////////////////////////////////////

#if NATIVE_F64X4 != 0
	{
		const f64x4 x = f64x4(5, 6, 7, 8) - f64x4(1, 2, 3, 4);

		if (!all(x == f64x4(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_S64X4 != 0
	{
		const s64x4 x = s64x4(5, 6, 7, 8) - s64x4(1, 2, 3, 4);

		if (!all(x == s64x4(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U64X4 != 0
	{
		const u64x4 x = u64x4(5, 6, 7, 8) - u64x4(1, 2, 3, 4);

		if (!all(x == u64x4(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const f64x2 x = f64x2(5, 6) - f64x2(1, 2);

		if (!all(x == f64x2(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s64x2 x = s64x2(5, 6) - s64x2(1, 2);

		if (!all(x == s64x2(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u64x2 x = u64x2(5, 6) - u64x2(1, 2);

		if (!all(x == u64x2(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#if NATIVE_F32X8 != 0
	{
		const f32x8 x = f32x8(5, 6, 7, 8, 9, 10, 11, 12) - f32x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == f32x8(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_S32X8 != 0
	{
		const s32x8 x = s32x8(5, 6, 7, 8, 9, 10, 11, 12) - s32x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == s32x8(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U32X8 != 0
	{
		const u32x8 x = u32x8(5, 6, 7, 8, 9, 10, 11, 12) - u32x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == u32x8(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const f32x4 x = f32x4(5, 6, 7, 8) - f32x4(1, 2, 3, 4);

		if (!all(x == f32x4(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s32x4 x = s32x4(5, 6, 7, 8) - s32x4(1, 2, 3, 4);

		if (!all(x == s32x4(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = u32x4(5, 6, 7, 8) - u32x4(1, 2, 3, 4);

		if (!all(x == u32x4(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_F32X2 != 0
	{
		const f32x2 x = f32x2(5, 6) - f32x2(1, 2);

		if (!all(x == f32x2(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_S32X2 != 0
	{
		const s32x2 x = s32x2(5, 6) - s32x2(1, 2);

		if (!all(x == s32x2(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U32X2 != 0
	{
		const u32x2 x = u32x2(5, 6) - u32x2(1, 2);

		if (!all(x == u32x2(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_S16X16 != 0
	{
		const s16x16 x = s16x16(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20) -
			s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

		if (!all(x == s16x16(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U16X16 != 0
	{
		const u16x16 x = u16x16(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20) -
			u16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

		if (!all(x == u16x16(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const s16x8 x = s16x8(5, 6, 7, 8, 9, 10, 11, 12) - s16x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == s16x8(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x8 x = u16x8(5, 6, 7, 8, 9, 10, 11, 12) - u16x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == u16x8(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_S16X4 != 0
	{
		const s16x4 x = s16x4(5, 6, 7, 8) - s16x4(1, 2, 3, 4);

		if (!all(x == s16x4(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U16X4 != 0
	{
		const u16x4 x = u16x4(5, 6, 7, 8) - u16x4(1, 2, 3, 4);

		if (!all(x == u16x4(4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	// multiplication //////////////////////////////////////////////////////////////////////////////

#if NATIVE_F64X4 != 0
	{
		const f64x4 x = f64x4(1, 2, 3, 4) * f64x4(4, 3, 2, 1);

		if (!all(x == f64x4(4, 6, 6, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const f64x2 x = f64x2(1, 2) * f64x2(4, 3);

		if (!all(x == f64x2(4, 6))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_F32X8 != 0
	{
		const f32x8 x = f32x8(1, 2, 3, 4, 5, 6, 7, 8) * f32x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == f32x8(8, 14, 18, 20, 20, 18, 14, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_S32X8 != 0
	{
		const s32x8 x = s32x8(1, 2, 3, 4, 5, 6, 7, 8) * s32x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == s32x8(8, 14, 18, 20, 20, 18, 14, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U32X8 != 0
	{
		const u32x8 x = u32x8(1, 2, 3, 4, 5, 6, 7, 8) * u32x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u32x8(8, 14, 18, 20, 20, 18, 14, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const f32x4 x = f32x4(1, 2, 3, 4) * f32x4(4, 3, 2, 1);

		if (!all(x == f32x4(4, 6, 6, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s32x4 x = s32x4(1, 2, 3, 4) * s32x4(4, 3, 2, 1);

		if (!all(x == s32x4(4, 6, 6, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = u32x4(1, 2, 3, 4) * u32x4(4, 3, 2, 1);

		if (!all(x == u32x4(4, 6, 6, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_F32X2 != 0
	{
		const f32x2 x = f32x2(1, 2) * f32x2(4, 3);

		if (!all(x == f32x2(4, 6))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_S32X2 != 0
	{
		const s32x2 x = s32x2(1, 2) * s32x2(4, 3);

		if (!all(x == s32x2(4, 6))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U32X2 != 0
	{
		const u32x2 x = u32x2(1, 2) * u32x2(4, 3);

		if (!all(x == u32x2(4, 6))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_S16X16 != 0
	{
		const s16x16 x = s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) *
			s16x16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == s16x16(16, 30, 42, 52, 60, 66, 70, 72, 72, 70, 66, 60, 52, 42, 30, 16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U16X16 != 0
	{
		const u16x16 x = u16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) *
			u16x16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u16x16(16, 30, 42, 52, 60, 66, 70, 72, 72, 70, 66, 60, 52, 42, 30, 16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const s16x8 x = s16x8(1, 2, 3, 4, 5, 6, 7, 8) * s16x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == s16x8(8, 14, 18, 20, 20, 18, 14, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

	{
		const u16x8 x = u16x8(1, 2, 3, 4, 5, 6, 7, 8) * u16x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u16x8(8, 14, 18, 20, 20, 18, 14, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_S16X4 != 0
	{
		const s16x4 x = s16x4(1, 2, 3, 4) * s16x4(4, 3, 2, 1);

		if (!all(x == s16x4(4, 6, 6, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U16X4 != 0
	{
		const u16x4 x = u16x4(1, 2, 3, 4) * u16x4(4, 3, 2, 1);

		if (!all(x == u16x4(4, 6, 6, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	// division ////////////////////////////////////////////////////////////////////////////////////

#if NATIVE_F64X4 != 0
	{
		const f64x4 x = reinterpret_cast< const f64x4& >(ff64[0]) / reinterpret_cast< const f64x4& >(ff64[0 + obfuscate]);

		if (!all(x == f64x4(1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const f64x2 x = reinterpret_cast< const f64x2& >(ff64[0]) / reinterpret_cast< const f64x2& >(ff64[0 + obfuscate]);

		if (!all(x == f64x2(1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_F32X8 != 0
	{
		const f32x8 x = reinterpret_cast< const f32x8& >(ff32[0]) / reinterpret_cast< const f32x8& >(ff32[0 + obfuscate]);

		// some compilers can go 1 ULP off (normally below) the mark -- never mind those
		if (!all(abs(f32x8(1) - x) <= f32x8(1.f / (1 << 24)))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const f32x4 x = reinterpret_cast< const f32x4& >(ff32[0]) / reinterpret_cast< const f32x4& >(ff32[0 + obfuscate]);

		// some compilers can go 1 ULP off (normally below) the mark -- never mind those
		if (!all(abs(f32x4(1) - x) <= f32x4(1.f / (1 << 24)))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_F32X2 != 0
	{
		const f32x2 x = reinterpret_cast< const f32x2& >(ff32[0]) / reinterpret_cast< const f32x2& >(ff32[0 + obfuscate]);

		if (!all(x == f32x2(1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	// negation ////////////////////////////////////////////////////////////////////////////////////
	{
		const s16x8 x = -s16x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == s16x8(-1, -2, -3, -4, -5, -6, -7, -8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s32x4 x = -s32x4(1, 2, 3, 4);

		if (!all(x == s32x4(-1, -2, -3, -4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const f32x4 x = -f32x4(1, 2, 3, 4);

		if (!all(x == f32x4(-1, -2, -3, -4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const f64x2 x = -f64x2(1, 2);

		if (!all(x == f64x2(-1, -2))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#if NATIVE_F64X4 != 0
	{
		const f64x4 x = -f64x4(1, 2, 3, 4);

		if (!all(x == f64x4(-1, -2, -3, -4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_F32X8 != 0
	{
		const f32x8 x = -f32x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == f32x8(-1, -2, -3, -4, -5, -6, -7, -8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S32X8 != 0
	{
		const s32x8 x = -s32x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == s32x8(-1, -2, -3, -4, -5, -6, -7, -8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S16X16 != 0
	{
		const s16x16 x = -s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

		if (!all(x == s16x16(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
	// less-than ///////////////////////////////////////////////////////////////////////////////////
	{
		const u64x2 x = f64x2(1, 2) < f64x2(2, 1);

		if (!all(x == u64x2(-1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = f32x4(1, 2, 3, 4) < f32x4(4, 3, 2, 1);

		if (!all(x == u32x4(-1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = s32x4(1, 2, 3, 4) < s32x4(4, 3, 2, 1);

		if (!all(x == u32x4(-1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x8 x = s16x8(1, 2, 3, 4, 5, 6, 7, 8) < s16x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u16x8(-1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#if NATIVE_F64X4 != 0
	{
		const u64x4 x = f64x4(1, 2, 3, 4) < f64x4(4, 3, 2, 1);

		if (!all(x == u64x4(-1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_F32X8 != 0
	{
		const u32x8 x = f32x8(1, 2, 3, 4, 5, 6, 7, 8) < f32x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u32x8(-1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S32X8 != 0
	{
		const u32x8 x = s32x8(1, 2, 3, 4, 5, 6, 7, 8) < s32x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u32x8(-1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S16X16 != 0
	{
		const u16x16 x = s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15) <
			s16x16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u16x16(-1, -1, -1, -1, -1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
	// less-or-equal ///////////////////////////////////////////////////////////////////////////////
	{
		const u64x2 x = f64x2(1, 2) <= f64x2(2, 1);

		if (!all(x == u64x2(-1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = f32x4(1, 2, 3, 4) <= f32x4(4, 3, 2, 1);

		if (!all(x == u32x4(-1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = s32x4(1, 2, 3, 4) <= s32x4(4, 3, 2, 1);

		if (!all(x == u32x4(-1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x8 x = s16x8(1, 2, 3, 4, 5, 6, 7, 8) <= s16x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u16x8(-1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#if NATIVE_F64X4 != 0
	{
		const u64x4 x = f64x4(1, 2, 3, 4) <= f64x4(4, 3, 2, 1);

		if (!all(x == u64x4(-1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_F32X8 != 0
	{
		const u32x8 x = f32x8(1, 2, 3, 4, 5, 6, 7, 8) <= f32x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u32x8(-1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S32X8 != 0
	{
		const u32x8 x = s32x8(1, 2, 3, 4, 5, 6, 7, 8) <= s32x8(8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u32x8(-1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S16X16 != 0
	{
		const u16x16 x = s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) <=
			s16x16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

		if (!all(x == u16x16(-1, -1, -1, -1, -1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
	// greater-than ////////////////////////////////////////////////////////////////////////////////
	{
		const u64x2 x = f64x2(2, 1) > f64x2(1, 2);

		if (!all(x == u64x2(-1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = f32x4(4, 3, 2, 1) > f32x4(1, 2, 3, 4);

		if (!all(x == u32x4(-1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = s32x4(4, 3, 2, 1) > s32x4(1, 2, 3, 4);

		if (!all(x == u32x4(-1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x8 x = s16x8(8, 7, 6, 5, 4, 3, 2, 1) > s16x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == u16x8(-1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#if NATIVE_F64X4 != 0
	{
		const u64x4 x = f64x4(4, 3, 2, 1) > f64x4(1, 2, 3, 4);

		if (!all(x == u64x4(-1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_F32X8 != 0
	{
		const u32x8 x = f32x8(8, 7, 6, 5, 4, 3, 2, 1) > f32x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == u32x8(-1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S32X8 != 0
	{
		const u32x8 x = s32x8(8, 7, 6, 5, 4, 3, 2, 1) > s32x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == u32x8(-1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S16X16 != 0
	{
		const u16x16 x = s16x16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1) >
			s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

		if (!all(x == u16x16(-1, -1, -1, -1, -1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
	// greater-or-equal ////////////////////////////////////////////////////////////////////////////
	{
		const u64x2 x = f64x2(2, 1) >= f64x2(1, 2);

		if (!all(x == u64x2(-1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = f32x4(4, 3, 2, 1) >= f32x4(1, 2, 3, 4);

		if (!all(x == u32x4(-1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = s32x4(4, 3, 2, 1) >= s32x4(1, 2, 3, 4);

		if (!all(x == u32x4(-1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x8 x = s16x8(8, 7, 6, 5, 4, 3, 2, 1) >= s16x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == u16x8(-1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#if NATIVE_F64X4 != 0
	{
		const u64x4 x = f64x4(4, 3, 2, 1) >= f64x4(1, 2, 3, 4);

		if (!all(x == u64x4(-1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_F32X8 != 0
	{
		const u32x8 x = f32x8(8, 7, 6, 5, 4, 3, 2, 1) >= f32x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == u32x8(-1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S32X8 != 0
	{
		const u32x8 x = s32x8(8, 7, 6, 5, 4, 3, 2, 1) >= s32x8(1, 2, 3, 4, 5, 6, 7, 8);

		if (!all(x == u32x8(-1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S16X16 != 0
	{
		const u16x16 x = s16x16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1) >=
			s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

		if (!all(x == u16x16(-1, -1, -1, -1, -1, -1, -1, -1, flag_zero()))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
	// absolute value //////////////////////////////////////////////////////////////////////////////
	{
		const s16x8 x = abs(s16x8(-1, -2, -3, -4, 1, 2, 3, 4));

		if (!all(x == s16x8(1, 2, 3, 4, 1, 2, 3, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s32x4 x = abs(s32x4(-1, -2, -3, -4));

		if (!all(x == s32x4(1, 2, 3, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const f32x4 x = abs(f32x4(-1, -2, -3, -4));

		if (!all(x == f32x4(1, 2, 3, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const f64x2 x = abs(f64x2(-1, -2));

		if (!all(x == f64x2(1, 2))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#if NATIVE_F64X4 != 0
	{
		const f64x4 x = abs(f64x4(-1, -2, -3, -4));

		if (!all(x == f64x4(1, 2, 3, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_F32X8 != 0
	{
		const f32x8 x = abs(f32x8(-1, -2, -3, -4, -5, -6, -7, -8));

		if (!all(x == f32x8(1, 2, 3, 4, 5, 6, 7, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S32X8 != 0
	{
		const s32x8 x = abs(s32x8(-1, -2, -3, -4, -5, -6, -7, -8));

		if (!all(x == s32x8(1, 2, 3, 4, 5, 6, 7, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S16X16 != 0
	{
		const s16x16 x = abs(s16x16(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16));

		if (!all(x == s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
	// masking /////////////////////////////////////////////////////////////////////////////////////
	{
		const f64x2 x = mask(f64x2(1, 2), u64x2(0, -1));

		if (!all(x == f64x2(0, 2))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const f32x4 x = mask(f32x4(1, 2, 3, 4), u32x4(0, -1, 0, -1));

		if (!all(x == f32x4(0, 2, 0, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s32x4 x = mask(s32x4(-1, -2, -3, -4), u32x4(0, -1, 0, -1));

		if (!all(x == s32x4(0, -2, 0, -4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = mask(u32x4(1, 2, 3, 4), u32x4(0, -1, 0, -1));

		if (!all(x == u32x4(0, 2, 0, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s16x8 x = mask(s16x8(-1, -2, -3, -4, 1, 2, 3, 4), u16x8(0, -1, 0, -1, 0, -1, 0, -1));

		if (!all(x == s16x8(0, -2, 0, -4, 0, 2, 0, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x8 x = mask(u16x8(1, 2, 3, 4, 5, 6, 7, 8), u16x8(0, -1, 0, -1, 0, -1, 0, -1));

		if (!all(x == u16x8(0, 2, 0, 4, 0, 6, 0, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#if NATIVE_F64X4 != 0
	{
		const f64x4 x = mask(f64x4(1, 2, 3, 4), u64x4(0, -1, 0, -1));

		if (!all(x == f64x4(0, 2, 0, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S64X4 != 0
	{
		const s64x4 x = mask(s64x4(1, 2, 3, 4), u64x4(0, -1, 0, -1));

		if (!all(x == s64x4(0, 2, 0, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_U64X4 != 0
	{
		const u64x4 x = mask(u64x4(1, 2, 3, 4), u64x4(0, -1, 0, -1));

		if (!all(x == u64x4(0, 2, 0, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_F32X8 != 0
	{
		const f32x8 x = mask(f32x8(1, 2, 3, 4, 5, 6, 7, 8), u32x8(0, -1, 0, -1, 0, -1, 0, -1));

		if (!all(x == f32x8(0, 2, 0, 4, 0, 6, 0, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S16X16 != 0
	{
		const s16x16 x = mask(s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), u16x16(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1));

		if (!all(x == s16x16(0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0, 14, 0, 16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_U16X16 != 0
	{
		const u16x16 x = mask(u16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), u16x16(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1));

		if (!all(x == u16x16(0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0, 14, 0, 16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
	// selection by bitmask literal ////////////////////////////////////////////////////////////////
	{
		const f64x2 x = select< 0x2 >(f64x2(1, 2), f64x2(3, 4));

		if (!all(x == f64x2(1, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const f32x4 x = select< 0xa >(f32x4(1, 2, 3, 4), f32x4(5, 6, 7, 8));

		if (!all(x == f32x4(1, 6, 3, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s32x4 x = select< 0xa >(s32x4(-1, -2, -3, -4), s32x4(5, 6, 7, 8));

		if (!all(x == s32x4(-1, 6, -3, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = select< 0xa >(u32x4(1, 2, 3, 4), u32x4(5, 6, 7, 8));

		if (!all(x == u32x4(1, 6, 3, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s16x8 x = select< 0xaa >(s16x8(-1, -2, -3, -4, 1, 2, 3, 4), s16x8(5, 6, 7, 8, -5, -6, -7, -8));

		if (!all(x == s16x8(-1, 6, -3, 8, 1, -6, 3, -8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x8 x = select< 0xaa >(u16x8(1, 2, 3, 4, 1, 2, 3, 4), u16x8(5, 6, 7, 8, 5, 6, 7, 8));

		if (!all(x == u16x8(1, 6, 3, 8, 1, 6, 3, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#if NATIVE_F64X4 != 0
	{
		const f64x4 x = select< 0xa >(f64x4(1, 2, 3, 4), f64x4(5, 6, 7, 8));

		if (!all(x == f64x4(1, 6, 3, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_F32X8 != 0
	{
		const f32x8 x = select< 0xaa >(f32x8(1, 2, 3, 4, 5, 6, 7, 8), f32x8(-1, -2, -3, -4, -5, -6, -7, -8));

		if (!all(x == f32x8(1, -2, 3, -4, 5, -6, 7, -8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S32X8 != 0
	{
		const s32x8 x = select< 0xaa >(s32x8(1, 2, 3, 4, 5, 6, 7, 8), s32x8(-1, -2, -3, -4, -5, -6, -7, -8));

		if (!all(x == s32x8(1, -2, 3, -4, 5, -6, 7, -8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_U32X8 != 0
	{
		const u32x8 x = select< 0xaa >(u32x8(1, 2, 3, 4, 5, 6, 7, 8), u32x8(9, 10, 11, 12, 13, 14, 15, 16));

		if (!all(x == u32x8(1, 10, 3, 12, 5, 14, 7, 16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S16X16 != 0
	{
		const s16x16 x = select< 0xaaaa >(s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), s16x16(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32));

		if (!all(x == s16x16(1, 18, 3, 20, 5, 22, 7, 24, 9, 26, 11, 28, 13, 30, 15, 32))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_U16X16 != 0
	{
		const u16x16 x = select< 0xaaaa >(u16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), u16x16(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32));

		if (!all(x == u16x16(1, 18, 3, 20, 5, 22, 7, 24, 9, 26, 11, 28, 13, 30, 15, 32))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
	// selection by lane mask //////////////////////////////////////////////////////////////////////
	{
		const f64x2 x = select(f64x2(1, 2), f64x2(3, 4), u64x2(0, -1));

		if (!all(x == f64x2(1, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const f32x4 x = select(f32x4(1, 2, 3, 4), f32x4(5, 6, 7, 8), u32x4(0, -1, 0, -1));

		if (!all(x == f32x4(1, 6, 3, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s32x4 x = select(s32x4(-1, -2, -3, -4), s32x4(5, 6, 7, 8), u32x4(0, -1, 0, -1));

		if (!all(x == s32x4(-1, 6, -3, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = select(u32x4(1, 2, 3, 4), u32x4(5, 6, 7, 8), u32x4(0, -1, 0, -1));

		if (!all(x == u32x4(1, 6, 3, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s16x8 x = select(s16x8(-1, -2, -3, -4, 1, 2, 3, 4), s16x8(5, 6, 7, 8, -5, -6, -7, -8), u16x8(0, -1, 0, -1, 0, -1, 0, -1));

		if (!all(x == s16x8(-1, 6, -3, 8, 1, -6, 3, -8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x8 x = select(u16x8(1, 2, 3, 4, 5, 6, 7, 8), u16x8(9, 10, 11, 12, 13, 14, 15, 16), u16x8(0, -1, 0, -1, 0, -1, 0, -1));

		if (!all(x == u16x8(1, 10, 3, 12, 5, 14, 7, 16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#if NATIVE_F64X4 != 0
	{
		const f64x4 x = select(f64x4(1, 2, 3, 4), f64x4(5, 6, 7, 8), u64x4(0, -1, 0, -1));

		if (!all(x == f64x4(1, 6, 3, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_F32X8 != 0
	{
		const f32x8 x = select(f32x8(1, 2, 3, 4, 5, 6, 7, 8), f32x8(-1, -2, -3, -4, -5, -6, -7, -8), u32x8(0, -1, 0, -1, 0, -1, 0, -1));

		if (!all(x == f32x8(1, -2, 3, -4, 5, -6, 7, -8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S32X8 != 0
	{
		const s32x8 x = select(s32x8(1, 2, 3, 4, 5, 6, 7, 8), s32x8(-1, -2, -3, -4, -5, -6, -7, -8), u32x8(0, -1, 0, -1, 0, -1, 0, -1));

		if (!all(x == s32x8(1, -2, 3, -4, 5, -6, 7, -8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_U32X8 != 0
	{
		const u32x8 x = select(u32x8(1, 2, 3, 4, 5, 6, 7, 8), u32x8(9, 10, 11, 12, 13, 14, 15, 16), u32x8(0, -1, 0, -1, 0, -1, 0, -1));

		if (!all(x == u32x8(1, 10, 3, 12, 5, 14, 7, 16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_S16X16 != 0
	{
		const s16x16 x = select(s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), s16x16(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32),
			u16x16(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1));

		if (!all(x == s16x16(1, 18, 3, 20, 5, 22, 7, 24, 9, 26, 11, 28, 13, 30, 15, 32))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_U16X16 != 0
	{
		const u16x16 x = select(u16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), u16x16(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32),
			u16x16(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1));

		if (!all(x == u16x16(1, 18, 3, 20, 5, 22, 7, 24, 9, 26, 11, 28, 13, 30, 15, 32))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif

	// bitshift left ///////////////////////////////////////////////////////////////////////////////
#if NATIVE_S64X4 !=0
	{
		const s64x4 x = shl(s64x4(1, 2, 3, 4), u64x4(1, 0, 2, 2));

		if (!all(x == s64x4(2, 2, 12, 16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_U64X4 != 0
	{
		const u64x4 x = shl(u64x4(1, 2, 3, 4), u64x4(1, 0, 2, 2));

		if (!all(x == u64x4(2, 2, 12, 16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
	{
		const s64x2 x = shl(s64x2(1, 2), u64x2(1, 2));

		if (!all(x == s64x2(2, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u64x2 x = shl(u64x2(1, 2), u64x2(1, 2));

		if (!all(x == u64x2(2, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#if NATIVE_S16X16 != 0
	{
		const s16x16 x = shl(s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), u16x16(1, 0, 0, 0, 2, 2, 2, 2, 1, 0, 0, 0, 2, 2, 2, 2));

		if (!all(x == s16x16(2, 2, 3, 4, 20, 24, 28, 32, 18, 10, 11, 12, 52, 56, 60, 64))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
#if NATIVE_U16X16 != 0
	{
		const u16x16 x = shl(u16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), u16x16(1, 0, 0, 0, 2, 2, 2, 2, 1, 0, 0, 0, 2, 2, 2, 2));

		if (!all(x == u16x16(2, 2, 3, 4, 20, 24, 28, 32, 18, 10, 11, 12, 52, 56, 60, 64))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
#endif
	{
		const s16x8 x = shl(s16x8(1, 2, 3, 4, 5, 6, 7, 8), u16x8(1, 0, 0, 0, 2, 2, 2, 2));

		if (!all(x == s16x8(2, 2, 3, 4, 20, 24, 28, 32))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x8 x = shl(u16x8(1, 2, 3, 4, 5, 6, 7, 8), u16x8(1, 0, 0, 0, 2, 2, 2, 2));

		if (!all(x == u16x8(2, 2, 3, 4, 20, 24, 28, 32))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

	// extrema /////////////////////////////////////////////////////////////////////////////////////
#if NATIVE_F64X4 != 0
	{
		const f64x4 x = min(f64x4(1, 2, 3, 4), f64x4(4, 3, 2, 1));

		if (!all(x == f64x4(1, 2, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const f64x4 x = max(f64x4(1, 2, 3, 4), f64x4(4, 3, 2, 1));

		if (!all(x == f64x4(4, 3, 3, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const f64x2 x = min(f64x2(1, 2), f64x2(2, 1));

		if (!all(x == f64x2(1, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const f64x2 x = max(f64x2(1, 2), f64x2(2, 1));

		if (!all(x == f64x2(2, 2))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_F32X8 != 0
	{
		const f32x8 x = min(f32x8(1, 2, 3, 4, 5, 6, 7, 8), f32x8(8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == f32x8(1, 2, 3, 4, 4, 3, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const f32x8 x = max(f32x8(1, 2, 3, 4, 5, 6, 7, 8), f32x8(8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == f32x8(8, 7, 6, 5, 5, 6, 7, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const f32x4 x = min(f32x4(1, 2, 3, 4), f32x4(4, 3, 2, 1));

		if (!all(x == f32x4(1, 2, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const f32x4 x = max(f32x4(1, 2, 3, 4), f32x4(4, 3, 2, 1));

		if (!all(x == f32x4(4, 3, 3, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_S32X8 != 0
	{
		const s32x8 x = min(s32x8(1, 2, 3, 4, 5, 6, 7, 8), s32x8(8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == s32x8(1, 2, 3, 4, 4, 3, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s32x8 x = max(s32x8(1, 2, 3, 4, 5, 6, 7, 8), s32x8(8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == s32x8(8, 7, 6, 5, 5, 6, 7, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const s32x4 x = min(s32x4(1, 2, 3, 4), s32x4(4, 3, 2, 1));

		if (!all(x == s32x4(1, 2, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s32x4 x = max(s32x4(1, 2, 3, 4), s32x4(4, 3, 2, 1));

		if (!all(x == s32x4(4, 3, 3, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_U32X8 != 0
	{
		const u32x8 x = min(u32x8(1, 2, 3, 4, 5, 6, 7, 8), u32x8(8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == u32x8(1, 2, 3, 4, 4, 3, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x8 x = max(u32x8(1, 2, 3, 4, 5, 6, 7, 8), u32x8(8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == u32x8(8, 7, 6, 5, 5, 6, 7, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const u32x4 x = min(u32x4(1, 2, 3, 4), u32x4(4, 3, 2, 1));

		if (!all(x == u32x4(1, 2, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u32x4 x = max(u32x4(1, 2, 3, 4), u32x4(4, 3, 2, 1));

		if (!all(x == u32x4(4, 3, 3, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_S16X16 != 0
	{
		const s16x16 x = min(s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
			s16x16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == s16x16(1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s16x16 x = max(s16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
			s16x16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == s16x16(16, 15, 14, 13, 12, 11, 10, 9, 9, 10, 11, 12, 13, 14, 15, 16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const s16x8 x = min(s16x8(1, 2, 3, 4, 5, 6, 7, 8), s16x8(8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == s16x8(1, 2, 3, 4, 4, 3, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s16x8 x = max(s16x8(1, 2, 3, 4, 5, 6, 7, 8), s16x8(8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == s16x8(8, 7, 6, 5, 5, 6, 7, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_S16X4 != 0
	{
		const s16x4 x = min(s16x4(1, 2, 3, 4), s16x4(4, 3, 2, 1));

		if (!all(x == s16x4(1, 2, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const s16x4 x = max(s16x4(1, 2, 3, 4), s16x4(4, 3, 2, 1));

		if (!all(x == s16x4(4, 3, 3, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
#if NATIVE_U16X16 != 0
	{
		const u16x16 x = min(u16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
			u16x16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == u16x16(1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x16 x = max(u16x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
			u16x16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == u16x16(16, 15, 14, 13, 12, 11, 10, 9, 9, 10, 11, 12, 13, 14, 15, 16))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	{
		const u16x8 x = min(u16x8(1, 2, 3, 4, 5, 6, 7, 8), u16x8(8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == u16x8(1, 2, 3, 4, 4, 3, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x8 x = max(u16x8(1, 2, 3, 4, 5, 6, 7, 8), u16x8(8, 7, 6, 5, 4, 3, 2, 1));

		if (!all(x == u16x8(8, 7, 6, 5, 5, 6, 7, 8))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#if NATIVE_U16X4 != 0
	{
		const u16x4 x = min(u16x4(1, 2, 3, 4), u16x4(4, 3, 2, 1));

		if (!all(x == u16x4(1, 2, 2, 1))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}
	{
		const u16x4 x = max(u16x4(1, 2, 3, 4), u16x4(4, 3, 2, 1));

		if (!all(x == u16x4(4, 3, 3, 4))) {
			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << ": " << x << '\n';
			success = false;
		}
	}

#endif
	// transpose 4x4 ///////////////////////////////////////////////////////////////////////////////
	{
		const f32x4 src0 = reinterpret_cast< const f32x4& >(ff32[ 0]);
		const f32x4 src1 = reinterpret_cast< const f32x4& >(ff32[ 4]);
		const f32x4 src2 = reinterpret_cast< const f32x4& >(ff32[ 8]);
		const f32x4 src3 = reinterpret_cast< const f32x4& >(ff32[12]);
		f32x4 dst0, dst1, dst2, dst3;

		transpose4x4(
			src0,
			src1,
			src2,
			src3,
			dst0,
			dst1,
			dst2,
			dst3);

		if (!all(dst0 == f32x4(0, 4,  8, 12) + f32x4(1)) ||
			!all(dst1 == f32x4(1, 5,  9, 13) + f32x4(1)) ||
			!all(dst2 == f32x4(2, 6, 10, 14) + f32x4(1)) ||
			!all(dst3 == f32x4(3, 7, 11, 15) + f32x4(1))) {

			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << '\n';
			success = false;
		}
	}

#if NATIVE_F32X8 != 0
	{
		const f32x4 src0 = reinterpret_cast< const f32x4& >(ff32[ 0]);
		const f32x4 src1 = reinterpret_cast< const f32x4& >(ff32[ 4]);
		const f32x4 src2 = reinterpret_cast< const f32x4& >(ff32[ 8]);
		const f32x4 src3 = reinterpret_cast< const f32x4& >(ff32[12]);
		f32x4 dst0, dst1, dst2, dst3;

		transpose4x4_ext(
			src0,
			src1,
			src2,
			src3,
			dst0,
			dst1,
			dst2,
			dst3);

		if (!all(dst0 == f32x4(0, 4,  8, 12) + f32x4(1)) ||
			!all(dst1 == f32x4(1, 5,  9, 13) + f32x4(1)) ||
			!all(dst2 == f32x4(2, 6, 10, 14) + f32x4(1)) ||
			!all(dst3 == f32x4(3, 7, 11, 15) + f32x4(1))) {

			stream::cerr << "failed test at line " << stream::dec << __LINE__ << stream::hex << '\n';
			success = false;
		}
	}

#endif
	// sqrt ////////////////////////////////////////////////////////////////////////////////////////
	{
		const size_t steps = 1 << 28;
		const size_t errbreak = 256;
		size_t errcount = 0;
		for (size_t i = 0; i < steps; ++i) {
			const float x = i / float(steps) * 256;
			const float a = std::sqrt(x);
			const f32x4 b = sqrt(f32x4(x));

			if (!all(f32x4(b[0]) == b)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << b << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

			const int32_t ulp = diff_ulp(b[0], a);
			if (-1 == ulp || 1 < ulp) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " <<
					b[0] << " (" << a << "), delta: " << formatter_f32(std::abs(b[0] - a)) << " (" << formatter_f32(b[0]) << ", " << formatter_f32(a) << ", " << ulp << ")\n";

				success = false;
				if (++errcount == errbreak)
					break;
			}

#if NATIVE_F32X8 != 0
			const f32x8 c = sqrt(f32x8(x));

			if (!all(f32x8(c[0]) == c)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << c << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

			if (!all(get_f32x4< 0 >(c) == b)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << c << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

#endif
		}
	}

	// log /////////////////////////////////////////////////////////////////////////////////////////
	{
		const size_t steps = 1 << 28;
		const size_t errbreak = 256;
		size_t errcount = 0;
		for (size_t i = 1; i <= steps; ++i) {
			const float x = i / float(steps) * 256;
			const float a = std::log(x);
			const f32x4 b = log(f32x4(x));

			if (!all(f32x4(b[0]) == b)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << b << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

			const int32_t ulp = diff_ulp(b[0], a);
			if (-1 == ulp || 2 < ulp) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " <<
					b[0] << " (" << a << "), delta: " << formatter_f32(std::abs(b[0] - a)) << " (" << formatter_f32(b[0]) << ", " << formatter_f32(a) << ", " << ulp << ")\n";

				success = false;
				if (++errcount == errbreak)
					break;
			}

#if NATIVE_F32X8 != 0
			const f32x8 c = log(f32x8(x));

			if (!all(f32x8(c[0]) == c)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << c << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

			if (!all(get_f32x4< 0 >(c) == b)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << c << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

#endif
		}
	}

	// exp /////////////////////////////////////////////////////////////////////////////////////////
	{
		const size_t steps = 1 << 28;
		const size_t errbreak = 256;
		size_t errcount = 0;
		for (size_t i = 0; i < steps; ++i) {
			const float x = i / float(steps) * 86 * 2 - 86;
			const float a = std::exp(x);
			const f32x4 b = exp(f32x4(x));

			if (!all(f32x4(b[0]) == b)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << b << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

			const int32_t ulp = diff_ulp(b[0], a);
			if (-1 == ulp || 2 < ulp) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " <<
					b[0] << " (" << a << "), delta: " << formatter_f32(std::abs(b[0] - a)) << " (" << formatter_f32(b[0]) << ", " << formatter_f32(a) << ", " << ulp << ")\n";

				success = false;
				if (++errcount == errbreak)
					break;
			}

#if NATIVE_F32X8 != 0
			const f32x8 c = exp(f32x8(x));

			if (!all(f32x8(c[0]) == c)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << c << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

			if (!all(get_f32x4< 0 >(c) == b)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << c << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

#endif
		}
	}

	// pow /////////////////////////////////////////////////////////////////////////////////////////
	{
		const size_t steps0 = 1 << 24;
		const size_t steps1 = 1 << 4;
		const size_t errbreak = 128;
		size_t errcount = 0;
		for (size_t i = 1; i <= steps0; ++i) {
			const float x = i / float(steps0) * 2;

			for (size_t j = 0; j < steps1; ++j) {
				const float y = j / float(steps1);
				const float a = std::pow(x, y);
				const f32x4 b = pow(f32x4(x), f32x4(y));

				if (!all(f32x4(b[0]) == b)) {
					stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << ", " << j << stream::hex << ": " << b << '\n';
					success = false;
					if (++errcount == errbreak)
						break;
				}

				const int32_t ulp = diff_ulp(b[0], a);
				if (-1 == ulp || 14 < ulp) {
					stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << ", " << j << stream::hex << ": " <<
						b[0] << " (" << a << "), delta: " << formatter_f32(std::abs(b[0] - a)) << " (" << formatter_f32(b[0]) << ", " << formatter_f32(a) << ", " << ulp << ")\n";

					success = false;
					if (++errcount == errbreak)
						break;
				}

#if NATIVE_F32X8 != 0
				const f32x8 c = pow(f32x8(x), f32x8(y));

				if (!all(f32x8(c[0]) == c)) {
					stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << c << '\n';
					success = false;
					if (++errcount == errbreak)
						break;
				}

				if (!all(get_f32x4< 0 >(c) == b)) {
					stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << c << '\n';
					success = false;
					if (++errcount == errbreak)
						break;
				}

#endif
			}
			if (errcount == errbreak)
				break;
		}
	}

	// sin /////////////////////////////////////////////////////////////////////////////////////////
	{
		const size_t steps = 1 << 28;
		const size_t errbreak = 256;
		size_t errcount = 0;
		for (size_t i = 0; i < steps; ++i) {
			const float x = i / float(steps) * M_PI * 4 - M_PI * 2;
			const float a = std::sin(x);
			const f32x4 b = sin(f32x4(x));

			if (!all(f32x4(b[0]) == b)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << b << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

			const int32_t ulp = diff_ulp(b[0], a);
			if (-1 == ulp || 14 < ulp) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " <<
					b[0] << " (" << a << "), delta: " << formatter_f32(std::abs(b[0] - a)) << " (" << formatter_f32(b[0]) << ", " << formatter_f32(a) << ", " << ulp << ")\n";

				success = false;
				if (++errcount == errbreak)
					break;
			}

#if NATIVE_F32X8 != 0
			const f32x8 c = sin(f32x8(x));

			if (!all(f32x8(c[0]) == c)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << c << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

			if (!all(get_f32x4< 0 >(c) == b)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << c << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

#endif
		}
	}

	// cos /////////////////////////////////////////////////////////////////////////////////////////
	{
		const size_t steps = 1 << 28;
		const size_t errbreak = 256;
		size_t errcount = 0;
		for (size_t i = 0; i < steps; ++i) {
			const float x = i / float(steps) * M_PI * 4 - M_PI * 2;
			const float a = std::cos(x);
			const f32x4 b = cos(f32x4(x));

			if (!all(f32x4(b[0]) == b)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << b << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

			const int32_t ulp = diff_ulp(b[0], a);
			if (-1 == ulp || 14 < ulp) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " <<
					b[0] << " (" << a << "), delta: " << formatter_f32(std::abs(b[0] - a)) << " (" << formatter_f32(b[0]) << ", " << formatter_f32(a) << ", " << ulp << ")\n";

				success = false;
				if (++errcount == errbreak)
					break;
			}

#if NATIVE_F32X8 != 0
			const f32x8 c = cos(f32x8(x));

			if (!all(f32x8(c[0]) == c)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << c << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

			if (!all(get_f32x4< 0 >(c) == b)) {
				stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << c << '\n';
				success = false;
				if (++errcount == errbreak)
					break;
			}

#endif
		}
	}

	// sincos //////////////////////////////////////////////////////////////////////////////////////
	{
		const size_t steps = 1 << 28;
		const size_t errbreak = 256;
		size_t errcount_sin = 0;
		size_t errcount_cos = 0;
		for (size_t i = 0; i < steps; ++i) {
			const float x = i / float(steps) * M_PI * 4 - M_PI * 2;
			const float asin = std::sin(x);
			const float acos = std::cos(x);
			f32x4 bsin, bcos;
			sincos(f32x4(x), bsin, bcos);

			if (!all(f32x4(bsin[0]) == bsin)) {
				if (errcount_sin < errbreak) {
					stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << bsin << '\n';
					success = false;

					if ((++errcount_sin & errcount_cos) == errbreak)
						break;
				}
			}

			if (!all(f32x4(bcos[0]) == bcos)) {
				if (errcount_cos < errbreak) {
					stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << bcos << '\n';
					success = false;

					if ((++errcount_cos & errcount_sin) == errbreak)
						break;
				}
			}

			const int32_t ulp_sin = diff_ulp(bsin[0], asin);
			if (-1 == ulp_sin || 14 < ulp_sin) {
				if (errcount_sin < errbreak) {
					stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " <<
						bsin[0] << " (" << asin << "), delta: " << formatter_f32(std::abs(bsin[0] - asin)) << " (" << formatter_f32(bsin[0]) << ", " << formatter_f32(asin) << ", " << ulp_sin << ")\n";
					success = false;

					if ((++errcount_sin & errcount_cos) == errbreak)
						break;
				}
			}

			const int32_t ulp_cos = diff_ulp(bcos[0], acos);
			if (-1 == ulp_cos || 14 < ulp_cos) {
				if (errcount_cos < errbreak) {
					stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " <<
						bcos[0] << " (" << acos << "), delta: " << formatter_f32(std::abs(bcos[0] - acos)) << " (" << formatter_f32(bcos[0]) << ", " << formatter_f32(acos) << ", " << ulp_cos << ")\n";
					success = false;

					if ((++errcount_cos & errcount_sin) == errbreak)
						break;
				}
			}

#if NATIVE_F32X8 != 0
			f32x8 csin, ccos;
			sincos(f32x8(x), csin, ccos);

			if (!all(f32x8(csin[0]) == csin)) {
				if (errcount_sin < errbreak) {
					stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << csin << '\n';
					success = false;

					if ((++errcount_sin & errcount_cos) == errbreak)
						break;
				}
			}

			if (!all(f32x8(ccos[0]) == ccos)) {
				if (errcount_cos < errbreak) {
					stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << ccos << '\n';
					success = false;

					if ((++errcount_cos & errcount_sin) == errbreak)
						break;
				}
			}

			if (!all(get_f32x4< 0 >(csin) == bsin)) {
				if (errcount_sin < errbreak) {
					stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << csin << '\n';
					success = false;

					if ((++errcount_sin & errcount_cos) == errbreak)
						break;
				}
			}

			if (!all(get_f32x4< 0 >(ccos) == bcos)) {
				if (errcount_cos < errbreak) {
					stream::cerr << "failed test at line " << stream::dec << __LINE__ << ", index " << i << stream::hex << ": " << ccos << '\n';
					success = false;

					if ((++errcount_cos & errcount_sin) == errbreak)
						break;
				}
			}

#endif
		}
	}

	return success ? 0 : -1;
}
