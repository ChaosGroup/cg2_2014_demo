#ifndef vectnative_H__
#define vectnative_H__

/// \file vectnative.hpp
/// A SIMD abstraction layer inspired by OpenCL-C/GLSL/CUDA vector types.
///
/// Copyright (c) 2017 Chaos Group
/// This software is provided under the MIT License; see LICENSE file for details.

#if _MSC_VER != 0
	#if defined(_M_AMD64) || defined(_M_X64)
		// x86-64 ISA extension recognition in MSVC is minimal - it knows only of SSE, SSE2, AVX and AVX2.
		// Hereby we recreate the ISA extension presence macros found in gcc-compatible compilers.
		#if __AVX__ != 0
			#define __SSE4_2__  1
		#endif
		#if __SSE4_2__ != 0
			#define __SSE4_1__  1
		#endif
		#if __SSE4_1__ != 0
			#define __SSSE3__   1
		#endif
		#if __SSSE3__ != 0
			#define __SSE3__    1
		#endif
		#define __SSE2__        1
		// The following header <intrin.h> aggregates all ISA extension headers up to AVX2, unconditionally of target architecture!
		// That header is also the sole provider of some SSE2-related intrisics missing from the standard SSE2 header <emmintrin.h>.
		#include <intrin.h>
	#elif defined(_M_ARM64)
		#define __ARM_NEON 1
	#endif
#endif

#if __AVX__ != 0
	#include <immintrin.h>
	#include "cephes_avx.h"
	#include "cephes_sse.h"
#elif __SSE4_2__ != 0
	#include <nmmintrin.h>
	#include "cephes_sse.h"
#elif __SSE4_1__ != 0
	#include <smmintrin.h>
	#include "cephes_sse.h"
#elif __SSSE3__ != 0
	#include <tmmintrin.h>
	#include "cephes_sse.h"
#elif __SSE3__ != 0
	#include <pmmintrin.h>
	#include "cephes_sse.h"
#elif __SSE2__ != 0
	#include <emmintrin.h>
	#include "cephes_sse.h"
#elif __ARM_NEON != 0
	#include <arm_neon.h>
	#include "cephes_neon.h"
#else
	#error SIMD required
#endif

#include <stdint.h>
#include <assert.h>

// note: this API tries to use generic vector types (i.e. user-definable vector types which are first-class citizens arithmetically- and accessor/mutator-wise) whenever possible,
// and as such, recognizes the following compiler quirks, providing respective workarounds:
//
// COMPILER_QUIRK_0000_ARITHMETIC_TYPE                   - the compiler does not support built-in simd types with arithmetic operators
// COMPILER_QUIRK_0001_SUBSCRIPT_IN_ARITHMETIC_TYPE      - the simd arithmetic type does not provide simple lane-access means
// COMPILER_QUIRK_0002_IMPLICIT_CAST_TO_ARITHMETIC_TYPE  - the compiler does not honor the cast operator to the simd arithmetic type in situations suitable for implicit casts
// COMPILER_QUIRK_0003_RELATIONAL_OPS                    - the simd arithmetic type does not provide relational ops
// COMPILER_QUIRK_0004_VMVNQ_U64                         - the arm neon header does not include an intrinsic for vmvnq_u64 (https://developer.arm.com/documentation/ihi0073/latest/)

#if _MSC_VER != 0
	#define COMPILER_QUIRK_0000_ARITHMETIC_TYPE                          1
	#define COMPILER_QUIRK_0004_VMVNQ_U64                                1
#else
	#if __clang__ == 0 && __GNUC__ != 0
		#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8)
			#define COMPILER_QUIRK_0001_SUBSCRIPT_IN_ARITHMETIC_TYPE     1
		#endif
		#define COMPILER_QUIRK_0002_IMPLICIT_CAST_TO_ARITHMETIC_TYPE     1
		#define COMPILER_QUIRK_0003_RELATIONAL_OPS                       1
	#endif
	#define COMPILER_QUIRK_0004_VMVNQ_U64                                1
#endif

namespace simd {
enum flag_zero {};      ///< formal type used to differentiate vector ctors that fill the missing lanes with zeors, as opposed to replicating the last supplied lane
enum flag_native {};    ///< formal type used to differentiate vector ctors which take a 'native' vector type, as opposed to taking a 'generic' vector type

// note: to save some KBytes from this source, while also improving the formatting, we use a terse-but-still-perfectly-intelligible naming of the scalar types; note that these aliases
// are not imposed on the user in any shape of form -- users of this API can pass whatever scalar types they see fit, leaving the mapping to the compiler.
typedef double   f64;
typedef int64_t  s64;
typedef uint64_t u64;
typedef float    f32;
typedef int32_t  s32;
typedef uint32_t u32;
typedef int16_t  s16;
typedef uint16_t u16;

template < bool >
struct compile_assert;

template <>
struct compile_assert< true > {
	compile_assert() {}
};

#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE != 0 || COMPILER_QUIRK_0001_SUBSCRIPT_IN_ARITHMETIC_TYPE != 0
/// Lane accessor for a generic vector type; an internal helper.
/// \param[in] n vector object of type VECTOR_T, semantically matching the type SCALAR_T[DIMENSION]
/// \param[in] idx lane of interest
/// \return value of the specified lane
template < size_t DIMENSION, typename SCALAR_T, typename VECTOR_T >
inline SCALAR_T getLane(const VECTOR_T n, const size_t idx) {
	const compile_assert< sizeof(SCALAR_T[DIMENSION]) == sizeof(VECTOR_T) > assert_size;

	assert(DIMENSION > idx);
	return reinterpret_cast< const SCALAR_T (&)[DIMENSION] >(n)[idx];
}

/// Lane mutator for a generic vector type; an internal helper.
/// \param[in,out] n vector object of type VECTOR_T, semantically matching the type SCALAR_T[DIMENSION]
/// \param[in]     idx lane of interest
/// \param[in]     c new value of the lane
template < size_t DIMENSION, typename SCALAR_T, typename VECTOR_T >
inline void setLane(VECTOR_T& n, const size_t idx, const SCALAR_T c) {
	const compile_assert< sizeof(SCALAR_T[DIMENSION]) == sizeof(VECTOR_T) > assert_size;

	assert(DIMENSION > idx);
	reinterpret_cast< SCALAR_T (&)[DIMENSION] >(n)[idx] = c;
}

#else
/// Lane accessor for a generic vector type; an internal helper.
/// \param[in] n vector object of type VECTOR_T, semantically matching the type SCALAR_T[DIMENSION]
/// \param[in] idx lane of interest
/// \return value of the specified lane
template < size_t DIMENSION, typename SCALAR_T, typename VECTOR_T >
inline SCALAR_T getLane(const VECTOR_T n, const size_t idx) {
	const compile_assert< sizeof(SCALAR_T[DIMENSION]) == sizeof(VECTOR_T) > assert_size;

	assert(DIMENSION > idx);
	return n[idx];
}

/// Lane mutator for a generic vector type; an internal helper.
/// \param[in,out] n vector object of type VECTOR_T, semantically matching the type SCALAR_T[DIMENSION]
/// \param[in]     idx lane of interest
/// \param[in]     c new value of the lane
template < size_t DIMENSION, typename SCALAR_T, typename VECTOR_T >
inline void setLane(VECTOR_T& n, const size_t idx, const SCALAR_T c) {
	const compile_assert< sizeof(SCALAR_T[DIMENSION]) == sizeof(VECTOR_T) > assert_size;

	assert(DIMENSION > idx);
	n[idx] = c;
}

#endif
/// Build a 2-lane vector from up to 2 scalars -- missing lanes are set to zero; an internal helper.
template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T vec2(
	const SCALAR_T c0,
	const SCALAR_T c1 = 0);

/// Build a 2-lane vector replicating a scalar across all lanes; an internal helper.
template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T splat2(
	const SCALAR_T c);

#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE == 0
template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T vec2(
	const SCALAR_T c0,
	const SCALAR_T c1) {
	const compile_assert< sizeof(SCALAR_T[2]) == sizeof(VECTOR_T) > assert_size;

	return (VECTOR_T){ c0, c1 };
}

template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T splat2(
	const SCALAR_T c) {
	const compile_assert< sizeof(SCALAR_T[2]) == sizeof(VECTOR_T) > assert_size;

	return (VECTOR_T){ c, c };
}

#else
// note: no generic vectors with this compiler - specialize vec2 for 'native' vector types
#if __SSE2__ != 0
template <>
inline __m128d vec2< __m128d, f64 >(
	const f64 c0,
	const f64 c1) {

	return _mm_set_pd(c1, c0);
}

template <>
inline __m128i vec2< __m128i, s64 >(
	const s64 c0,
	const s64 c1) {

	return _mm_set_epi64x(c1, c0);
}

template <>
inline __m128i vec2< __m128i, u64 >(
	const u64 c0,
	const u64 c1) {

	return _mm_set_epi64x(c1, c0);
}

template <>
inline __m128d splat2< __m128d, f64 >(
	const f64 c) {

	return _mm_set1_pd(c);
}

template <>
inline __m128i splat2< __m128i, s64 >(
	const s64 c) {

	return _mm_set1_epi64x(c);
}

template <>
inline __m128i splat2< __m128i, u64 >(
	const u64 c) {

	return _mm_set1_epi64x(c);
}

#elif __ARM_NEON != 0
template <>
inline float32x2_t vec2< float32x2_t, f32 >(
	const f32 c0,
	const f32 c1) {

	return float32x2_t{ .n64_f32 = { c0, c1 } };
}

template <>
inline int32x2_t vec2< int32x2_t, s32 >(
	const s32 c0,
	const s32 c1) {

	return int32x2_t{ .n64_i32 = { c0, c1 } };
}

template <>
inline uint32x2_t vec2< uint32x2_t, u32 >(
	const u32 c0,
	const u32 c1) {

	return uint32x2_t{ .n64_u32 = { c0, c1 } };
}

template <>
inline float32x2_t splat2< float32x2_t, f32 >(
	const f32 c) {

	return vdup_n_f32(c);
}

template <>
inline int32x2_t splat2< int32x2_t, s32 >(
	const s32 c) {

	return vdup_n_s32(c);
}

template <>
inline uint32x2_t splat2< uint32x2_t, u32 >(
	const u32 c) {

	return vdup_n_u32(c);
}

template <>
inline float64x2_t vec2< float64x2_t, f64 >(
	const f64 c0,
	const f64 c1) {

	return float64x2_t{ .n128_f64 = { c0, c1 } };
}

template <>
inline int64x2_t vec2< int64x2_t, s64 >(
	const s64 c0,
	const s64 c1) {

	return int64x2_t{ .n128_i64 = { c0, c1 } };
}

template <>
inline uint64x2_t vec2< uint64x2_t, u64 >(
	const u64 c0,
	const u64 c1) {

	return uint64x2_t{ .n128_u64 = { c0, c1 } };
}

template <>
inline float64x2_t splat2< float64x2_t, f64 >(
	const f64 c) {

	return vdupq_n_f64(c);
}

template <>
inline int64x2_t splat2< int64x2_t, s64 >(
	const s64 c) {

	return vdupq_n_s64(c);
}

template <>
inline uint64x2_t splat2< uint64x2_t, u64 >(
	const u64 c) {

	return vdupq_n_u64(c);
}

#endif
#endif
/// Build a 4-lane vector from up to 4 scalars -- missing lanes are set to zero; an internal helper.
template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T vec4(
	const SCALAR_T c0,
	const SCALAR_T c1 = 0,
	const SCALAR_T c2 = 0,
	const SCALAR_T c3 = 0);

/// Build a 4-lane vector replicating a scalar across all lanes; an internal helper.
template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T splat4(
	const SCALAR_T c);

#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE == 0
template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T vec4(
	const SCALAR_T c0,
	const SCALAR_T c1,
	const SCALAR_T c2,
	const SCALAR_T c3) {
	const compile_assert< sizeof(SCALAR_T[4]) == sizeof(VECTOR_T) > assert_size;

	return (VECTOR_T){ c0, c1, c2, c3 };
}

template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T splat4(
	const SCALAR_T c) {
	const compile_assert< sizeof(SCALAR_T[4]) == sizeof(VECTOR_T) > assert_size;

	return (VECTOR_T){ c, c, c, c };
}

#else
// note: no generic vectors with this compiler - specialize vec4 for 'native' vector types
#if __AVX__ != 0
template <>
inline __m256d vec4< __m256d, f64 >(
	const f64 c0,
	const f64 c1,
	const f64 c2,
	const f64 c3) {

	return _mm256_set_pd(c3, c2, c1, c0);
}

template <>
inline __m256i vec4< __m256i, s64 >(
	const s64 c0,
	const s64 c1,
	const s64 c2,
	const s64 c3) {

	return _mm256_set_epi64x(c3, c2, c1, c0);
}

template <>
inline __m256i vec4< __m256i, u64 >(
	const u64 c0,
	const u64 c1,
	const u64 c2,
	const u64 c3) {

	return _mm256_set_epi64x(c3, c2, c1, c0);
}

template <>
inline __m256d splat4< __m256d, f64 >(
	const f64 c) {

	return _mm256_set1_pd(c);
}

template <>
inline __m256i splat4< __m256i, s64 >(
	const s64 c) {

	return _mm256_set1_epi64x(c);
}

template <>
inline __m256i splat4< __m256i, u64 >(
	const u64 c) {

	return _mm256_set1_epi64x(c);
}

#endif
#if __SSE2__ != 0
template <>
inline __m128 vec4< __m128, f32 >(
	const f32 c0,
	const f32 c1,
	const f32 c2,
	const f32 c3) {

	return _mm_set_ps(c3, c2, c1, c0);
}

template <>
inline __m128i vec4< __m128i, s32 >(
	const s32 c0,
	const s32 c1,
	const s32 c2,
	const s32 c3) {

	return _mm_set_epi32(c3, c2, c1, c0);
}

template <>
inline __m128i vec4< __m128i, u32 >(
	const u32 c0,
	const u32 c1,
	const u32 c2,
	const u32 c3) {

	return _mm_set_epi32(c3, c2, c1, c0);
}

template <>
inline __m128 splat4< __m128, f32 >(
	const f32 c) {

	return _mm_set1_ps(c);
}

template <>
inline __m128i splat4< __m128i, s32 >(
	const s32 c) {

	return _mm_set1_epi32(c);
}

template <>
inline __m128i splat4< __m128i, u32 >(
	const u32 c) {

	return _mm_set1_epi32(c);
}

#elif __ARM_NEON != 0
template <>
inline float32x4_t vec4< float32x4_t, f32 >(
	const f32 c0,
	const f32 c1,
	const f32 c2,
	const f32 c3) {

	return float32x4_t{ .n128_f32 = { c0, c1, c2, c3 } };
}

template <>
inline int32x4_t vec4< int32x4_t, s32 >(
	const s32 c0,
	const s32 c1,
	const s32 c2,
	const s32 c3) {

	return int32x4_t{ .n128_i32 = { c0, c1, c2, c3 } };
}

template <>
inline uint32x4_t vec4< uint32x4_t, u32 >(
	const u32 c0,
	const u32 c1,
	const u32 c2,
	const u32 c3) {

	return uint32x4_t{ .n128_u32 = { c0, c1, c2, c3 } };
}

template <>
inline int16x4_t vec4< int16x4_t, s16 >(
	const s16 c0,
	const s16 c1,
	const s16 c2,
	const s16 c3) {

	return int16x4_t{ .n64_i16 = { c0, c1, c2, c3 } };
}

template <>
inline uint16x4_t vec4< uint16x4_t, u16 >(
	const u16 c0,
	const u16 c1,
	const u16 c2,
	const u16 c3) {

	return uint16x4_t{ .n64_u16 = { c0, c1, c2, c3 } };
}

template <>
inline float32x4_t splat4< float32x4_t, f32 >(
	const f32 c) {

	return vdupq_n_f32(c);
}

template <>
inline int32x4_t splat4< int32x4_t, s32 >(
	const s32 c) {

	return vdupq_n_s32(c);
}

template <>
inline uint32x4_t splat4< uint32x4_t, u32 >(
	const u32 c) {

	return vdupq_n_u32(c);
}

template <>
inline int16x4_t splat4< int16x4_t, s16 >(
	const s16 c) {

	return vdup_n_s16(c);
}

template <>
inline uint16x4_t splat4< uint16x4_t, u16 >(
	const u16 c) {

	return vdup_n_u16(c);
}

#endif
#endif
/// Build a 8-lane vector from up to 8 scalars -- missing lanes are set to zero; an internal helper.
template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T vec8(
	const SCALAR_T c0,
	const SCALAR_T c1 = 0,
	const SCALAR_T c2 = 0,
	const SCALAR_T c3 = 0,
	const SCALAR_T c4 = 0,
	const SCALAR_T c5 = 0,
	const SCALAR_T c6 = 0,
	const SCALAR_T c7 = 0);

/// Build a 8-lane vector replicating a scalar across all lanes; an internal helper.
template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T splat8(
	const SCALAR_T c);

#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE == 0
template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T vec8(
	const SCALAR_T c0,
	const SCALAR_T c1,
	const SCALAR_T c2,
	const SCALAR_T c3,
	const SCALAR_T c4,
	const SCALAR_T c5,
	const SCALAR_T c6,
	const SCALAR_T c7) {
	const compile_assert< sizeof(SCALAR_T[8]) == sizeof(VECTOR_T) > assert_size;

	return (VECTOR_T){ c0, c1, c2, c3, c4, c5, c6, c7 };
}

template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T splat8(
	const SCALAR_T c) {
	const compile_assert< sizeof(SCALAR_T[8]) == sizeof(VECTOR_T) > assert_size;

	return (VECTOR_T){ c, c, c, c, c, c, c, c };
}

#else
// note: no generic vectors with this compiler - specialize vec8 for 'native' vector types
#if __AVX__ != 0
template <>
inline __m256 vec8< __m256, f32 >(
	const f32 c0,
	const f32 c1,
	const f32 c2,
	const f32 c3,
	const f32 c4,
	const f32 c5,
	const f32 c6,
	const f32 c7) {

	return _mm256_set_ps(c7, c6, c5, c4, c3, c2, c1, c0);
}

template <>
inline __m256i vec8< __m256i, s32 >(
	const s32 c0,
	const s32 c1,
	const s32 c2,
	const s32 c3,
	const s32 c4,
	const s32 c5,
	const s32 c6,
	const s32 c7) {

	return _mm256_set_epi32(c7, c6, c5, c4, c3, c2, c1, c0);
}

template <>
inline __m256i vec8< __m256i, u32 >(
	const u32 c0,
	const u32 c1,
	const u32 c2,
	const u32 c3,
	const u32 c4,
	const u32 c5,
	const u32 c6,
	const u32 c7) {

	return _mm256_set_epi32(c7, c6, c5, c4, c3, c2, c1, c0);
}

template <>
inline __m256 splat8< __m256, f32 >(
	const f32 c) {

	return _mm256_set1_ps(c);
}

template <>
inline __m256i splat8< __m256i, s32 >(
	const s32 c) {

	return _mm256_set1_epi32(c);
}

template <>
inline __m256i splat8< __m256i, u32 >(
	const u32 c) {

	return _mm256_set1_epi32(c);
}

#endif
#if __SSE2__ != 0
template <>
inline __m128i vec8< __m128i, s16 >(
	const s16 c0,
	const s16 c1,
	const s16 c2,
	const s16 c3,
	const s16 c4,
	const s16 c5,
	const s16 c6,
	const s16 c7) {

	return _mm_set_epi16(c7, c6, c5, c4, c3, c2, c1, c0);
}

template <>
inline __m128i vec8< __m128i, u16 >(
	const u16 c0,
	const u16 c1,
	const u16 c2,
	const u16 c3,
	const u16 c4,
	const u16 c5,
	const u16 c6,
	const u16 c7) {

	return _mm_set_epi16(c7, c6, c5, c4, c3, c2, c1, c0);
}

template <>
inline __m128i splat8< __m128i, s16 >(
	const s16 c) {

	return _mm_set1_epi16(c);
}

template <>
inline __m128i splat8< __m128i, u16 >(
	const u16 c) {

	return _mm_set1_epi16(c);
}

#elif __ARM_NEON != 0
template <>
inline int16x8_t vec8< int16x8_t, s16 >(
	const s16 c0,
	const s16 c1,
	const s16 c2,
	const s16 c3,
	const s16 c4,
	const s16 c5,
	const s16 c6,
	const s16 c7) {

	return int16x8_t{ .n128_i16 = { c0, c1, c2, c3, c4, c5, c6, c7 } };
}

template <>
inline uint16x8_t vec8< uint16x8_t, u16 >(
	const u16 c0,
	const u16 c1,
	const u16 c2,
	const u16 c3,
	const u16 c4,
	const u16 c5,
	const u16 c6,
	const u16 c7) {

	return uint16x8_t{ .n128_u16 = { c0, c1, c2, c3, c4, c5, c6, c7 } };
}

template <>
inline int16x8_t splat8< int16x8_t, s16 >(
	const s16 c) {

	return vdupq_n_s16(c);
}

template <>
inline uint16x8_t splat8< uint16x8_t, u16 >(
	const u16 c) {

	return vdupq_n_u16(c);
}

#endif
#endif
/// Build a 16-lane vector from up to 16 scalars -- missing lanes are set to zero; an internal helper.
template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T vec16(
	const SCALAR_T c0,
	const SCALAR_T c1  = 0,
	const SCALAR_T c2  = 0,
	const SCALAR_T c3  = 0,
	const SCALAR_T c4  = 0,
	const SCALAR_T c5  = 0,
	const SCALAR_T c6  = 0,
	const SCALAR_T c7  = 0,
	const SCALAR_T c8  = 0,
	const SCALAR_T c9  = 0,
	const SCALAR_T c10 = 0,
	const SCALAR_T c11 = 0,
	const SCALAR_T c12 = 0,
	const SCALAR_T c13 = 0,
	const SCALAR_T c14 = 0,
	const SCALAR_T c15 = 0);

/// Build a 16-lane vector replicating a scalar across all lanes; an internal helper.
template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T splat16(
	const SCALAR_T c);

#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE == 0
template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T vec16(
	const SCALAR_T c0,
	const SCALAR_T c1,
	const SCALAR_T c2,
	const SCALAR_T c3,
	const SCALAR_T c4,
	const SCALAR_T c5,
	const SCALAR_T c6,
	const SCALAR_T c7,
	const SCALAR_T c8,
	const SCALAR_T c9,
	const SCALAR_T c10,
	const SCALAR_T c11,
	const SCALAR_T c12,
	const SCALAR_T c13,
	const SCALAR_T c14,
	const SCALAR_T c15) {
	const compile_assert< sizeof(SCALAR_T[16]) == sizeof(VECTOR_T) > assert_size;

	return (VECTOR_T){ c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 };
}

template < typename VECTOR_T, typename SCALAR_T >
inline VECTOR_T splat16(
	const SCALAR_T c) {
	const compile_assert< sizeof(SCALAR_T[16]) == sizeof(VECTOR_T) > assert_size;

	return (VECTOR_T){ c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c };
}

#else
// note: no generic vectors with this compiler - specialize vec16 for 'native' vector types
#if __AVX__ != 0
template <>
inline __m256i vec16< __m256i, s16 >(
	const s16 c0,
	const s16 c1,
	const s16 c2,
	const s16 c3,
	const s16 c4,
	const s16 c5,
	const s16 c6,
	const s16 c7,
	const s16 c8,
	const s16 c9,
	const s16 c10,
	const s16 c11,
	const s16 c12,
	const s16 c13,
	const s16 c14,
	const s16 c15) {

	return _mm256_set_epi16(c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0);
}

template <>
inline __m256i vec16< __m256i, u16 >(
	const u16 c0,
	const u16 c1,
	const u16 c2,
	const u16 c3,
	const u16 c4,
	const u16 c5,
	const u16 c6,
	const u16 c7,
	const u16 c8,
	const u16 c9,
	const u16 c10,
	const u16 c11,
	const u16 c12,
	const u16 c13,
	const u16 c14,
	const u16 c15) {

	return _mm256_set_epi16(c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0);
}

template <>
inline __m256i splat16< __m256i, s16 >(
	const s16 c) {

	return _mm256_set1_epi16(c);
}

template <>
inline __m256i splat16< __m256i, u16 >(
	const u16 c) {

	return _mm256_set1_epi16(c);
}

#endif
#endif
/// Deinterleave the even-position (0, 2, 4, ..) bits from a 16-bit argument;
/// odd-position bits get discarded; e.g. from bit pattern xaxbxcxd -> 0000abcd
inline uint16_t deinterleaveBits8(const uint16_t bits) {
	uint16_t x = bits & 0x5555;
	x = (x | (x >> 1)) & 0x3333;
	x = (x | (x >> 2)) & 0x0f0f;
	x = (x | (x >> 4)) & 0x00ff;
	return x;
}

/// Deinterleave the even-position (0, 2, 4, ..) bits from a 32-bit argument
/// odd-position bits get discarded; e.g. from bit pattern xaxbxcxd -> 0000abcd
inline uint32_t deinterleaveBits16(const uint32_t bits) {
	uint32_t x = bits & 0x55555555;
	x = (x | (x >> 1)) & 0x33333333;
	x = (x | (x >> 2)) & 0x0f0f0f0f;
	x = (x | (x >> 4)) & 0x00ff00ff;
	x = (x | (x >> 8)) & 0x0000ffff;
	return x;
}

/// Interleave two 8-bit arguments into one 16-bit result; first argument
/// occupies the even bit positions in result, second - the odd bit positions;
/// e.g. from bit patterns abcd, ABCD -> AaBbCcDd
inline uint16_t interleaveBits8(const uint8_t a, const uint8_t b) {
	uint32_t x = uint32_t(a) | uint32_t(b) << 16;
	x = (x | (x << 4)) & 0x0f0f0f0f;
	x = (x | (x << 2)) & 0x33333333;
	x = (x | (x << 1)) & 0x55555555;
	return uint16_t(x) | uint16_t(x >> 15);
}

// note: in the following templates, we cannot define a generic vector type based on a template type parameter
// (clang error: 'vector_size' attribute requires an integer constant) so instead we supply the entire generic
// vector type as a template type parameter.

/// Vector type of 2 lanes, parametrizable by a scalar type and two vector types -- an underlying 'generic' vector type,
/// and a secondary vector type dubbed 'native' that we can construct from and cast to.
template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
class native2 {
public:
	enum { dimension = 2 };     ///< dimension of this vector type
	typedef SCALAR_T scalar;    ///< scalar type of the elements of this vector type
	typedef GENERIC_T generic;  ///< underlying 'generic' vectory type
	typedef NATIVE_T native;    ///< secondary 'native' vector type this type can be constructed from and cast to

private:
	generic m;

public:
	native2() {}

	/// Construct from 'native' vector.
	native2(const native src, const flag_native)
	: m(reinterpret_cast< const generic& >(src)) {
		const compile_assert< sizeof(native) == sizeof(scalar[dimension]) > assert_size_native;
	}

	/// Construct from 'generic' vector; intentionally non-explicit.
	native2(const generic src)
	: m(src) {
		const compile_assert< sizeof(generic) == sizeof(scalar[dimension]) > assert_size_generic;
	}

#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE == 0
	/// Cast this vector to the underlying 'generic' vector type.
	operator generic() const {
		return m;
	}

#endif
	/// Immutably bitcast this vector to a 'native' vector type.
	typename native2::native getn() const {
		return reinterpret_cast< const native& >(m);
	}

	/// Construct from 2 scalars.
	native2(
		const scalar c0,
		const scalar c1)
	: m(vec2< generic >(c0, c1)) {
	}

	/// Construct from one scalar, replicating that across all lanes.
	explicit native2(
		const scalar c)
	: m(splat2< generic >(c)) {
	}

	/// Construct from 1 scalar, filling the remaining lanes with zeroes.
	native2(
		const scalar c0,
		const flag_zero)
	: m(vec2< generic >(c0)) {
	}

	/// Get the value of the specified lane of this vector.
	typename native2::scalar get(const size_t idx) const {
		return getLane< dimension, scalar >(m, idx);
	}

	/// Set the specified lane of this vector to the specified value.
	native2& set(const size_t idx, const scalar c) {
		setLane< dimension >(m, idx, c);
		return *this;
	}

	/// Get the value of the specified lane of this vector, subscript version.
	typename native2::scalar operator[](const size_t idx) const {
		return get(idx);
	}
};

/// Vector type of 4 lanes, parametrizable by a scalar type and two vector types -- an underlying 'generic' vector type,
/// and a secondary vector type dubbed 'native' that we can construct from and cast to.
template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
class native4 {
public:
	enum { dimension = 4 };     ///< dimension of this vector type
	typedef SCALAR_T scalar;    ///< scalar type of the elements of this vector type
	typedef GENERIC_T generic;  ///< underlying 'generic' vectory type
	typedef NATIVE_T native;    ///< secondary 'native' vector type this type can be constructed from and cast to

private:
	generic m;

public:
	native4() {}

	/// Construct from 'native' vector.
	native4(const native src, const flag_native)
	: m(reinterpret_cast< const generic& >(src)) {
		const compile_assert< sizeof(native) == sizeof(scalar[dimension]) > assert_size_native;
	}

	/// Construct from 'generic' vector; intentionally non-explicit.
	native4(const generic src)
	: m(src) {
		const compile_assert< sizeof(generic) == sizeof(scalar[dimension]) > assert_size_generic;
	}

#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE == 0
	/// Cast this vector to the underlying 'generic' vector type.
	operator generic() const {
		return m;
	}

#endif
	/// Immutably bitcast this vector to a 'native' vector type.
	typename native4::native getn() const {
		return reinterpret_cast< const native& >(m);
	}

	/// Construct from 4 scalars.
	native4(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3)
	: m(vec4< generic >(c0, c1, c2, c3)) {
	}

	/// Construct from 3 scalars, replicating the 3rd across the remaining lanes.
	native4(
		const scalar c0,
		const scalar c1,
		const scalar c2)
	: m(vec4< generic >(c0, c1, c2, c2)) {
	}

	/// Construct from 2 scalars, replicating the 2nd across the remaining lanes.
	native4(
		const scalar c0,
		const scalar c1)
	: m(vec4< generic >(c0, c1, c1, c1)) {
	}

	/// Construct from one scalar, replicating that across all lanes.
	explicit native4(
		const scalar c)
	: m(splat4< generic >(c)) {
	}

	/// Construct from 1 scalar, filling the remaining lanes with zeroes.
	native4(
		const scalar c0,
		const flag_zero)
	: m(vec4< generic >(c0)) {
	}

	/// Construct from 2 scalars, filling the remaining lanes with zeroes.
	native4(
		const scalar c0,
		const scalar c1,
		const flag_zero)
	: m(vec4< generic >(c0, c1)) {
	}

	/// Construct from 3 scalars, filling the remaining lanes with zeroes.
	native4(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const flag_zero)
	: m(vec4< generic >(c0, c1, c2)) {
	}

	/// Get the value of the specified lane of this vector.
	typename native4::scalar get(const size_t idx) const {
		return getLane< dimension, scalar >(m, idx);
	}

	/// Set the specified lane of this vector to the specified value.
	native4& set(const size_t idx, const scalar c) {
		setLane< dimension >(m, idx, c);
		return *this;
	}

	/// Get the value of the specified lane of this vector, subscript version.
	typename native4::scalar operator[](const size_t idx) const {
		return get(idx);
	}
};

/// Vector type of 8 lanes, parametrizable by a scalar type and two vector types -- an underlying 'generic' vector type,
/// and a secondary vector type dubbed 'native' that we can construct from and cast to.
template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
class native8 {
public:
	enum { dimension = 8 };     ///< dimension of this vector type
	typedef SCALAR_T scalar;    ///< scalar type of the elements of this vector type
	typedef GENERIC_T generic;  ///< underlying 'generic' vectory type
	typedef NATIVE_T native;    ///< secondary 'native' vector type this type can be constructed from and cast to

private:
	generic m;

public:
	native8() {}

	/// Construct from 'native' vector.
	native8(const native src, const flag_native)
	: m(reinterpret_cast< const generic& >(src)) {
		const compile_assert< sizeof(native) == sizeof(scalar[dimension]) > assert_size_native;
	}

	/// Construct from 'generic' vector; intentionally non-explicit.
	native8(const generic src)
	: m(src) {
		const compile_assert< sizeof(generic) == sizeof(scalar[dimension]) > assert_size_generic;
	}

#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE == 0
	/// Cast this vector to the underlying 'generic' vector type.
	operator generic() const {
		return m;
	}

#endif
	/// Immutably bitcast this vector to a 'native' vector type.
	typename native8::native getn() const {
		return reinterpret_cast< const native& >(m);
	}

	/// Construct from 8 scalars.
	native8(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7)
	: m(vec8< generic >(c0, c1, c2, c3, c4, c5, c6, c7)) {
	}

	/// Construct from 7 scalars, replicating the 7th across the remaining lanes.
	native8(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6)
	: m(vec8< generic >(c0, c1, c2, c3, c4, c5, c6, c6)) {
	}

	/// Construct from 6 scalars, replicating the 6th across the remaining lanes.
	native8(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5)
	: m(vec8< generic >(c0, c1, c2, c3, c4, c5, c5, c5)) {
	}

	/// Construct from 5 scalars, replicating the 5th across the remaining lanes.
	native8(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4)
	: m(vec8< generic >(c0, c1, c2, c3, c4, c4, c4, c4)) {
	}

	/// Construct from 4 scalars, replicating the 4th across the remaining lanes.
	native8(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3)
	: m(vec8< generic >(c0, c1, c2, c3, c3, c3, c3, c3)) {
	}

	/// Construct from 3 scalars, replicating the 3rd across the remaining lanes.
	native8(
		const scalar c0,
		const scalar c1,
		const scalar c2)
	: m(vec8< generic >(c0, c1, c2, c2, c2, c2, c2, c2)) {
	}

	/// Construct from 2 scalars, replicating the 2nd across the remaining lanes.
	native8(
		const scalar c0,
		const scalar c1)
	: m(vec8< generic >(c0, c1, c1, c1, c1, c1, c1, c1)) {
	}

	/// Construct from one scalar, replicating that across all lanes.
	explicit native8(
		const scalar c)
	: m(splat8< generic >(c)) {
	}

	/// Construct from 1 scalar, filling the remaining lanes with zeroes.
	native8(
		const scalar c0,
		const flag_zero)
	: m(vec8< generic >(c0)) {
	}

	/// Construct from 2 scalars, filling the remaining lanes with zeroes.
	native8(
		const scalar c0,
		const scalar c1,
		const flag_zero)
	: m(vec8< generic >(c0, c1)) {
	}

	/// Construct from 3 scalars, filling the remaining lanes with zeroes.
	native8(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const flag_zero)
	: m(vec8< generic >(c0, c1, c2)) {
	}

	/// Construct from 4 scalars, filling the remaining lanes with zeroes.
	native8(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const flag_zero)
	: m(vec8< generic >(c0, c1, c2, c3)) {
	}

	/// Construct from 5 scalars, filling the remaining lanes with zeroes.
	native8(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const flag_zero)
	: m(vec8< generic >(c0, c1, c2, c3, c4)) {
	}

	/// Construct from 6 scalars, filling the remaining lanes with zeroes.
	native8(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const flag_zero)
	: m(vec8< generic >(c0, c1, c2, c3, c4, c5)) {
	}

	/// Construct from 7 scalars, filling the remaining lanes with zeroes.
	native8(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const flag_zero)
	: m(vec8< generic >(c0, c1, c2, c3, c4, c5, c6)) {
	}

	/// Get the value of the specified lane of this vector.
	typename native8::scalar get(const size_t idx) const {
		return getLane< dimension, scalar >(m, idx);
	}

	/// Set the specified lane of this vector to the specified value.
	native8& set(const size_t idx, const scalar c) {
		setLane< dimension >(m, idx, c);
		return *this;
	}

	/// Get the value of the specified lane of this vector, subscript version.
	typename native8::scalar operator[](const size_t idx) const {
		return get(idx);
	}
};

/// Vector type of 16 lanes, parametrizable by a scalar type and two vector types -- an underlying 'generic' vector type,
/// and a secondary vector type dubbed 'native' that we can construct from and cast to.
template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
class native16 {
public:
	enum { dimension = 16 };    ///< dimension of this vector type
	typedef SCALAR_T scalar;    ///< scalar type of the elements of this vector type
	typedef GENERIC_T generic;  ///< underlying 'generic' vectory type
	typedef NATIVE_T native;    ///< secondary 'native' vector type this type can be constructed from and cast to

private:
	generic m;

public:
	native16() {}

	/// Construct from 'native' vector.
	native16(const native src, const flag_native)
	: m(reinterpret_cast< const generic& >(src)) {
		const compile_assert< sizeof(native) == sizeof(scalar[dimension]) > assert_size_native;
	}

	/// Construct from 'generic' vector; intentionally non-explicit.
	native16(const generic src)
	: m(src) {
		const compile_assert< sizeof(generic) == sizeof(scalar[dimension]) > assert_size_generic;
	}

#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE == 0
	/// Cast this vector to the underlying 'generic' vector type.
	operator generic() const {
		return m;
	}

#endif
	/// Immutably bitcast this vector to a 'native' vector type.
	typename native16::native getn() const {
		return reinterpret_cast< const native& >(m);
	}

	/// Construct from 16 scalars.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9,
		const scalar c10,
		const scalar c11,
		const scalar c12,
		const scalar c13,
		const scalar c14,
		const scalar c15)
	:m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15)) {
	}

	/// Construct from 15 scalars, replicating the 15th across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9,
		const scalar c10,
		const scalar c11,
		const scalar c12,
		const scalar c13,
		const scalar c14)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c14)) {
	}

	/// Construct from 14 scalars, replicating the 14th across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9,
		const scalar c10,
		const scalar c11,
		const scalar c12,
		const scalar c13)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c13, c13)) {
	}

	/// Construct from 13 scalars, replicating the 13th across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9,
		const scalar c10,
		const scalar c11,
		const scalar c12)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c12, c12, c12)) {
	}

	/// Construct from 12 scalars, replicating the 12th across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9,
		const scalar c10,
		const scalar c11)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c11, c11, c11, c11)) {
	}

	/// Construct from 11 scalars, replicating the 11th across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9,
		const scalar c10)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c10, c10, c10, c10, c10)) {
	}

	/// Construct from 10 scalars, replicating the 10th across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c9, c9, c9, c9, c9, c9)) {
	}

	/// Construct from 9 scalars, replicating the 9th across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c8, c8, c8, c8, c8, c8, c8)) {
	}

	/// Construct from 8 scalars, replicating the 8th across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c7, c7, c7, c7, c7, c7, c7, c7)) {
	}

	/// Construct from 7 scalars, replicating the 7th across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c6, c6, c6, c6, c6, c6, c6, c6, c6)) {
	}

	/// Construct from 6 scalars, replicating the 6th across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c5, c5, c5, c5, c5, c5, c5, c5, c5, c5)) {
	}

	/// Construct from 5 scalars, replicating the 5th across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c4, c4, c4, c4, c4, c4, c4, c4, c4, c4, c4)) {
	}

	/// Construct from 4 scalars, replicating the 4th across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3)
	: m(vec16< generic >(c0, c1, c2, c3, c3, c3, c3, c3, c3, c3, c3, c3, c3, c3, c3, c3)) {
	}

	/// Construct from 3 scalars, replicating the 3rd across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2)
	: m(vec16< generic >(c0, c1, c2, c2, c2, c2, c2, c2, c2, c2, c2, c2, c2, c2, c2, c2)) {
	}

	/// Construct from 2 scalars, replicating the 2nd across the remaining lanes.
	native16(
		const scalar c0,
		const scalar c1)
	: m(vec16< generic >(c0, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1)) {
	}

	/// Construct from one scalar, replicating that across all lanes.
	explicit native16(
		const scalar c)
	: m(splat16< generic >(c)) {
	}

	/// Construct from 1 scalar, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const flag_zero)
	: m(vec16< generic >(c0)) {
	}

	/// Construct from 2 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const flag_zero)
	: m(vec16< generic >(c0, c1)) {
	}

	/// Construct from 3 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2)) {
	}

	/// Construct from 4 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2, c3)) {
	}

	/// Construct from 5 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2, c3, c4)) {
	}

	/// Construct from 6 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5)) {
	}

	/// Construct from 7 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6)) {
	}

	/// Construct from 8 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7)) {
	}

	/// Construct from 9 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8)) {
	}

	/// Construct from 10 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)) {
	}

	/// Construct from 11 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9,
		const scalar c10,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10)) {
	}

	/// Construct from 12 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9,
		const scalar c10,
		const scalar c11,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)) {
	}

	/// Construct from 13 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9,
		const scalar c10,
		const scalar c11,
		const scalar c12,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12)) {
	}

	/// Construct from 14 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9,
		const scalar c10,
		const scalar c11,
		const scalar c12,
		const scalar c13,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13)) {
	}

	/// Construct from 15 scalars, filling the remaining lanes with zeroes.
	native16(
		const scalar c0,
		const scalar c1,
		const scalar c2,
		const scalar c3,
		const scalar c4,
		const scalar c5,
		const scalar c6,
		const scalar c7,
		const scalar c8,
		const scalar c9,
		const scalar c10,
		const scalar c11,
		const scalar c12,
		const scalar c13,
		const scalar c14,
		const flag_zero)
	: m(vec16< generic >(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14)) {
	}

	/// Get the value of the specified lane of this vector.
	typename native16::scalar get(const size_t idx) const {
		return getLane< dimension, scalar >(m, idx);
	}

	/// Set the specified lane of this vector to the specified value.
	native16& set(const size_t idx, const scalar c) {
		setLane< dimension >(m, idx, c);
		return *this;
	}

	/// Get the value of the specified lane of this vector, subscript version.
	typename native16::scalar operator[](const size_t idx) const {
		return get(idx);
	}
};

#if __SSE2__ != 0
#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE != 0
	// note: no generic vectors with this compiler - use the 'native' vector type as the underlying vector type
	typedef native2< f64, __m128d, __m128d > f64x2;
	typedef native2< s64, __m128i, __m128i > s64x2;
	typedef native2< u64, __m128i, __m128i > u64x2;
	typedef native4< f32, __m128,  __m128  > f32x4;
	typedef native4< s32, __m128i, __m128i > s32x4;
	typedef native4< u32, __m128i, __m128i > u32x4;
	typedef native8< s16, __m128i, __m128i > s16x8;
	typedef native8< u16, __m128i, __m128i > u16x8;

#else
	typedef native2< f64, f64 __attribute__ ((vector_size(2 * sizeof(f64)))), __m128d > f64x2;
	typedef native2< s64, s64 __attribute__ ((vector_size(2 * sizeof(s64)))), __m128i > s64x2;
	typedef native2< u64, u64 __attribute__ ((vector_size(2 * sizeof(u64)))), __m128i > u64x2;
	typedef native4< f32, f32 __attribute__ ((vector_size(4 * sizeof(f32)))), __m128  > f32x4;
	typedef native4< s32, s32 __attribute__ ((vector_size(4 * sizeof(s32)))), __m128i > s32x4;
	typedef native4< u32, u32 __attribute__ ((vector_size(4 * sizeof(u32)))), __m128i > u32x4;
	typedef native8< s16, s16 __attribute__ ((vector_size(8 * sizeof(s16)))), __m128i > s16x8;
	typedef native8< u16, u16 __attribute__ ((vector_size(8 * sizeof(u16)))), __m128i > u16x8;

#endif
	#define NATIVE_F64X2    1
	#define NATIVE_S64X2    1
	#define NATIVE_U64X2    1
	#define NATIVE_F32X4    1
	#define NATIVE_S32X4    1
	#define NATIVE_U32X4    1
	#define NATIVE_S16X8    1
	#define NATIVE_U16X8    1

	// bitcasts ////////////////////////////////////////////////////////////////////////////////////

	/// Re-interpret the argument as the target type, preserving the original lane count and width.
	/// \param x argument of some source type.
	/// \return the re-interpretation of the argument as the return type
	inline f64x2 as_f64x2(const s64x2 x) {
		return f64x2(_mm_castsi128_pd(x.getn()), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline f64x2 as_f64x2(const u64x2 x) {
		return f64x2(_mm_castsi128_pd(x.getn()), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline s64x2 as_s64x2(const f64x2 x) {
		return s64x2(_mm_castpd_si128(x.getn()), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline s64x2 as_s64x2(const u64x2 x) {
		return s64x2(x.getn(), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline u64x2 as_u64x2(const f64x2 x) {
		return u64x2(_mm_castpd_si128(x.getn()), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline u64x2 as_u64x2(const s64x2 x) {
		return u64x2(x.getn(), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline f32x4 as_f32x4(const s32x4 x) {
		return f32x4(_mm_castsi128_ps(x.getn()), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline f32x4 as_f32x4(const u32x4 x) {
		return f32x4(_mm_castsi128_ps(x.getn()), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline s32x4 as_s32x4(const f32x4 x) {
		return s32x4(_mm_castps_si128(x.getn()), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline s32x4 as_s32x4(const u32x4 x) {
		return s32x4(x.getn(), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline u32x4 as_u32x4(const f32x4 x) {
		return u32x4(_mm_castps_si128(x.getn()), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline u32x4 as_u32x4(const s32x4 x) {
		return u32x4(x.getn(), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline s16x8 as_s16x8(const u16x8 x) {
		return s16x8(x.getn(), flag_native());
	}

	/// \copydoc as_f64x2(const s64x2)
	inline u16x8 as_u16x8(const s16x8 x) {
		return u16x8(x.getn(), flag_native());
	}

	// reduction predicates ////////////////////////////////////////////////////////////////////////

	/// All-reduction. True if all masked lanes of the specified integral-type vector are set to -1.
	/// \param lane_mask integral-type vector containing only 0's and -1's.
	/// \param bit_submask bitmask of the lanes of interest; a set bit indicates interest in the lane
	/// of the same index as the bit.
	/// \return true if all lanes, specified in the submask, are set to -1, false otherwise.
	inline bool all(const u64x2 lane_mask, const uint64_t bit_submask = 3) {
		assert(bit_submask > 0 && bit_submask < 4);

		return bit_submask == (bit_submask & _mm_movemask_pd(_mm_castsi128_pd(lane_mask.getn())));
	}

	/// \copydoc all(const u64x2, const uint64_t)
	inline bool all(const u32x4 lane_mask, const uint32_t bit_submask = 15) {
		assert(bit_submask > 0 && bit_submask < 16);

		return bit_submask == (bit_submask & _mm_movemask_ps(_mm_castsi128_ps(lane_mask.getn())));
	}

	/// \copydoc all(const u64x2, const uint64_t)
	inline bool all(const u16x8 lane_mask, const uint16_t bit_submask = 255) {
		assert(bit_submask > 0 && bit_submask < 256);

		return bit_submask == (bit_submask & deinterleaveBits8(_mm_movemask_epi8(lane_mask.getn())));
	}

	/// Any-reduction. True if any of the masked lanes of the specified integral-type vector are set to -1.
	/// \param lane_mask integral-type vector containing only 0's and -1's.
	/// \param bit_submask bitmask of the lanes of interest; a set bit indicates interest in the lane
	/// of the same index as the bit.
	/// \return true if any of the lanes, specified in the submask, are set to -1, false otherwise.
	inline bool any(const u64x2 lane_mask, const uint64_t bit_submask = 3) {
		assert(bit_submask > 0 && bit_submask < 4);

		return 0 != (bit_submask & _mm_movemask_pd(_mm_castsi128_pd(lane_mask.getn())));
	}

	/// \copydoc any(const u64x2, const uint64_t)
	inline bool any(const u32x4 lane_mask, const uint32_t bit_submask = 15) {
		assert(bit_submask > 0 && bit_submask < 16);

		return 0 != (bit_submask & _mm_movemask_ps(_mm_castsi128_ps(lane_mask.getn())));
	}

	/// \copydoc any(const u64x2, const uint64_t)
	inline bool any(const u16x8 lane_mask, const uint16_t bit_submask = 255) {
		assert(bit_submask > 0 && bit_submask < 256);

		return 0 != (bit_submask & deinterleaveBits8(_mm_movemask_epi8(lane_mask.getn())));
	}

	/// None-reduction. True if none of the masked lanes of the specified integral-type vector are set to -1.
	/// \param lane_mask integral-type vector containing only 0's and -1's.
	/// \param bit_submask bitmask of the lanes of interest; a set bit indicates interest in the lane
	/// of the same index as the bit.
	/// \return true if none of the lanes, specified in the submask, are set to -1, false otherwise.
	inline bool none(const u64x2 lane_mask, const uint64_t bit_submask = 3) {
		assert(bit_submask > 0 && bit_submask < 4);

		return 0 == (bit_submask & _mm_movemask_pd(_mm_castsi128_pd(lane_mask.getn())));
	}

	/// \copydoc none(const u64x4, const uint64_t)
	inline bool none(const u32x4 lane_mask, const uint32_t bit_submask = 15) {
		assert(bit_submask > 0 && bit_submask < 16);

		return 0 == (bit_submask & _mm_movemask_ps(_mm_castsi128_ps(lane_mask.getn())));
	}

	/// \copydoc none(const u64x4, const uint64_t)
	inline bool none(const u16x8 lane_mask, const uint16_t bit_submask = 255) {
		assert(bit_submask > 0 && bit_submask < 256);

		return 0 == (bit_submask & deinterleaveBits8(_mm_movemask_epi8(lane_mask.getn())));
	}

	// extrema /////////////////////////////////////////////////////////////////////////////////////

	/// Compute vector minimum. Return a vector where each lane is the minimum of the corresponding
	/// lanes of the argument vectors.
	/// \param a first argument.
	/// \param b second argument.
	/// \return a vector of the same type as the arguments, where each lane is the minimum
	/// of the corresponding lanes from the two arguments.
	inline f64x2 min(const f64x2 a, const f64x2 b) {
		return f64x2(_mm_min_pd(a.getn(), b.getn()), flag_native());
	}

	/// Compute vector maximum. Return a vector where each lane is the maximum of the corresponding
	/// lanes of the argument vectors.
	/// \param a first argument.
	/// \param b second argument.
	/// \return a vector of the same type as the arguments, where each lane is the maximum
	/// of the corresponding lanes from the two arguments.
	inline f64x2 max(const f64x2 a, const f64x2 b) {
		return f64x2(_mm_max_pd(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc min(const f64x2, const f64x2)
	inline f32x4 min(const f32x4 a, const f32x4 b) {
		return f32x4(_mm_min_ps(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc max(const f64x2, const f64x2)
	inline f32x4 max(const f32x4 a, const f32x4 b) {
		return f32x4(_mm_max_ps(a.getn(), b.getn()), flag_native());
	}

#if __SSE4_1__ != 0
	/// \copydoc min(const f64x2, const f64x2)
	inline s32x4 min(const s32x4 a, const s32x4 b) {
		return s32x4(_mm_min_epi32(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc max(const f64x2, const f64x2)
	inline s32x4 max(const s32x4 a, const s32x4 b) {
		return s32x4(_mm_max_epi32(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc min(const f64x2, const f64x2)
	inline u32x4 min(const u32x4 a, const u32x4 b) {
		return u32x4(_mm_min_epu32(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc max(const f64x2, const f64x2)
	inline u32x4 max(const u32x4 a, const u32x4 b) {
		return u32x4(_mm_max_epu32(a.getn(), b.getn()), flag_native());
	}

#else
	/// \copydoc min(const f64x2, const f64x2)
	inline s32x4 min(const s32x4 a, const s32x4 b) {
		const __m128i mask = _mm_cmplt_epi32(a.getn(), b.getn());
		return s32x4(_mm_or_si128(_mm_and_si128(mask, a.getn()), _mm_andnot_si128(mask, b.getn())), flag_native());
	}

	/// \copydoc max(const f64x2, const f64x2)
	inline s32x4 max(const s32x4 a, const s32x4 b) {
		const __m128i mask = _mm_cmpgt_epi32(a.getn(), b.getn());
		return s32x4(_mm_or_si128(_mm_and_si128(mask, a.getn()), _mm_andnot_si128(mask, b.getn())), flag_native());
	}

	/// \copydoc min(const f64x2, const f64x2)
	inline u32x4 min(const u32x4 a, const u32x4 b) {
		const __m128i maskNegA = _mm_cmplt_epi32(a.getn(), _mm_setzero_si128());
		const __m128i maskNegB = _mm_cmplt_epi32(b.getn(), _mm_setzero_si128());
		const __m128i maskLess = _mm_cmplt_epi32(a.getn(), b.getn());
		const __m128i maskOpp  = _mm_xor_si128(maskNegA, maskNegB);
		const __m128i mask     = _mm_xor_si128(maskLess, maskOpp);
		return u32x4(_mm_or_si128(_mm_and_si128(mask, a.getn()), _mm_andnot_si128(mask, b.getn())), flag_native());
	}

	/// \copydoc max(const f64x2, const f64x2)
	inline u32x4 max(const u32x4 a, const u32x4 b) {
		const __m128i maskNegA = _mm_cmplt_epi32(a.getn(), _mm_setzero_si128());
		const __m128i maskNegB = _mm_cmplt_epi32(b.getn(), _mm_setzero_si128());
		const __m128i maskLess = _mm_cmpgt_epi32(a.getn(), b.getn());
		const __m128i maskOpp  = _mm_xor_si128(maskNegA, maskNegB);
		const __m128i mask     = _mm_xor_si128(maskLess, maskOpp);
		return u32x4(_mm_or_si128(_mm_and_si128(mask, a.getn()), _mm_andnot_si128(mask, b.getn())), flag_native());
	}

#endif
	/// \copydoc min(const f64x2, const f64x2)
	inline s16x8 min(const s16x8 a, const s16x8 b) {
		return s16x8(_mm_min_epi16(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc max(const f64x2, const f64x2)
	inline s16x8 max(const s16x8 a, const s16x8 b) {
		return s16x8(_mm_max_epi16(a.getn(), b.getn()), flag_native());
	}

#if __SSE4_1__ != 0
	/// \copydoc min(const f64x2, const f64x2)
	inline u16x8 min(const u16x8 a, const u16x8 b) {
		return u16x8(_mm_min_epu16(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc max(const f64x2, const f64x2)
	inline u16x8 max(const u16x8 a, const u16x8 b) {
		return u16x8(_mm_max_epu16(a.getn(), b.getn()), flag_native());
	}

#else
	/// \copydoc min(const f64x2, const f64x2)
	inline u16x8 min(const u16x8 a, const u16x8 b) {
		const __m128i maskNegA = _mm_cmplt_epi16(a.getn(), _mm_setzero_si128());
		const __m128i maskNegB = _mm_cmplt_epi16(b.getn(), _mm_setzero_si128());
		const __m128i maskLess = _mm_cmplt_epi16(a.getn(), b.getn());
		const __m128i maskOpp  = _mm_xor_si128(maskNegA, maskNegB);
		const __m128i mask     = _mm_xor_si128(maskLess, maskOpp);
		return u16x8(_mm_or_si128(_mm_and_si128(mask, a.getn()), _mm_andnot_si128(mask, b.getn())), flag_native());
	}

	/// \copydoc max(const f64x2, const f64x2)
	inline u16x8 max(const u16x8 a, const u16x8 b) {
		const __m128i maskNegA = _mm_cmplt_epi16(a.getn(), _mm_setzero_si128());
		const __m128i maskNegB = _mm_cmplt_epi16(b.getn(), _mm_setzero_si128());
		const __m128i maskLess = _mm_cmpgt_epi16(a.getn(), b.getn());
		const __m128i maskOpp  = _mm_xor_si128(maskNegA, maskNegB);
		const __m128i mask     = _mm_xor_si128(maskLess, maskOpp);
		return u16x8(_mm_or_si128(_mm_and_si128(mask, a.getn()), _mm_andnot_si128(mask, b.getn())), flag_native());
	}

#endif
	// mask off lanes as nil ///////////////////////////////////////////////////////////////////////

	/// Mask off lanes of a vector as nil, according to a lane mask vector.
	/// \param x source vector.
	/// \param lane_mask integral-type vector of the same lane count and width as the source vector,
	/// whose lanes have values of -1 or 0, preserving or zeroing the resulting lanes, respectively.
	/// \return vector of the same type as the source, whose lanes contain either the value of the
	/// corresponding source lanes or zero.
	inline f64x2 mask(const f64x2 x, const u64x2 lane_mask) {
		return f64x2(_mm_and_pd(_mm_castsi128_pd(lane_mask.getn()), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x2, const u64x2)
	inline s64x2 mask(const s64x2 x, const u64x2 lane_mask) {
		return s64x2(_mm_and_si128(lane_mask.getn(), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x2, const u64x2)
	inline u64x2 mask(const u64x2 x, const u64x2 lane_mask) {
		return u64x2(_mm_and_si128(lane_mask.getn(), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x2, const u64x2)
	inline f32x4 mask(const f32x4 x, const u32x4 lane_mask) {
		return f32x4(_mm_and_ps(_mm_castsi128_ps(lane_mask.getn()), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x2, const u64x2)
	inline s32x4 mask(const s32x4 x, const u32x4 lane_mask) {
		return s32x4(_mm_and_si128(lane_mask.getn(), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x2, const u64x2)
	inline u32x4 mask(const u32x4 x, const u32x4 lane_mask) {
		return u32x4(_mm_and_si128(lane_mask.getn(), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x2, const u64x2)
	inline s16x8 mask(const s16x8 x, const u16x8 lane_mask) {
		return s16x8(_mm_and_si128(lane_mask.getn(), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x2, const u64x2)
	inline u16x8 mask(const u16x8 x, const u16x8 lane_mask) {
		return u16x8(_mm_and_si128(lane_mask.getn(), x.getn()), flag_native());
	}

	// combine lanes by bitmask literal ////////////////////////////////////////////////////////////

#if __AVX2__ != 0
	/// Combine the lanes from two arguments according to a bitmask literal; a zero bit in the literal
	/// causes the selection of a lane from the 1st argument, a one bit - from the 2nd argument.
	/// \param a first source.
	/// \param b second source.
	/// \return a vector of the same lane type and count as the source vectors.
	template < uint32_t SELECTOR >
	inline s32x4 select(const s32x4 a, const s32x4 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 16 > assert_selector;

		return s32x4(_mm_blend_epi32(a.getn(), b.getn(), SELECTOR), flag_native());
	}

	/// \copydoc select(const s32x4, const s32x4)
	template < uint32_t SELECTOR >
	inline u32x4 select(const u32x4 a, const u32x4 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 16 > assert_selector;

		return u32x4(_mm_blend_epi32(a.getn(), b.getn(), SELECTOR), flag_native());
	}

#else
	/// Combine the lanes from two arguments according to a bitmask literal; a zero bit in the literal
	/// causes the selection of a lane from the 1st argument, a one bit - from the 2nd argument.
	/// \param a first source.
	/// \param b second source.
	/// \return a vector of the same lane type and count as the source vectors.
	template < uint32_t SELECTOR >
	inline s32x4 select(const s32x4 a, const s32x4 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 16 > assert_selector;

		const int32_t u = -1;
		const __m128i vmsk = _mm_set_epi32(
			SELECTOR & 0x8 ? u : 0, SELECTOR & 0x4 ? u : 0, SELECTOR & 0x2 ? u : 0, SELECTOR & 0x1 ? u : 0);
		return s32x4(_mm_or_si128(_mm_andnot_si128(vmsk, a.getn()), _mm_and_si128(vmsk, b.getn())), flag_native());
	}

	/// \copydoc select(const s32x4, const s32x4)
	template < uint32_t SELECTOR >
	inline u32x4 select(const u32x4 a, const u32x4 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 16 > assert_selector;

		const int32_t u = -1;
		const __m128i vmsk = _mm_set_epi32(
			SELECTOR & 0x8 ? u : 0, SELECTOR & 0x4 ? u : 0, SELECTOR & 0x2 ? u : 0, SELECTOR & 0x1 ? u : 0);
		return u32x4(_mm_or_si128(_mm_andnot_si128(vmsk, a.getn()), _mm_and_si128(vmsk, b.getn())), flag_native());
	}

#endif
#if __SSE4_1__ != 0
	/// \copydoc select(const s32x4, const s32x4)
	template < uint32_t SELECTOR >
	inline f64x2 select(const f64x2 a, const f64x2 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 4 > assert_selector;

		return f64x2(_mm_blend_pd(a.getn(), b.getn(), SELECTOR), flag_native());
	}

	/// \copydoc select(const s32x4, const s32x4)
	template < uint32_t SELECTOR >
	inline f32x4 select(const f32x4 a, const f32x4 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 16 > assert_selector;

		return f32x4(_mm_blend_ps(a.getn(), b.getn(), SELECTOR), flag_native());
	}

	/// \copydoc select(const s32x4, const s32x4)
	template < uint32_t SELECTOR >
	inline s16x8 select(const s16x8 a, const s16x8 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 256 > assert_selector;

		return s16x8(_mm_blend_epi16(a.getn(), b.getn(), SELECTOR), flag_native());
	}

	/// \copydoc select(const s32x4, const s32x4)
	template < uint32_t SELECTOR >
	inline u16x8 select(const u16x8 a, const u16x8 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 256 > assert_selector;

		return u16x8(_mm_blend_epi16(a.getn(), b.getn(), SELECTOR), flag_native());
	}

	// combine lanes by lane mask //////////////////////////////////////////////////////////////////

	/// Combine the lanes from two arguments according to a lane-mask vector; a lane of 0 in the lane mask
	/// causes the selection of a lane from the 1st argument, a lane of -1 - from the 2nd argument.
	/// \param a first source.
	/// \param b second source.
	/// \param selector lane selection control.
	/// \return a vector of the same lane type and count as the source vectors.
	inline f64x2 select(const f64x2 a, const f64x2 b, const u64x2 selector) {
		return f64x2(_mm_blendv_pd(a.getn(), b.getn(), _mm_castsi128_pd(selector.getn())), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline s64x2 select(const s64x2 a, const s64x2 b, const u64x2 selector) {
		return s64x2(_mm_blendv_epi8(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline u64x2 select(const u64x2 a, const u64x2 b, const u64x2 selector) {
		return u64x2(_mm_blendv_epi8(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline f32x4 select(const f32x4 a, const f32x4 b, const u32x4 selector) {
		return f32x4(_mm_blendv_ps(a.getn(), b.getn(), _mm_castsi128_ps(selector.getn())), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline s32x4 select(const s32x4 a, const s32x4 b, const u32x4 selector) {
		return s32x4(_mm_blendv_epi8(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline u32x4 select(const u32x4 a, const u32x4 b, const u32x4 selector) {
		return u32x4(_mm_blendv_epi8(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline s16x8 select(const s16x8 a, const s16x8 b, const u16x8 selector) {
		return s16x8(_mm_blendv_epi8(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline u16x8 select(const u16x8 a, const u16x8 b, const u16x8 selector) {
		return u16x8(_mm_blendv_epi8(a.getn(), b.getn(), selector.getn()), flag_native());
	}

#else
	/// \copydoc select(const s32x4, const s32x4)
	template < uint32_t SELECTOR >
	inline f64x2 select(const f64x2 a, const f64x2 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 4 > assert_selector;

		const int64_t u = -1;
		const __m128d vmsk =  _mm_castsi128_pd(_mm_set_epi64x(
			SELECTOR & 0x2 ? u : 0, SELECTOR & 0x1 ? u : 0));
		return f64x2(_mm_or_pd(_mm_andnot_pd(vmsk, a.getn()), _mm_and_pd(vmsk, b.getn())), flag_native());
	}

	/// \copydoc select(const s32x4, const s32x4)
	template < uint32_t SELECTOR >
	inline f32x4 select(const f32x4 a, const f32x4 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 16 > assert_selector;

		const int32_t u = -1;
		const __m128 vmsk = _mm_castsi128_ps(_mm_set_epi32(
			SELECTOR & 0x8 ? u : 0, SELECTOR & 0x4 ? u : 0, SELECTOR & 0x2 ? u : 0, SELECTOR & 0x1 ? u : 0));
		return f32x4(_mm_or_ps(_mm_andnot_ps(vmsk, a.getn()), _mm_and_ps(vmsk, b.getn())), flag_native());
	}

	/// \copydoc select(const s32x4, const s32x4)
	template < uint32_t SELECTOR >
	inline s16x8 select(const s16x8 a, const s16x8 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 256 > assert_selector;

		const int16_t u = -1;
		const __m128i vmsk = _mm_set_epi16(
			SELECTOR & 0x80 ? u : 0, SELECTOR & 0x40 ? u : 0, SELECTOR & 0x20 ? u : 0, SELECTOR & 0x10 ? u : 0,
			SELECTOR & 0x08 ? u : 0, SELECTOR & 0x04 ? u : 0, SELECTOR & 0x02 ? u : 0, SELECTOR & 0x01 ? u : 0);
		return s16x8(_mm_or_si128(_mm_andnot_si128(vmsk, a.getn()), _mm_and_si128(vmsk, b.getn())), flag_native());
	}

	/// \copydoc select(const s32x4, const s32x4)
	template < uint32_t SELECTOR >
	inline u16x8 select(const u16x8 a, const u16x8 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 256 > assert_selector;

		const int16_t u = -1;
		const __m128i vmsk = _mm_set_epi16(
			SELECTOR & 0x80 ? u : 0, SELECTOR & 0x40 ? u : 0, SELECTOR & 0x20 ? u : 0, SELECTOR & 0x10 ? u : 0,
			SELECTOR & 0x08 ? u : 0, SELECTOR & 0x04 ? u : 0, SELECTOR & 0x02 ? u : 0, SELECTOR & 0x01 ? u : 0);
		return u16x8(_mm_or_si128(_mm_andnot_si128(vmsk, a.getn()), _mm_and_si128(vmsk, b.getn())), flag_native());
	}

	// combine lanes by lane mask //////////////////////////////////////////////////////////////////

	/// Combine the lanes from two arguments according to a lane-mask vector; a lane of 0 in the lane mask
	/// causes the selection of a lane from the 1st argument, a lane of -1 - from the 2nd argument.
	/// \param a first source.
	/// \param b second source.
	/// \param selector lane selection control.
	/// \return a vector of the same lane type and count as the source vectors.
	inline f64x2 select(const f64x2 a, const f64x2 b, const u64x2 selector) {
		const __m128d vmask = _mm_castsi128_pd(selector.getn());
		return f64x2(_mm_or_pd(_mm_andnot_pd(vmask, a.getn()), _mm_and_pd(vmask, b.getn())), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline s64x2 select(const s64x2 a, const s64x2 b, const u64x2 selector) {
		const __m128i vmask = selector.getn();
		return s64x2(_mm_or_si128(_mm_andnot_si128(vmask, a.getn()), _mm_and_si128(vmask, b.getn())), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline u64x2 select(const u64x2 a, const u64x2 b, const u64x2 selector) {
		const __m128i vmask = selector.getn();
		return u64x2(_mm_or_si128(_mm_andnot_si128(vmask, a.getn()), _mm_and_si128(vmask, b.getn())), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline f32x4 select(const f32x4 a, const f32x4 b, const u32x4 selector) {
		const __m128 vmask = _mm_castsi128_ps(selector.getn());
		return f32x4(_mm_or_ps(_mm_andnot_ps(vmask, a.getn()), _mm_and_ps(vmask, b.getn())), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline s32x4 select(const s32x4 a, const s32x4 b, const u32x4 selector) {
		const __m128i vmask = selector.getn();
		return s32x4(_mm_or_si128(_mm_andnot_si128(vmask, a.getn()), _mm_and_si128(vmask, b.getn())), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline u32x4 select(const u32x4 a, const u32x4 b, const u32x4 selector) {
		const __m128i vmask = selector.getn();
		return u32x4(_mm_or_si128(_mm_andnot_si128(vmask, a.getn()), _mm_and_si128(vmask, b.getn())), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline s16x8 select(const s16x8 a, const s16x8 b, const u16x8 selector) {
		const __m128i vmask = selector.getn();
		return s16x8(_mm_or_si128(_mm_andnot_si128(vmask, a.getn()), _mm_and_si128(vmask, b.getn())), flag_native());
	}

	/// \copydoc select(const f64x2 a, const f64x2 b, const u64x2)
	inline u16x8 select(const u16x8 a, const u16x8 b, const u16x8 selector) {
		const __m128i vmask = selector.getn();
		return u16x8(_mm_or_si128(_mm_andnot_si128(vmask, a.getn()), _mm_and_si128(vmask, b.getn())), flag_native());
	}

#endif
	// absolute value //////////////////////////////////////////////////////////////////////////////

	/// Compute the absolute value of the source vector.
	/// \param x source vector.
	/// \return a vector of the same lane type and count as the source,
	/// whose lanes contain the absolute values of the source lanes.
	inline f64x2 abs(const f64x2 x) {
		return f64x2(_mm_andnot_pd(_mm_set1_pd(-0.0), x.getn()), flag_native());
	}

	/// \copydoc abs(const f64x2)
	inline f32x4 abs(const f32x4 x) {
		return f32x4(_mm_andnot_ps(_mm_set1_ps(-0.f), x.getn()), flag_native());
	}

#if __SSSE3__ != 0
	/// \copydoc abs(const f64x2)
	inline s32x4 abs(const s32x4 x) {
		return s32x4(_mm_abs_epi32(x.getn()), flag_native());
	}

	/// \copydoc abs(const f64x2)
	inline s16x8 abs(const s16x8 x) {
		return s16x8(_mm_abs_epi16(x.getn()), flag_native());
	}

#else
	/// \copydoc abs(const f64x2)
	inline s32x4 abs(const s32x4 x) {
		const __m128i mask = _mm_cmpgt_epi32(_mm_setzero_si128(), x.getn());
		const __m128i r = _mm_sub_epi32(_mm_xor_si128(mask, x.getn()), mask);
		return s32x4(r, flag_native());
	}

	/// \copydoc abs(const f64x2)
	inline s16x8 abs(const s16x8 x) {
		const __m128i mask = _mm_cmpgt_epi16(_mm_setzero_si128(), x.getn());
		const __m128i r = _mm_sub_epi16(_mm_xor_si128(mask, x.getn()), mask);
		return s16x8(r, flag_native());
	}

#endif
#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE != 0 || COMPILER_QUIRK_0003_RELATIONAL_OPS != 0
	/// Compare two vectors for equality.
	/// \param a first source vector.
	/// \param b second source vector.
	/// \return an integral-type vector of the same lane count and width as the sources,
	/// containing -1 in the lanes where the condition was satisfied, and 0 in the remaining lanes.
	inline u64x2 operator ==(const f64x2 a, const f64x2 b) {
		return u64x2(_mm_castpd_si128(_mm_cmpeq_pd(a.getn(), b.getn())), flag_native());
	}

#if __SSE4_1__ != 0
	inline u64x2 operator ==(const s64x2 a, const s64x2 b) {
		return u64x2(_mm_cmpeq_epi64(a.getn(), b.getn()), flag_native());
	}

#else
	inline u64x2 operator ==(const s64x2 a, const s64x2 b) {
		const __m128i eq0123 = _mm_cmpeq_epi32(a.getn(), b.getn());
		const __m128i eq0011 = _mm_unpacklo_epi32(eq0123, eq0123);
		const __m128i eq2233 = _mm_unpackhi_epi32(eq0123, eq0123);
		const __m128i eq0022 = _mm_unpacklo_epi64(eq0011, eq2233);
		const __m128i eq1133 = _mm_unpackhi_epi64(eq0011, eq2233);
		const __m128i r      = _mm_and_si128(eq0022, eq1133);
		return u64x2(r, flag_native());
	}

#endif
	inline u64x2 operator ==(const u64x2 a, const u64x2 b) {
		return as_s64x2(a) == as_s64x2(b);
	}

	inline u32x4 operator ==(const f32x4 a, const f32x4 b) {
		return u32x4(_mm_castps_si128(_mm_cmpeq_ps(a.getn(), b.getn())), flag_native());
	}

	inline u32x4 operator ==(const s32x4 a, const s32x4 b) {
		return u32x4(_mm_cmpeq_epi32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator ==(const u32x4 a, const u32x4 b) {
		return as_s32x4(a) == as_s32x4(b);
	}

	inline u16x8 operator ==(const s16x8 a, const s16x8 b) {
		return u16x8(_mm_cmpeq_epi16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator ==(const u16x8 a, const u16x8 b) {
		return as_s16x8(a) == as_s16x8(b);
	}

	inline u64x2 operator !=(const f64x2 a, const f64x2 b) {
		return u64x2(_mm_castpd_si128(_mm_cmpneq_pd(a.getn(), b.getn())), flag_native());
	}

#if __SSE4_1__ != 0
	inline u64x2 operator !=(const s64x2 a, const s64x2 b) {
		return u64x2(_mm_xor_si128(_mm_cmpeq_epi64(a.getn(), b.getn()), _mm_set1_epi64x(-1)), flag_native());
	}

#else
	inline u64x2 operator !=(const s64x2 a, const s64x2 b) {
		const __m128i eq0123 = _mm_xor_si128(_mm_cmpeq_epi32(a.getn(), b.getn()), _mm_set1_epi64x(-1));
		const __m128i eq0011 = _mm_unpacklo_epi32(eq0123, eq0123);
		const __m128i eq2233 = _mm_unpackhi_epi32(eq0123, eq0123);
		const __m128i eq0022 = _mm_unpacklo_epi64(eq0011, eq2233);
		const __m128i eq1133 = _mm_unpackhi_epi64(eq0011, eq2233);
		const __m128i r      = _mm_and_si128(eq0022, eq1133);
		return u64x2(r, flag_native());
	}

#endif
	inline u64x2 operator !=(const u64x2 a, const u64x2 b) {
		return as_s64x2(a) != as_s64x2(b);
	}

	inline u32x4 operator !=(const f32x4 a, const f32x4 b) {
		return u32x4(_mm_castps_si128(_mm_cmpneq_ps(a.getn(), b.getn())), flag_native());
	}

	inline u32x4 operator !=(const s32x4 a, const s32x4 b) {
		return u32x4(_mm_xor_si128(_mm_cmpeq_epi32(a.getn(), b.getn()), _mm_set1_epi64x(-1)), flag_native());
	}

	inline u32x4 operator !=(const u32x4 a, const u32x4 b) {
		return as_s32x4(a) != as_s32x4(b);
	}

	inline u16x8 operator !=(const s16x8 a, const s16x8 b) {
		return u16x8(_mm_xor_si128(_mm_cmpeq_epi16(a.getn(), b.getn()), _mm_set1_epi64x(-1)), flag_native());
	}

	inline u16x8 operator !=(const u16x8 a, const u16x8 b) {
		return as_s16x8(a) != as_s16x8(b);
	}

	inline u64x2 operator <(const f64x2 a, const f64x2 b) {
		return u64x2(_mm_castpd_si128(_mm_cmplt_pd(a.getn(), b.getn())), flag_native());
	}

#if __SSE4_2__ != 0
	inline u64x2 operator <(const s64x2 a, const s64x2 b) {
		return u64x2(_mm_cmpgt_epi64(b.getn(), a.getn()), flag_native());
	}

#endif
	inline u32x4 operator <(const f32x4 a, const f32x4 b) {
		return u32x4(_mm_castps_si128(_mm_cmplt_ps(a.getn(), b.getn())), flag_native());
	}

	inline u32x4 operator <(const s32x4 a, const s32x4 b) {
		return u32x4(_mm_cmplt_epi32(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator <(const s16x8 a, const s16x8 b) {
		return u16x8(_mm_cmplt_epi16(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator >(const f64x2 a, const f64x2 b) {
		return b < a;
	}

#if __SSE4_2__ != 0
	inline u64x2 operator >(const s64x2 a, const s64x2 b) {
		return b < a;
	}

#endif
	inline u32x4 operator >(const f32x4 a, const f32x4 b) {
		return b < a;
	}

	inline u32x4 operator >(const s32x4 a, const s32x4 b) {
		return b < a;
	}

	inline u16x8 operator >(const s16x8 a, const s16x8 b) {
		return b < a;
	}

	inline u64x2 operator <=(const f64x2 a, const f64x2 b) {
		return u64x2(_mm_castpd_si128(_mm_cmple_pd(a.getn(), b.getn())), flag_native());
	}

#if __SSE4_2__ != 0
	inline u64x2 operator <=(const s64x2 a, const s64x2 b) {
		return u64x2(_mm_xor_si128(_mm_cmpgt_epi64(a.getn(), b.getn()), _mm_set1_epi64x(-1)), flag_native());
	}

#endif
	inline u32x4 operator <=(const f32x4 a, const f32x4 b) {
		return u32x4(_mm_castps_si128(_mm_cmple_ps(a.getn(), b.getn())), flag_native());
	}

	inline u32x4 operator <=(const s32x4 a, const s32x4 b) {
		return u32x4(_mm_xor_si128(_mm_cmpgt_epi32(a.getn(), b.getn()), _mm_set1_epi64x(-1)), flag_native());
	}

	inline u16x8 operator <=(const s16x8 a, const s16x8 b) {
		return u16x8(_mm_xor_si128(_mm_cmpgt_epi16(a.getn(), b.getn()), _mm_set1_epi64x(-1)), flag_native());
	}

	inline u64x2 operator >=(const f64x2 a, const f64x2 b) {
		return b <= a;
	}

#if __SSE4_2__ != 0
	inline u64x2 operator >=(const s64x2 a, const s64x2 b) {
		return b <= a;
	}

#endif
	inline u32x4 operator >=(const f32x4 a, const f32x4 b) {
		return b <= a;
	}

	inline u32x4 operator >=(const s32x4 a, const s32x4 b) {
		return b <= a;
	}

	inline u16x8 operator >=(const s16x8 a, const s16x8 b) {
		return b <= a;
	}

#endif
#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE != 0
	inline f64x2 operator +(const f64x2 a, const f64x2 b) {
		return f64x2(_mm_add_pd(a.getn(), b.getn()), flag_native());
	}

	inline s64x2 operator +(const s64x2 a, const s64x2 b) {
		return s64x2(_mm_add_epi64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator +(const u64x2 a, const u64x2 b) {
		return u64x2(_mm_add_epi64(a.getn(), b.getn()), flag_native());
	}

	inline f32x4 operator +(const f32x4 a, const f32x4 b) {
		return f32x4(_mm_add_ps(a.getn(), b.getn()), flag_native());
	}

	inline s32x4 operator +(const s32x4 a, const s32x4 b) {
		return s32x4(_mm_add_epi32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator +(const u32x4 a, const u32x4 b) {
		return u32x4(_mm_add_epi32(a.getn(), b.getn()), flag_native());
	}

	inline s16x8 operator +(const s16x8 a, const s16x8 b) {
		return s16x8(_mm_add_epi16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator +(const u16x8 a, const u16x8 b) {
		return u16x8(_mm_add_epi16(a.getn(), b.getn()), flag_native());
	}

	inline f64x2 operator -(const f64x2 a, const f64x2 b) {
		return f64x2(_mm_sub_pd(a.getn(), b.getn()), flag_native());
	}

	inline s64x2 operator -(const s64x2 a, const s64x2 b) {
		return s64x2(_mm_sub_epi64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator -(const u64x2 a, const u64x2 b) {
		return u64x2(_mm_sub_epi64(a.getn(), b.getn()), flag_native());
	}

	inline f32x4 operator -(const f32x4 a, const f32x4 b) {
		return f32x4(_mm_sub_ps(a.getn(), b.getn()), flag_native());
	}

	inline s32x4 operator -(const s32x4 a, const s32x4 b) {
		return s32x4(_mm_sub_epi32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator -(const u32x4 a, const u32x4 b) {
		return u32x4(_mm_sub_epi32(a.getn(), b.getn()), flag_native());
	}

	inline s16x8 operator -(const s16x8 a, const s16x8 b) {
		return s16x8(_mm_sub_epi16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator -(const u16x8 a, const u16x8 b) {
		return u16x8(_mm_sub_epi16(a.getn(), b.getn()), flag_native());
	}

	inline f64x2 operator *(const f64x2 a, const f64x2 b) {
		return f64x2(_mm_mul_pd(a.getn(), b.getn()), flag_native());
	}

	inline f32x4 operator *(const f32x4 a, const f32x4 b) {
		return f32x4(_mm_mul_ps(a.getn(), b.getn()), flag_native());
	}

#if __SSE4_1__ != 0
	inline s32x4 operator *(const s32x4 a, const s32x4 b) {
		return s32x4(_mm_mullo_epi32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator *(const u32x4 a, const u32x4 b) {
		return u32x4(_mm_mullo_epi32(a.getn(), b.getn()), flag_native());
	}

#endif
	inline s16x8 operator *(const s16x8 a, const s16x8 b) {
		return s16x8(_mm_mullo_epi16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator *(const u16x8 a, const u16x8 b) {
		return u16x8(_mm_mullo_epi16(a.getn(), b.getn()), flag_native());
	}

	inline f64x2 operator /(const f64x2 a, const f64x2 b) {
		return f64x2(_mm_div_pd(a.getn(), b.getn()), flag_native());
	}

	inline f32x4 operator /(const f32x4 a, const f32x4 b) {
		return f32x4(_mm_div_ps(a.getn(), b.getn()), flag_native());
	}

	inline f64x2 operator -(const f64x2 x) {
		return f64x2(_mm_xor_pd(x.getn(), _mm_set1_pd(-0.0)), flag_native());
	}

	inline f32x4 operator -(const f32x4 x) {
		return f32x4(_mm_xor_ps(x.getn(), _mm_set1_ps(-0.0)), flag_native());
	}

	inline s32x4 operator -(const s32x4 x) {
		return s32x4(_mm_sub_epi32(_mm_setzero_si128(), x.getn()), flag_native());
	}

	inline s16x8 operator -(const s16x8 x) {
		return s16x8(_mm_sub_epi16(_mm_setzero_si128(), x.getn()), flag_native());
	}

#endif
	/// Shift left logical /////////////////////////////////////////////////////////////////////////

	/// Shift left the argument by the count literal; if the count is larger or equal to the
	/// number of bits of the source argument, the result is undefined (as per the C++ standard)
	/// \param a source to be shifted
	/// \return vector where each lane is the result from the above-described op on the corresponding lane of the source
	template < uint32_t COUNT >
	inline s64x2 shl(const s64x2 a) {
		return s64x2(_mm_slli_epi64(a.getn(), int(COUNT)), flag_native());
	}

	/// \copydoc shl(const s64x2 a)
	template < uint32_t COUNT >
	inline u64x2 shl(const u64x2 a) {
		return as_u64x2(shl< COUNT >(as_s64x2(a)));
	}

	/// \copydoc shl(const s64x2 a)
	template < uint32_t COUNT >
	inline s32x4 shl(const s32x4 a) {
		return s32x4(_mm_slli_epi32(a.getn(), int(COUNT)), flag_native());
	}

	/// \copydoc shl(const s64x2 a)
	template < uint32_t COUNT >
	inline u32x4 shl(const u32x4 a) {
		return as_u32x4(shl< COUNT >(as_s32x4(a)));
	}

	/// \copydoc shl(const s64x2 a)
	template < uint32_t COUNT >
	inline s16x8 shl(const s16x8 a) {
		return s16x8(_mm_slli_epi16(a.getn(), int(COUNT)), flag_native());
	}

	/// \copydoc shl(const s64x2 a)
	template < uint32_t COUNT >
	inline u16x8 shl(const u16x8 a) {
		return as_u16x8(shl< COUNT >(as_s16x8(a)));
	}

	/// Shift left the first argument by the count in the second argument; if the count is larger or equal
	/// to the number of bits of the first argument, the result is undefined (as per the C++ standard)
	/// \param a source to be shifted
	/// \param c shift count
	/// \return vector where each lane is the result from the above-described op on the corresponding lane of the source
	inline s64x2 shl(const s64x2 a, const u64x2 c) {

#if __AVX2__ != 0
		return s64x2(_mm_sllv_epi64(a.getn(), c.getn()), flag_native());

#else
		// Pre-AVX2 ISAs don't have independent lane shifts.
		const __m128i mask0 = _mm_set_epi64x(0, -1);
		const __m128i mask1 = _mm_slli_si128(mask0, 8);
		const __m128i c0 =                c.getn();
		const __m128i c1 = _mm_srli_si128(c.getn(), 8);

		const __m128i r0 = _mm_and_si128(_mm_sll_epi64(a.getn(), c0), mask0);
		const __m128i r1 = _mm_and_si128(_mm_sll_epi64(a.getn(), c1), mask1);

		return s64x2(_mm_or_si128(r0, r1), flag_native());

#endif
	}

	/// \copydoc shl(const s64x2 a, const u64x2 c)
	inline u64x2 shl(const u64x2 a, const u64x2 c) {
		return as_u64x2(shl(as_s64x2(a), c));
	}

	/// \copydoc shl(const s64x2 a, const u64x2 c)
	inline s32x4 shl(const s32x4 a, const u32x4 c) {

#if __AVX2__ != 0
		return s32x4(_mm_sllv_epi32(a.getn(), c.getn()), flag_native());

#else
		// Pre-AVX2 ISAs don't have independent lane shifts.

#if __SSE4_1__ != 0
		const __m128i exp = _mm_add_epi32(_mm_slli_epi32(c.getn(), 23), _mm_set1_epi32(0x3f800000));
		const __m128i pot = _mm_cvttps_epi32(_mm_castsi128_ps(exp));
		const __m128i r   = _mm_mullo_epi32(a.getn(), pot);

		return s32x4(r, flag_native());

#else
		const __m128i exp = _mm_add_epi32(_mm_slli_epi32(c.getn(), 23), _mm_set1_epi32(0x3f800000));
		const __m128i pot = _mm_cvttps_epi32(_mm_castsi128_ps(exp));
		const __m128i a13 = _mm_shuffle_epi32(a.getn(), 0xf5);
		const __m128i r02 = _mm_shuffle_epi32(_mm_mul_epu32(pot, a.getn()), 0xe8);
		const __m128i p13 = _mm_shuffle_epi32(pot, 0xf5);
		const __m128i r13 = _mm_shuffle_epi32(_mm_mul_epu32(p13, a13), 0xe8);

		return s32x4(_mm_unpacklo_epi32(r02, r13), flag_native());

#endif
#endif
	}

	/// \copydoc shl(const s64x2 a, const u64x2 c)
	inline u32x4 shl(const u32x4 a, const u32x4 c) {
		return as_u32x4(shl(as_s32x4(a), c));
	}

	/// \copydoc shl(const s64x2 a, const u64x2 c)
	inline s16x8 shl(const s16x8 a, const u16x8 c) {

#if __AVX2__ != 0
		const __m256i mask = _mm256_unpacklo_epi16(_mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256()), _mm256_setzero_si256());
		const __m256i a32 = _mm256_cvtepu16_epi32(a.getn());
		const __m256i c32 = _mm256_cvtepu16_epi32(c.getn());
		const __m256i r32 = _mm256_and_si256(_mm256_sllv_epi32(a32, c32), mask);
		const __m128i rl = _mm256_castsi256_si128(r32);
		const __m128i rh = _mm256_extracti128_si256(r32, 1);

		return s16x8(_mm_packus_epi32(rl, rh), flag_native());

#else
		// Pre-AVX2 ISAs don't have independent lane shifts.

#if __SSE4_1__ != 0
		const __m128i mask3 = _mm_or_si128(_mm_slli_epi16(c.getn(), 12), _mm_slli_epi16(c.getn(), 4));
		__m128i r = a.getn();

		const __m128i r_s8 = _mm_slli_epi16(r, 8);
		r = _mm_blendv_epi8(r, r_s8, mask3);

		const __m128i mask2 = _mm_slli_epi16(mask3, 1);
		const __m128i r_s4 = _mm_slli_epi16(r, 4);
		r = _mm_blendv_epi8(r, r_s4, mask2);

		const __m128i mask1 = _mm_slli_epi16(mask3, 2);
		const __m128i r_s2 = _mm_slli_epi16(r, 2);
		r = _mm_blendv_epi8(r, r_s2, mask1);

		const __m128i mask0 = _mm_slli_epi16(mask3, 3);
		const __m128i r_s1 = _mm_slli_epi16(r, 1);
		r = _mm_blendv_epi8(r, r_s1, mask0);

		return s16x8(r, flag_native());

#else
		const __m128i mask3 = _mm_srai_epi16(_mm_slli_epi16(c.getn(), 12), 15);
		const __m128i mask2 = _mm_srai_epi16(_mm_slli_epi16(c.getn(), 13), 15);
		const __m128i mask1 = _mm_srai_epi16(_mm_slli_epi16(c.getn(), 14), 15);
		const __m128i mask0 = _mm_srai_epi16(_mm_slli_epi16(c.getn(), 15), 15);
		__m128i r = a.getn();

		const __m128i r_s8 = _mm_slli_epi16(r, 8);
		r = _mm_or_si128(_mm_and_si128(mask3, r_s8), _mm_andnot_si128(mask3, r));

		const __m128i r_s4 = _mm_slli_epi16(r, 4);
		r = _mm_or_si128(_mm_and_si128(mask2, r_s4), _mm_andnot_si128(mask2, r));

		const __m128i r_s2 = _mm_slli_epi16(r, 2);
		r = _mm_or_si128(_mm_and_si128(mask1, r_s2), _mm_andnot_si128(mask1, r));

		const __m128i r_s1 = _mm_slli_epi16(r, 1);
		r = _mm_or_si128(_mm_and_si128(mask0, r_s1), _mm_andnot_si128(mask0, r));

		return s16x8(r, flag_native());

#endif
#endif
	}

	/// \copydoc shl(const s64x2 a, const u64x2 c)
	inline u16x8 shl(const u16x8 a, const u16x8 c) {
		return as_u16x8(shl(as_s16x8(a), c));
	}

	/// Shift right logical (unsigned) and arithmetic (signed) /////////////////////////////////////

	/// Shift right arithmetically the argument by the count literal; if the count is larger or equal to
	/// the number of bits of the source argument, the result is undefined (as per the C++ standard)
	/// \param a source to be shifted
	/// \return vector where each lane is the result from the above-described op on the corresponding lane of the source
	template < uint32_t COUNT >
	inline s64x2 shr(const s64x2 a) {

#if __SSE4_2__ != 0
		if (32 >= COUNT) {
			const __m128i s = _mm_srai_epi32(a.getn(), int(COUNT));
			const __m128i r = _mm_srli_epi64(a.getn(), int(COUNT));
			return s64x2(_mm_blend_epi16(r, s, 0xcc), flag_native());
		}
		else {
			const __m128i s = _mm_slli_epi64(_mm_cmpgt_epi64(_mm_setzero_si128(), a.getn()), int(sizeof(s64) * 8 - COUNT));
			const __m128i r = _mm_srli_epi64(a.getn(), int(COUNT));
			return s64x2(_mm_or_si128(r, s), flag_native());
		}

#else
		if (32 >= COUNT) {
			const __m128i s = _mm_srai_epi32(_mm_and_si128(a.getn(), _mm_set1_epi64x(0xffffffff00000000)), int(COUNT));
			const __m128i r = _mm_srli_epi64(a.getn(), int(COUNT));
			return s64x2(_mm_or_si128(r, s), flag_native());
		}
		else {
			const __m128i rl = _mm_srai_epi32(a.getn(), int(COUNT - 32));
			const __m128i rh = _mm_srai_epi32(a.getn(), 32);
			const __m128i r02 = _mm_shuffle_epi32(rl, 0xfd);
			const __m128i r13 = _mm_shuffle_epi32(rh, 0xfd);
			return s64x2(_mm_unpacklo_epi32(r02, r13), flag_native());
		}

#endif
	}

	/// Shift right logically the argument by the count literal; if the count is larger or equal to
	/// the number of bits of the source argument, the result is undefined (as per the C++ standard)
	/// \param a source to be shifted
	/// \return vector where each lane is the result from the above-described op on the corresponding lane of the source
	template < uint32_t COUNT >
	inline u64x2 shr(const u64x2 a) {
		return u64x2(_mm_srli_epi64(a.getn(), int(COUNT)), flag_native());
	}

	/// \copydoc shr(const s64x2 a)
	template < uint32_t COUNT >
	inline s32x4 shr(const s32x4 a) {
		return s32x4(_mm_srai_epi32(a.getn(), int(COUNT)), flag_native());
	}

	/// \copydoc shr(const u64x2 a)
	template < uint32_t COUNT >
	inline u32x4 shr(const u32x4 a) {
		return u32x4(_mm_srli_epi32(a.getn(), int(COUNT)), flag_native());
	}

	/// \copydoc shr(const s64x2 a)
	template < uint32_t COUNT >
	inline s16x8 shr(const s16x8 a) {
		return s16x8(_mm_srai_epi16(a.getn(), int(COUNT)), flag_native());
	}

	/// \copydoc shr(const u64x2 a)
	template < uint32_t COUNT >
	inline u16x8 shr(const u16x8 a) {
		return u16x8(_mm_srli_epi16(a.getn(), int(COUNT)), flag_native());
	}

	/// Shift right arithmetically the first argument by the count in the second argument; if the count is larger
	/// or equal to the number of bits of the first argument, the result is undefined (as per the C++ standard)
	/// \param a source to be shifted
	/// \param c shift count
	/// \return vector where each lane is the result from the above-described op on the corresponding lane of the source
	inline s64x2 shr(const s64x2 a, const u64x2 c) {
		// Here we rely that both gcc and msvc do arithmetic shifts on signed integrals on x86-64
		const compile_assert< (-1 >> 1) == -1 > assert_shift_arithmetic_right;

		const s64 a0 = a[0];
		const s64 c0 = c[0];
		const s64 a1 = a[1];
		const s64 c1 = c[1];

		const s64 r0 = a0 >> c0;
		const s64 r1 = a1 >> c1;
		return s64x2(r0, r1);
	}

	/// Shift right logically the first argument by the count in the second argument; if the count is larger
	/// or equal to the number of bits of the first argument, the result is undefined (as per the C++ standard)
	/// \param a source to be shifted
	/// \param c shift count
	/// \return vector where each lane is the result from the above-described op on the corresponding lane of the source
	inline u64x2 shr(const u64x2 a, const u64x2 c) {

#if __AVX2__ != 0
		return u64x2(_mm_srlv_epi64(a.getn(), c.getn()), flag_native());

#else
		const __m128i a1 = _mm_srli_si128(a.getn(), 8);
		const __m128i c1 = _mm_srli_si128(c.getn(), 8);
		const __m128i r0 = _mm_srl_epi64(a.getn(), c.getn());
		const __m128i r1 = _mm_srl_epi64(a1, c1);
		return u64x2(_mm_unpacklo_epi64(r0, r1), flag_native());

#endif
	}

	/// \copydoc shr(const s64x2 a, const u64x2 c)
	inline s32x4 shr(const s32x4 a, const u32x4 c) {

#if __AVX2__ != 0
		return s32x4(_mm_srav_epi32(a.getn(), c.getn()), flag_native());

#else
#if __SSE4_1__ != 0
		const __m128i c01 = _mm_unpacklo_epi32(c.getn(), _mm_setzero_si128());
		const __m128i c23 = _mm_unpackhi_epi32(c.getn(), _mm_setzero_si128());
		const __m128i c1 = _mm_srli_si128(c01, 8);
		const __m128i c3 = _mm_srli_si128(c23, 8);
		const __m128i r0 = _mm_sra_epi32(a.getn(), c01);
		const __m128i r2 = _mm_sra_epi32(a.getn(), c23);
		const __m128i r1 = _mm_sra_epi32(a.getn(), c1);
		const __m128i r3 = _mm_sra_epi32(a.getn(), c3);
		const __m128i r01 = _mm_blend_epi16(r0, r1, 0xcc);
		const __m128i r23 = _mm_blend_epi16(r2, r3, 0xcc);
		return s32x4(_mm_blend_epi16(r01, r23, 0xf0), flag_native());

#else
		const __m128i c01 = _mm_unpacklo_epi32(c.getn(), _mm_setzero_si128());
		const __m128i c23 = _mm_unpackhi_epi32(c.getn(), _mm_setzero_si128());
		const __m128i c1 = _mm_srli_si128(c01, 8);
		const __m128i c3 = _mm_srli_si128(c23, 8);
		const __m128i a0 =                a.getn();
		const __m128i a1 = _mm_srli_si128(a.getn(), 4);
		const __m128i a2 = _mm_srli_si128(a.getn(), 8);
		const __m128i a3 = _mm_srli_si128(a.getn(), 12);
		const __m128i r0 = _mm_sra_epi32(a0, c01);
		const __m128i r2 = _mm_sra_epi32(a2, c23);
		const __m128i r1 = _mm_sra_epi32(a1, c1);
		const __m128i r3 = _mm_sra_epi32(a3, c3);
		const __m128i r02 = _mm_unpacklo_epi32(r0, r2);
		const __m128i r13 = _mm_unpacklo_epi32(r1, r3);
		return s32x4(_mm_unpacklo_epi32(r02, r13), flag_native());

#endif
#endif
	}

	/// \copydoc shr(const s64x2 a, const u64x2 c)
	inline u32x4 shr(const u32x4 a, const u32x4 c) {

#if __AVX2__ != 0
		return u32x4(_mm_srlv_epi32(a.getn(), c.getn()), flag_native());

#else
#if __SSE4_1__ != 0
		const __m128i c01 = _mm_unpacklo_epi32(c.getn(), _mm_setzero_si128());
		const __m128i c23 = _mm_unpackhi_epi32(c.getn(), _mm_setzero_si128());
		const __m128i c1 = _mm_srli_si128(c01, 8);
		const __m128i c3 = _mm_srli_si128(c23, 8);
		const __m128i r0 = _mm_srl_epi32(a.getn(), c01);
		const __m128i r2 = _mm_srl_epi32(a.getn(), c23);
		const __m128i r1 = _mm_srl_epi32(a.getn(), c1);
		const __m128i r3 = _mm_srl_epi32(a.getn(), c3);
		const __m128i r01 = _mm_blend_epi16(r0, r1, 0xcc);
		const __m128i r23 = _mm_blend_epi16(r2, r3, 0xcc);
		return u32x4(_mm_blend_epi16(r01, r23, 0xf0), flag_native());

#else
		const __m128i c01 = _mm_unpacklo_epi32(c.getn(), _mm_setzero_si128());
		const __m128i c23 = _mm_unpackhi_epi32(c.getn(), _mm_setzero_si128());
		const __m128i c1 = _mm_srli_si128(c01, 8);
		const __m128i c3 = _mm_srli_si128(c23, 8);
		const __m128i a0 =                a.getn();
		const __m128i a1 = _mm_srli_si128(a.getn(), 4);
		const __m128i a2 = _mm_srli_si128(a.getn(), 8);
		const __m128i a3 = _mm_srli_si128(a.getn(), 12);
		const __m128i r0 = _mm_srl_epi32(a0, c01);
		const __m128i r2 = _mm_srl_epi32(a2, c23);
		const __m128i r1 = _mm_srl_epi32(a1, c1);
		const __m128i r3 = _mm_srl_epi32(a3, c3);
		const __m128i r02 = _mm_unpacklo_epi32(r0, r2);
		const __m128i r13 = _mm_unpacklo_epi32(r1, r3);
		return u32x4(_mm_unpacklo_epi32(r02, r13), flag_native());

#endif
#endif
	}

	/// \copydoc shr(const s64x2 a, const u64x2 c)
	inline s16x8 shr(const s16x8 a, const u16x8 c) {

#if __AVX2__ != 0
		const __m256i mask = _mm256_unpacklo_epi16(_mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256()), _mm256_setzero_si256());
		const __m256i a32 = _mm256_cvtepi16_epi32(a.getn());
		const __m256i c32 = _mm256_cvtepu16_epi32(c.getn());
		const __m256i r32 = _mm256_and_si256(_mm256_srav_epi32(a32, c32), mask);
		const __m128i rl = _mm256_castsi256_si128(r32);
		const __m128i rh = _mm256_extracti128_si256(r32, 1);

		return s16x8(_mm_packus_epi32(rl, rh), flag_native());

#else
#if __SSE4_1__ != 0
		const __m128i mask3 = _mm_or_si128(_mm_slli_epi16(c.getn(), 12), _mm_slli_epi16(c.getn(), 4));
		__m128i r = a.getn();

		const __m128i r_s8 = _mm_srai_epi16(r, 8);
		r = _mm_blendv_epi8(r, r_s8, mask3);

		const __m128i mask2 = _mm_slli_epi16(mask3, 1);
		const __m128i r_s4 = _mm_srai_epi16(r, 4);
		r = _mm_blendv_epi8(r, r_s4, mask2);

		const __m128i mask1 = _mm_slli_epi16(mask3, 2);
		const __m128i r_s2 = _mm_srai_epi16(r, 2);
		r = _mm_blendv_epi8(r, r_s2, mask1);

		const __m128i mask0 = _mm_slli_epi16(mask3, 3);
		const __m128i r_s1 = _mm_srai_epi16(r, 1);
		r = _mm_blendv_epi8(r, r_s1, mask0);

		return s16x8(r, flag_native());

#else
		const __m128i mask3 = _mm_srai_epi16(_mm_slli_epi16(c.getn(), 12), 15);
		const __m128i mask2 = _mm_srai_epi16(_mm_slli_epi16(c.getn(), 13), 15);
		const __m128i mask1 = _mm_srai_epi16(_mm_slli_epi16(c.getn(), 14), 15);
		const __m128i mask0 = _mm_srai_epi16(_mm_slli_epi16(c.getn(), 15), 15);
		__m128i r = a.getn();

		const __m128i r_s8 = _mm_srai_epi16(r, 8);
		r = _mm_or_si128(_mm_and_si128(mask3, r_s8), _mm_andnot_si128(mask3, r));

		const __m128i r_s4 = _mm_srai_epi16(r, 4);
		r = _mm_or_si128(_mm_and_si128(mask2, r_s4), _mm_andnot_si128(mask2, r));

		const __m128i r_s2 = _mm_srai_epi16(r, 2);
		r = _mm_or_si128(_mm_and_si128(mask1, r_s2), _mm_andnot_si128(mask1, r));

		const __m128i r_s1 = _mm_srai_epi16(r, 1);
		r = _mm_or_si128(_mm_and_si128(mask0, r_s1), _mm_andnot_si128(mask0, r));

		return s16x8(r, flag_native());

#endif
#endif
	}

	/// \copydoc shr(const s64x2 a, const u64x2 c)
	inline u16x8 shr(const u16x8 a, const u16x8 c) {

#if __AVX2__ != 0
		const __m256i a32 = _mm256_cvtepu16_epi32(a.getn());
		const __m256i c32 = _mm256_cvtepu16_epi32(c.getn());
		const __m256i r32 = _mm256_srlv_epi32(a32, c32);
		const __m128i rl = _mm256_castsi256_si128(r32);
		const __m128i rh = _mm256_extracti128_si256(r32, 1);

		return u16x8(_mm_packus_epi32(rl, rh), flag_native());

#else
#if __SSE4_1__ != 0
		const __m128i mask3 = _mm_or_si128(_mm_slli_epi16(c.getn(), 12), _mm_slli_epi16(c.getn(), 4));
		__m128i r = a.getn();

		const __m128i r_s8 = _mm_srli_epi16(r, 8);
		r = _mm_blendv_epi8(r, r_s8, mask3);

		const __m128i mask2 = _mm_slli_epi16(mask3, 1);
		const __m128i r_s4 = _mm_srli_epi16(r, 4);
		r = _mm_blendv_epi8(r, r_s4, mask2);

		const __m128i mask1 = _mm_slli_epi16(mask3, 2);
		const __m128i r_s2 = _mm_srli_epi16(r, 2);
		r = _mm_blendv_epi8(r, r_s2, mask1);

		const __m128i mask0 = _mm_slli_epi16(mask3, 3);
		const __m128i r_s1 = _mm_srli_epi16(r, 1);
		r = _mm_blendv_epi8(r, r_s1, mask0);

		return u16x8(r, flag_native());

#else
		const __m128i mask3 = _mm_srai_epi16(_mm_slli_epi16(c.getn(), 12), 15);
		const __m128i mask2 = _mm_srai_epi16(_mm_slli_epi16(c.getn(), 13), 15);
		const __m128i mask1 = _mm_srai_epi16(_mm_slli_epi16(c.getn(), 14), 15);
		const __m128i mask0 = _mm_srai_epi16(_mm_slli_epi16(c.getn(), 15), 15);
		__m128i r = a.getn();

		const __m128i r_s8 = _mm_srli_epi16(r, 8);
		r = _mm_or_si128(_mm_and_si128(mask3, r_s8), _mm_andnot_si128(mask3, r));

		const __m128i r_s4 = _mm_srli_epi16(r, 4);
		r = _mm_or_si128(_mm_and_si128(mask2, r_s4), _mm_andnot_si128(mask2, r));

		const __m128i r_s2 = _mm_srli_epi16(r, 2);
		r = _mm_or_si128(_mm_and_si128(mask1, r_s2), _mm_andnot_si128(mask1, r));

		const __m128i r_s1 = _mm_srli_epi16(r, 1);
		r = _mm_or_si128(_mm_and_si128(mask0, r_s1), _mm_andnot_si128(mask0, r));

		return u16x8(r, flag_native());

#endif
#endif
	}

	/// Bitwise AND
	inline s64x2 operator & (const s64x2 a, const s64x2 b) {
		return s64x2(_mm_and_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator & (const s64x2 a, const s64x2 c)
	inline u64x2 operator & (const u64x2 a, const u64x2 b) {
		return u64x2(_mm_and_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator & (const s64x2 a, const s64x2 c)
	inline s32x4 operator & (const s32x4 a, const s32x4 b) {
		return s32x4(_mm_and_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator & (const s64x2 a, const s64x2 c)
	inline u32x4 operator & (const u32x4 a, const u32x4 b) {
		return u32x4(_mm_and_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator & (const s64x2 a, const s64x2 c)
	inline s16x8 operator & (const s16x8 a, const s16x8 b) {
		return s16x8(_mm_and_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator & (const s64x2 a, const s64x2 c)
	inline u16x8 operator & (const u16x8 a, const u16x8 b) {
		return u16x8(_mm_and_si128(a.getn(), b.getn()), flag_native());
	}

	/// Bitwise OR
	inline s64x2 operator | (const s64x2 a, const s64x2 b) {
		return s64x2(_mm_or_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator | (const s64x2 a, const s64x2 c)
	inline u64x2 operator | (const u64x2 a, const u64x2 b) {
		return u64x2(_mm_or_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator | (const s64x2 a, const s64x2 c)
	inline s32x4 operator | (const s32x4 a, const s32x4 b) {
		return s32x4(_mm_or_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator | (const s64x2 a, const s64x2 c)
	inline u32x4 operator | (const u32x4 a, const u32x4 b) {
		return u32x4(_mm_or_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator | (const s64x2 a, const s64x2 c)
	inline s16x8 operator | (const s16x8 a, const s16x8 b) {
		return s16x8(_mm_or_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator | (const s64x2 a, const s64x2 c)
	inline u16x8 operator | (const u16x8 a, const u16x8 b) {
		return u16x8(_mm_or_si128(a.getn(), b.getn()), flag_native());
	}

	/// Bitwise XOR
	inline s64x2 operator ^ (const s64x2 a, const s64x2 b) {
		return s64x2(_mm_xor_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator ^ (const s64x2 a, const s64x2 c)
	inline u64x2 operator ^ (const u64x2 a, const u64x2 b) {
		return u64x2(_mm_xor_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator ^ (const s64x2 a, const s64x2 c)
	inline s32x4 operator ^ (const s32x4 a, const s32x4 b) {
		return s32x4(_mm_xor_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator ^ (const s64x2 a, const s64x2 c)
	inline u32x4 operator ^ (const u32x4 a, const u32x4 b) {
		return u32x4(_mm_xor_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator ^ (const s64x2 a, const s64x2 c)
	inline s16x8 operator ^ (const s16x8 a, const s16x8 b) {
		return s16x8(_mm_xor_si128(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator ^ (const s64x2 a, const s64x2 c)
	inline u16x8 operator ^ (const u16x8 a, const u16x8 b) {
		return u16x8(_mm_xor_si128(a.getn(), b.getn()), flag_native());
	}

	/// Bitwise NOT
	inline s64x2 operator ~ (const s64x2 a) {
		const __m128i nmask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
		return s64x2(_mm_xor_si128(a.getn(), nmask), flag_native());
	}

	/// \copydoc operator ~ (const s64x2 a)
	inline u64x2 operator ~ (const u64x2 a) {
		const __m128i nmask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
		return u64x2(_mm_xor_si128(a.getn(), nmask), flag_native());
	}

	/// \copydoc operator ~ (const s64x2 a)
	inline s32x4 operator ~ (const s32x4 a) {
		const __m128i nmask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
		return s32x4(_mm_xor_si128(a.getn(), nmask), flag_native());
	}

	/// \copydoc operator ~ (const s64x2 a)
	inline u32x4 operator ~ (const u32x4 a) {
		const __m128i nmask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
		return u32x4(_mm_xor_si128(a.getn(), nmask), flag_native());
	}

	/// \copydoc operator ~ (const s64x2 a)
	inline s16x8 operator ~ (const s16x8 a) {
		const __m128i nmask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
		return s16x8(_mm_xor_si128(a.getn(), nmask), flag_native());
	}

	/// \copydoc operator ~ (const s64x2 a)
	inline u16x8 operator ~ (const u16x8 a) {
		const __m128i nmask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
		return u16x8(_mm_xor_si128(a.getn(), nmask), flag_native());
	}

	/// Transpose a 4x4 matrix.
	/// \param src0 first row of the source matrix
	/// \param src1 second row of the source matrix
	/// \param src2 third row of the source matrix
	/// \param src3 fourth row of the source matrix
	/// \param[out] dst0 first row of the result matrix
	/// \param[out] dst1 second row of the result matrix
	/// \param[out] dst2 third row of the result matrix
	/// \param[out] dst3 fourth row of the result matrix
	inline void transpose4x4(
		const f32x4 src0,
		const f32x4 src1,
		const f32x4 src2,
		const f32x4 src3,
		f32x4& dst0,
		f32x4& dst1,
		f32x4& dst2,
		f32x4& dst3) {

		const __m128 t0 = _mm_unpacklo_ps(src0.getn(), src1.getn());
		const __m128 t1 = _mm_unpacklo_ps(src2.getn(), src3.getn());
		const __m128 t2 = _mm_unpackhi_ps(src0.getn(), src1.getn());
		const __m128 t3 = _mm_unpackhi_ps(src2.getn(), src3.getn());
		dst0 = f32x4(_mm_movelh_ps(t0, t1), flag_native());
		dst1 = f32x4(_mm_movehl_ps(t1, t0), flag_native());
		dst2 = f32x4(_mm_movelh_ps(t2, t3), flag_native());
		dst3 = f32x4(_mm_movehl_ps(t3, t2), flag_native());
	}

	/// Transpose a 3x3 matrix.
	/// \param src0 first row of the source matrix
	/// \param src1 second row of the source matrix
	/// \param src2 third row of the source matrix
	/// \param[out] dst0 first row of the result matrix
	/// \param[out] dst1 second row of the result matrix
	/// \param[out] dst2 third row of the result matrix
	inline void transpose3x3(
		const f32x4 src0,
		const f32x4 src1,
		const f32x4 src2,
		f32x4& dst0,
		f32x4& dst1,
		f32x4& dst2) {

#if 0
		const __m128 l02 = _mm_unpacklo_ps(src0.getn(), src2.getn()); // src0.x, src2.x, src0.y, src2.y
		const __m128 h02 = _mm_unpackhi_ps(src0.getn(), src2.getn()); // src0.z, src2.z, src0.w, src2.w
		const __m128 l11 = _mm_unpacklo_ps(src1.getn(), src1.getn()); // src1.x, src1.x, src1.y, src1.y
		const __m128 h11 = _mm_unpackhi_ps(src1.getn(), src1.getn()); // src1.z, src1.z, src1.w, src1.w
		dst0 = f32x4(_mm_unpacklo_ps(l02, l11), flag_native()); // src0.x, src1.x, src2.x, (src1.x)
		dst1 = f32x4(_mm_unpackhi_ps(l02, l11), flag_native()); // src0.y, src1.y, src2.y, (src1.y)
		dst2 = f32x4(_mm_unpacklo_ps(h02, h11), flag_native()); // src0.z, src1.z, src2.z, (src1.z)

#else
		const __m128 l01 = _mm_unpacklo_ps(src0.getn(), src1.getn()); // src0.x, src1.x, src0.y, src1.y
		const __m128 h01 = _mm_unpackhi_ps(src0.getn(), src1.getn()); // src0.z, src1.z, src0.w, src1.w
		const __m128 l23 = _mm_unpacklo_ps(src2.getn(), src2.getn()); // src2.x, src2.x, src2.y, src2.y
		const __m128 h23 = _mm_unpackhi_ps(src2.getn(), src2.getn()); // src2.z, src2.z, src2.w, src2.w
		dst0 = f32x4(_mm_movelh_ps(l01, l23), flag_native()); // src0.x, src1.x, src2.x, (src2.x)
		dst1 = f32x4(_mm_movehl_ps(l23, l01), flag_native()); // src0.y, src1.y, src2.y, (src2.y)
		dst2 = f32x4(_mm_movelh_ps(h01, h23), flag_native()); // src0.z, src1.z, src2.z, (src2.z)

#endif
	}

	/// Square root of the argument.
	/// \param x non-negative argument
	/// \return square root of argument; NaN if argument is negative.
	inline f64x2 sqrt(
		const f64x2 x) {

		return f64x2(_mm_sqrt_pd(x.getn()), flag_native());
	}

	/// \copydoc sqrt(const f64x2)
	inline f32x4 sqrt(
		const f32x4 x) {

		return f32x4(_mm_sqrt_ps(x.getn()), flag_native());
	}

	/// Natural logarithm of the argument.
	/// \param x a positive argument
	/// \return natural logarithm of the argument; NaN if argument is non-positive.
	inline f32x4 log(
		const f32x4 x) {

		return f32x4(cephes_log(x.getn()), flag_native());
	}

	/// Natural exponent of the argument.
	/// \param x argument
	/// \return natural exponent of the argument.
	inline f32x4 exp(
		const f32x4 x) {

		return f32x4(cephes_exp(x.getn()), flag_native());
	}

	/// Raise non-negative base to the given power. Function uses natural exp and log
	/// to carry its computation, and thus is less exact than some other methods.
	/// \param x base argument
	/// \param y power argument
	/// \return base argument raised to the power; NaN if base is negative.
	inline f32x4 pow(
		const f32x4 x,
		const f32x4 y) {

		return f32x4(cephes_pow(x.getn(), y.getn()), flag_native());
	}

	/// Sine of the argument.
	/// \param x angle in radians; keep less than 8192 for best precision.
	/// \return sine of the argument.
	inline f32x4 sin(
		const f32x4 x) {

		return f32x4(cephes_sin(x.getn()), flag_native());
	}

	/// Cosine of the argument.
	/// \param x angle in radians; keep less than 8192 for best precision.
	/// \return cosine of the argument.
	inline f32x4 cos(
		const f32x4 x) {

		return f32x4(cephes_cos(x.getn()), flag_native());
	}

	/// Simultaneous sine and cosine of the arument.
	/// \param[in] x angle in radians; keep less than 8192 for best precision.
	/// \param[out] sin sine of the argument.
	/// \param[out] cos cosine of the argument.
	inline void sincos(
		const f32x4 x,
		f32x4& sin,
		f32x4& cos) {

		f32x4::native s, c;
		cephes_sincos(x.getn(), &s, &c);
		sin = f32x4(s, flag_native());
		cos = f32x4(c, flag_native());
	}

#endif
#if __AVX__ != 0
#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE != 0
	// note: no generic vectors with this compiler - use the 'native' vector type as the underlying vector type
	typedef native4 < f64, __m256d, __m256d > f64x4;
	typedef native4 < s64, __m256i, __m256i > s64x4;
	typedef native4 < u64, __m256i, __m256i > u64x4;
	typedef native8 < f32, __m256,  __m256  > f32x8;
	typedef native8 < s32, __m256i, __m256i > s32x8;
	typedef native8 < u32, __m256i, __m256i > u32x8;
	typedef native16< s16, __m256i, __m256i > s16x16;
	typedef native16< u16, __m256i, __m256i > u16x16;

#else
	typedef native4 < f64, f64 __attribute__ ((vector_size( 4 * sizeof(f64)))), __m256d > f64x4;
	typedef native4 < s64, s64 __attribute__ ((vector_size( 4 * sizeof(s64)))), __m256i > s64x4;
	typedef native4 < u64, u64 __attribute__ ((vector_size( 4 * sizeof(u64)))), __m256i > u64x4;
	typedef native8 < f32, f32 __attribute__ ((vector_size( 8 * sizeof(f32)))), __m256  > f32x8;
	typedef native8 < s32, s32 __attribute__ ((vector_size( 8 * sizeof(s32)))), __m256i > s32x8;
	typedef native8 < u32, u32 __attribute__ ((vector_size( 8 * sizeof(u32)))), __m256i > u32x8;
	typedef native16< s16, s16 __attribute__ ((vector_size(16 * sizeof(s16)))), __m256i > s16x16;
	typedef native16< u16, u16 __attribute__ ((vector_size(16 * sizeof(u16)))), __m256i > u16x16;

#endif
	#define NATIVE_F64X4    1
	#define NATIVE_S64X4    1
	#define NATIVE_U64X4    1
	#define NATIVE_F32X8    1
	#define NATIVE_S32X8    1
	#define NATIVE_U32X8    1
	#define NATIVE_S16X16   1
	#define NATIVE_U16X16   1

	// bitcasts ////////////////////////////////////////////////////////////////////////////////////

	/// Re-interpret the argument as the target type, preserving the original lane count and width.
	/// \param x argument of some source type.
	/// \return the re-interpretation of the argument as the return type
	inline f64x4 as_f64x4(const s64x4 x) {
		return f64x4(_mm256_castsi256_pd(x.getn()), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline f64x4 as_f64x4(const u64x4 x) {
		return f64x4(_mm256_castsi256_pd(x.getn()), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline s64x4 as_s64x4(const f64x4 x) {
		return s64x4(_mm256_castpd_si256(x.getn()), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline s64x4 as_s64x4(const u64x4 x) {
		return s64x4(x.getn(), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline u64x4 as_u64x4(const f64x4 x) {
		return u64x4(_mm256_castpd_si256(x.getn()), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline u64x4 as_u64x4(const s64x4 x) {
		return u64x4(x.getn(), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline f32x8 as_f32x8(const s32x8 x) {
		return f32x8(_mm256_castsi256_ps(x.getn()), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline f32x8 as_f32x8(const u32x8 x) {
		return f32x8(_mm256_castsi256_ps(x.getn()), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline s32x8 as_s32x8(const f32x8 x) {
		return s32x8(_mm256_castps_si256(x.getn()), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline s32x8 as_s32x8(const u32x8 x) {
		return s32x8(x.getn(), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline u32x8 as_u32x8(const f32x8 x) {
		return u32x8(_mm256_castps_si256(x.getn()), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline u32x8 as_u32x8(const s32x8 x) {
		return u32x8(x.getn(), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline s16x16 as_s16x16(const u16x16 x) {
		return s16x16(x.getn(), flag_native());
	}

	/// \copydoc as_f64x4(const s64x4)
	inline u16x16 as_u16x16(const s16x16 x) {
		return u16x16(x.getn(), flag_native());
	}

	// half-vector extraction //////////////////////////////////////////////////////////////////////

	/// Extract one half of the source, based on template argument 'index': 0 - lower half, 1 - upper half.
	/// \param x argument of some source type.
	/// \return the one half of the argument specified by index
	template < uint32_t INDEX >
	inline f64x2 get_f64x2(const f64x4 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
			return f64x2(_mm256_extractf128_pd(x.getn(), 1), flag_native());

		return f64x2(_mm256_castpd256_pd128(x.getn()), flag_native());
	}

	/// \copydoc get_f64x2(const f64x4)
	template < uint32_t INDEX >
	inline s64x2 get_s64x2(const s64x4 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
#if __AVX2__ != 0
			return s64x2(_mm256_extracti128_si256(x.getn(), 1), flag_native());

#else
			return s64x2(_mm256_extractf128_si256(x.getn(), 1), flag_native());

#endif
		return s64x2(_mm256_castsi256_si128(x.getn()), flag_native());
	}

	/// \copydoc get_f64x2(const f64x4)
	template < uint32_t INDEX >
	inline u64x2 get_u64x2(const u64x4 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
#if __AVX2__ != 0
			return u64x2(_mm256_extracti128_si256(x.getn(), 1), flag_native());

#else
			return u64x2(_mm256_extractf128_si256(x.getn(), 1), flag_native());

#endif
		return u64x2(_mm256_castsi256_si128(x.getn()), flag_native());
	}

	/// \copydoc get_f64x2(const f64x4)
	template < uint32_t INDEX >
	inline f32x4 get_f32x4(const f32x8 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
			return f32x4(_mm256_extractf128_ps(x.getn(), 1), flag_native());

		return f32x4(_mm256_castps256_ps128(x.getn()), flag_native());
	}

	/// \copydoc get_f64x2(const f64x4)
	template < uint32_t INDEX >
	inline s32x4 get_s32x4(const s32x8 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
#if __AVX2__ != 0
			return s32x4(_mm256_extracti128_si256(x.getn(), 1), flag_native());

#else
			return s32x4(_mm256_extractf128_si256(x.getn(), 1), flag_native());

#endif
		return s32x4(_mm256_castsi256_si128(x.getn()), flag_native());
	}

	/// \copydoc get_f64x2(const f64x4)
	template < uint32_t INDEX >
	inline u32x4 get_u32x4(const u32x8 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
#if __AVX2__ != 0
			return u32x4(_mm256_extracti128_si256(x.getn(), 1), flag_native());

#else
			return u32x4(_mm256_extractf128_si256(x.getn(), 1), flag_native());

#endif
		return u32x4(_mm256_castsi256_si128(x.getn()), flag_native());
	}

	/// \copydoc get_f64x2(const f64x4)
	template < uint32_t INDEX >
	inline s16x8 get_s16x8(const s16x16 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
#if __AVX2__ != 0
			return s16x8(_mm256_extracti128_si256(x.getn(), 1), flag_native());

#else
			return s16x8(_mm256_extractf128_si256(x.getn(), 1), flag_native());

#endif
		return s16x8(_mm256_castsi256_si128(x.getn()), flag_native());
	}

	/// \copydoc get_f64x2(const f64x4)
	template < uint32_t INDEX >
	inline u16x8 get_u16x8(const u16x16 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
#if __AVX2__ != 0
			return u16x8(_mm256_extracti128_si256(x.getn(), 1), flag_native());

#else
			return u16x8(_mm256_extractf128_si256(x.getn(), 1), flag_native());

#endif
		return u16x8(_mm256_castsi256_si128(x.getn()), flag_native());
	}

	// composition from half-vectors ///////////////////////////////////////////////////////////////

	/// Compose vector from two halves.
	/// \param x argument for the lower half.
	/// \param y argument for the upper half.
	/// \return vector comprised of the lower and upper halves.
	inline f64x4 get_f64x4(const f64x2 x, const f64x2 y) {
		return f64x4(_mm256_insertf128_pd(_mm256_castpd128_pd256(x.getn()), y.getn(), 1), flag_native());
	}

	/// \copydoc get_f64x4(const f64x2, const f64x2)
	inline s64x4 get_s64x4(const s64x2 x, const s64x2 y) {

#if __AVX2__ != 0
		return s64x4(_mm256_inserti128_si256(_mm256_castsi128_si256(x.getn()), y.getn(), 1), flag_native());

#else
		return s64x4(_mm256_insertf128_si256(_mm256_castsi128_si256(x.getn()), y.getn(), 1), flag_native());

#endif
	}

	/// \copydoc get_f64x4(const f64x2, const f64x2)
	inline u64x4 get_u64x4(const u64x2 x, const u64x2 y) {

#if __AVX2__ != 0
		return u64x4(_mm256_inserti128_si256(_mm256_castsi128_si256(x.getn()), y.getn(), 1), flag_native());

#else
		return u64x4(_mm256_insertf128_si256(_mm256_castsi128_si256(x.getn()), y.getn(), 1), flag_native());

#endif
	}

	/// \copydoc get_f64x4(const f64x2, const f64x2)
	inline f32x8 get_f32x8(const f32x4 x, const f32x4 y) {
		return f32x8(_mm256_insertf128_ps(_mm256_castps128_ps256(x.getn()), y.getn(), 1), flag_native());
	}

	/// \copydoc get_f64x4(const f64x2, const f64x2)
	inline s32x8 get_s32x8(const s32x4 x, const s32x4 y) {

#if __AVX2__ != 0
		return s32x8(_mm256_inserti128_si256(_mm256_castsi128_si256(x.getn()), y.getn(), 1), flag_native());

#else
		return s32x8(_mm256_insertf128_si256(_mm256_castsi128_si256(x.getn()), y.getn(), 1), flag_native());

#endif
	}

	/// \copydoc get_f64x4(const f64x2, const f64x2)
	inline u32x8 get_u32x8(const u32x4 x, const u32x4 y) {

#if __AVX2__ != 0
		return u32x8(_mm256_inserti128_si256(_mm256_castsi128_si256(x.getn()), y.getn(), 1), flag_native());

#else
		return u32x8(_mm256_insertf128_si256(_mm256_castsi128_si256(x.getn()), y.getn(), 1), flag_native());

#endif
	}

	/// \copydoc get_f64x4(const f64x2, const f64x2)
	inline s16x16 get_s16x16(const s16x8 x, const s16x8 y) {

#if __AVX2__ != 0
		return s16x16(_mm256_inserti128_si256(_mm256_castsi128_si256(x.getn()), y.getn(), 1), flag_native());

#else
		return s16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(x.getn()), y.getn(), 1), flag_native());

#endif
	}

	/// \copydoc get_f64x4(const f64x2, const f64x2)
	inline u16x16 get_u16x16(const u16x8 x, const u16x8 y) {

#if __AVX2__ != 0
		return u16x16(_mm256_inserti128_si256(_mm256_castsi128_si256(x.getn()), y.getn(), 1), flag_native());

#else
		return u16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(x.getn()), y.getn(), 1), flag_native());

#endif
	}

	// reduction predicates ////////////////////////////////////////////////////////////////////////

	/// All-reduction. True if all masked lanes of the specified integral-type vector are set to -1.
	/// \param lane_mask integral-type vector containing only 0's and -1's.
	/// \param bit_submask bitmask of the lanes of interest; a set bit indicates interest in the lane
	/// of the same index as the bit.
	/// \return true if all lanes, specified in the submask, are set to -1, false otherwise.
	inline bool all(const u64x4 lane_mask, const uint64_t bit_submask = 15) {
		assert(bit_submask > 0 && bit_submask < 16);

		return bit_submask == (bit_submask & _mm256_movemask_pd(_mm256_castsi256_pd(lane_mask.getn())));
	}

	/// \copydoc all(const u64x4, const uint64_t)
	inline bool all(const u32x8 lane_mask, const uint32_t bit_submask = 255) {
		assert(bit_submask > 0 && bit_submask < 256);

		return bit_submask == (bit_submask & _mm256_movemask_ps(_mm256_castsi256_ps(lane_mask.getn())));
	}

	/// Any-reduction. True if any of the masked lanes of the specified integral-type vector are set to -1.
	/// \param lane_mask integral-type vector containing only 0's and -1's.
	/// \param bit_submask bitmask of the lanes of interest; a set bit indicates interest in the lane
	/// of the same index as the bit.
	/// \return true if any of the lanes, specified in the submask, are set to -1, false otherwise.
	inline bool any(const u64x4 lane_mask, const uint64_t bit_submask = 15) {
		assert(bit_submask > 0 && bit_submask < 16);

		return 0 != (bit_submask & _mm256_movemask_pd(_mm256_castsi256_pd(lane_mask.getn())));
	}

	/// \copydoc any(const u64x4, const uint64_t)
	inline bool any(const u32x8 lane_mask, const uint32_t bit_submask = 255) {
		assert(bit_submask > 0 && bit_submask < 256);

		return 0 != (bit_submask & _mm256_movemask_ps(_mm256_castsi256_ps(lane_mask.getn())));
	}

	/// None-reduction. True if none of the masked lanes of the specified integral-type vector are set to -1.
	/// \param lane_mask integral-type vector containing only 0's and -1's.
	/// \param bit_submask bitmask of the lanes of interest; a set bit indicates interest in the lane
	/// of the same index as the bit.
	/// \return true if none of the lanes, specified in the submask, are set to -1, false otherwise.
	inline bool none(const u64x4 lane_mask, const uint64_t bit_submask = 15) {
		assert(bit_submask > 0 && bit_submask < 16);

		return 0 == (bit_submask & _mm256_movemask_pd(_mm256_castsi256_pd(lane_mask.getn())));
	}

	/// \copydoc none(const u64x4, const uint64_t)
	inline bool none(const u32x8 lane_mask, const uint32_t bit_submask = 255) {
		assert(bit_submask > 0 && bit_submask < 256);

		return 0 == (bit_submask & _mm256_movemask_ps(_mm256_castsi256_ps(lane_mask.getn())));
	}

#if __AVX2__ != 0
	/// \copydoc all(const u64x4, const uint64_t)
	inline bool all(const u16x16 lane_mask, const uint32_t bit_submask = 65535) {
		assert(bit_submask > 0 && bit_submask < 65536);

		return bit_submask == (bit_submask & deinterleaveBits16(_mm256_movemask_epi8(lane_mask.getn())));
	}

	/// \copydoc any(const u64x4, const uint64_t)
	inline bool any(const u16x16 lane_mask, const uint32_t bit_submask = 65535) {
		assert(bit_submask > 0 && bit_submask < 65536);

		return 0 != (bit_submask & deinterleaveBits16(_mm256_movemask_epi8(lane_mask.getn())));
	}

	/// \copydoc none(const u64x4, const uint64_t)
	inline bool none(const u16x16 lane_mask, const uint32_t bit_submask = 65535) {
		assert(bit_submask > 0 && bit_submask < 65536);

		return 0 == (bit_submask & deinterleaveBits16(_mm256_movemask_epi8(lane_mask.getn())));
	}

#else
	// internal helper - do not use directly
	inline uint32_t movemask_base(const __m256i x) {
		const __m128i x0 = _mm256_castsi256_si128(x);
		const __m128i x1 = _mm256_extractf128_si256(x, 1);
		const int r0 = _mm_movemask_epi8(x0);
		const int r1 = _mm_movemask_epi8(x1);
		const uint32_t r = r0 & 0xffff | (r1 & 0xffff) << 16;
		return r;
	}

	/// \copydoc all(const u64x4, const uint64_t)
	inline bool all(const u16x16 lane_mask, const uint32_t bit_submask = 65535) {
		assert(bit_submask > 0 && bit_submask < 65536);

		return bit_submask == (bit_submask & deinterleaveBits16(movemask_base(lane_mask.getn())));
	}

	/// \copydoc any(const u64x4, const uint64_t)
	inline bool any(const u16x16 lane_mask, const uint32_t bit_submask = 65535) {
		assert(bit_submask > 0 && bit_submask < 65536);

		return 0 != (bit_submask & deinterleaveBits16(movemask_base(lane_mask.getn())));
	}

	/// \copydoc none(const u64x4, const uint64_t)
	inline bool none(const u16x16 lane_mask, const uint32_t bit_submask = 65535) {
		assert(bit_submask > 0 && bit_submask < 65536);

		return 0 == (bit_submask & deinterleaveBits16(movemask_base(lane_mask.getn())));
	}

#endif
	// extrema /////////////////////////////////////////////////////////////////////////////////////

	/// Compute vector minimum. Return a vector where each lane is the minimum of the corresponding
	/// lanes of the argument vectors.
	/// \param a first argument.
	/// \param b second argument.
	/// \return a vector of the same type as the arguments, where each lane is the minimum
	/// of the corresponding lanes from the two arguments.
	inline f64x4 min(const f64x4 a, const f64x4 b) {
		return f64x4(_mm256_min_pd(a.getn(), b.getn()), flag_native());
	}

	/// Compute vector maximum. Return a vector where each lane is the maximum of the corresponding
	/// lanes of the argument vectors.
	/// \param a first argument.
	/// \param b second argument.
	/// \return a vector of the same type as the arguments, where each lane is the maximum
	/// of the corresponding lanes from the two arguments.
	inline f64x4 max(const f64x4 a, const f64x4 b) {
		return f64x4(_mm256_max_pd(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc min(const f64x4, const f64x4)
	inline f32x8 min(const f32x8 a, const f32x8 b) {
		return f32x8(_mm256_min_ps(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc max(const f64x4, const f64x4)
	inline f32x8 max(const f32x8 a, const f32x8 b) {
		return f32x8(_mm256_max_ps(a.getn(), b.getn()), flag_native());
	}

#if __AVX2__ != 0
	/// \copydoc min(const f64x4, const f64x4)
	inline s32x8 min(const s32x8 a, const s32x8 b) {
		return s32x8(_mm256_min_epi32(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc max(const f64x4, const f64x4)
	inline s32x8 max(const s32x8 a, const s32x8 b) {
		return s32x8(_mm256_max_epi32(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc min(const f64x4, const f64x4)
	inline u32x8 min(const u32x8 a, const u32x8 b) {
		return u32x8(_mm256_min_epu32(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc max(const f64x4, const f64x4)
	inline u32x8 max(const u32x8 a, const u32x8 b) {
		return u32x8(_mm256_max_epu32(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc min(const f64x4, const f64x4)
	inline s16x16 min(const s16x16 a, const s16x16 b) {
		return s16x16(_mm256_min_epi16(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc max(const f64x4, const f64x4)
	inline s16x16 max(const s16x16 a, const s16x16 b) {
		return s16x16(_mm256_max_epi16(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc min(const f64x4, const f64x4)
	inline u16x16 min(const u16x16 a, const u16x16 b) {
		return u16x16(_mm256_min_epu16(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc max(const f64x4, const f64x4)
	inline u16x16 max(const u16x16 a, const u16x16 b) {
		return u16x16(_mm256_max_epu16(a.getn(), b.getn()), flag_native());
	}

#else
	/// \copydoc min(const f64x4, const f64x4)
	inline s32x8 min(const s32x8 a, const s32x8 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i min0 = _mm_min_epi32(a0, b0);
		const __m128i min1 = _mm_min_epi32(a1, b1);
		const __m256i min = _mm256_castsi128_si256(min0);

		return s32x8(_mm256_insertf128_si256(min, min1, 1), flag_native());
	}

	/// \copydoc max(const f64x4, const f64x4)
	inline s32x8 max(const s32x8 a, const s32x8 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i max0 = _mm_max_epi32(a0, b0);
		const __m128i max1 = _mm_max_epi32(a1, b1);
		const __m256i max = _mm256_castsi128_si256(max0);

		return s32x8(_mm256_insertf128_si256(max, max1, 1), flag_native());
	}

	/// \copydoc min(const f64x4, const f64x4)
	inline u32x8 min(const u32x8 a, const u32x8 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i min0 = _mm_min_epu32(a0, b0);
		const __m128i min1 = _mm_min_epu32(a1, b1);
		const __m256i min = _mm256_castsi128_si256(min0);

		return u32x8(_mm256_insertf128_si256(min, min1, 1), flag_native());
	}

	/// \copydoc max(const f64x4, const f64x4)
	inline u32x8 max(const u32x8 a, const u32x8 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i max0 = _mm_max_epu32(a0, b0);
		const __m128i max1 = _mm_max_epu32(a1, b1);
		const __m256i max = _mm256_castsi128_si256(max0);

		return u32x8(_mm256_insertf128_si256(max, max1, 1), flag_native());
	}

	/// \copydoc min(const f64x4, const f64x4)
	inline s16x16 min(const s16x16 a, const s16x16 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i min0 = _mm_min_epi16(a0, b0);
		const __m128i min1 = _mm_min_epi16(a1, b1);
		const __m256i min = _mm256_castsi128_si256(min0);

		return s16x16(_mm256_insertf128_si256(min, min1, 1), flag_native());
	}

	/// \copydoc max(const f64x4, const f64x4)
	inline s16x16 max(const s16x16 a, const s16x16 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i max0 = _mm_max_epi16(a0, b0);
		const __m128i max1 = _mm_max_epi16(a1, b1);
		const __m256i max = _mm256_castsi128_si256(max0);

		return s16x16(_mm256_insertf128_si256(max, max1, 1), flag_native());
	}

	/// \copydoc min(const f64x4, const f64x4)
	inline u16x16 min(const u16x16 a, const u16x16 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i min0 = _mm_min_epu16(a0, b0);
		const __m128i min1 = _mm_min_epu16(a1, b1);
		const __m256i min = _mm256_castsi128_si256(min0);

		return u16x16(_mm256_insertf128_si256(min, min1, 1), flag_native());
	}

	/// \copydoc max(const f64x4, const f64x4)
	inline u16x16 max(const u16x16 a, const u16x16 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i max0 = _mm_max_epu16(a0, b0);
		const __m128i max1 = _mm_max_epu16(a1, b1);
		const __m256i max = _mm256_castsi128_si256(max0);

		return u16x16(_mm256_insertf128_si256(max, max1, 1), flag_native());
	}

#endif
	// mask off lanes as nil ///////////////////////////////////////////////////////////////////////

	/// Mask off lanes of a vector as nil, according to a lane mask vector.
	/// \param x source vector.
	/// \param lane_mask integral-type vector of the same lane count and width as the source vector,
	/// whose lanes have values of -1 or 0, preserving or zeroing the resulting lanes, respectively.
	/// \return vector of the same type as the source, whose lanes contain either the value of the
	/// corresponding source lanes or zero.
	inline f64x4 mask(const f64x4 x, const u64x4 lane_mask) {
		return f64x4(_mm256_and_pd(_mm256_castsi256_pd(lane_mask.getn()), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x4, const u64x4)
	inline f32x8 mask(const f32x8 x, const u32x8 lane_mask) {
		return f32x8(_mm256_and_ps(_mm256_castsi256_ps(lane_mask.getn()), x.getn()), flag_native());
	}

#if __AVX2__ != 0
	/// \copydoc mask(const f64x4, const u64x4)
	inline s64x4 mask(const s64x4 x, const u64x4 lane_mask) {
		return s64x4(_mm256_and_si256(lane_mask.getn(), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x4, const u64x4)
	inline u64x4 mask(const u64x4 x, const u64x4 lane_mask) {
		return u64x4(_mm256_and_si256(lane_mask.getn(), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x4, const u64x4)
	inline s32x8 mask(const s32x8 x, const u32x8 lane_mask) {
		return s32x8(_mm256_and_si256(lane_mask.getn(), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x4, const u64x4)
	inline u32x8 mask(const u32x8 x, const u32x8 lane_mask) {
		return u32x8(_mm256_and_si256(lane_mask.getn(), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x4, const u64x4)
	inline s16x16 mask(const s16x16 x, const u16x16 lane_mask) {
		return s16x16(_mm256_and_si256(lane_mask.getn(), x.getn()), flag_native());
	}

	/// \copydoc mask(const f64x4, const u64x4)
	inline u16x16 mask(const u16x16 x, const u16x16 lane_mask) {
		return u16x16(_mm256_and_si256(lane_mask.getn(), x.getn()), flag_native());
	}

#else
	// internal helper - do not use directly
	inline __m256i mask_base(const __m256i a, const __m256i b) {
		const __m128i a0 = _mm256_castsi256_si128(a);
		const __m128i b0 = _mm256_castsi256_si128(b);
		const __m128i a1 = _mm256_extractf128_si256(a, 1);
		const __m128i b1 = _mm256_extractf128_si256(b, 1);
		const __m128i r0 = _mm_and_si128(a0, b0);
		const __m128i r1 = _mm_and_si128(a1, b1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return r;
	}

	/// \copydoc mask(const f64x4, const u64x4)
	inline s64x4 mask(const s64x4 x, const u64x4 lane_mask) {
		return s64x4(mask_base(x.getn(), lane_mask.getn()), flag_native());
	}

	/// \copydoc mask(const f64x4, const u64x4)
	inline u64x4 mask(const u64x4 x, const u64x4 lane_mask) {
		return u64x4(mask_base(x.getn(), lane_mask.getn()), flag_native());
	}

	/// \copydoc mask(const f64x4, const u64x4)
	inline s32x8 mask(const s32x8 x, const u32x8 lane_mask) {
		return s32x8(mask_base(x.getn(), lane_mask.getn()), flag_native());
	}

	/// \copydoc mask(const f64x4, const u64x4)
	inline u32x8 mask(const u32x8 x, const u32x8 lane_mask) {
		return u32x8(mask_base(x.getn(), lane_mask.getn()), flag_native());
	}

	/// \copydoc mask(const f64x4, const u64x4)
	inline s16x16 mask(const s16x16 x, const u16x16 lane_mask) {
		return s16x16(mask_base(x.getn(), lane_mask.getn()), flag_native());
	}

	/// \copydoc mask(const f64x4, const u64x4)
	inline u16x16 mask(const u16x16 x, const u16x16 lane_mask) {
		return u16x16(mask_base(x.getn(), lane_mask.getn()), flag_native());
	}

#endif
	// combine lanes by bitmask literal ////////////////////////////////////////////////////////////

	/// Combine the lanes from two arguments according to a bitmask literal; a zero bit in the literal
	/// causes the selection of a lane from the 1st argument, a one bit - from the 2nd argument.
	/// \param a first source.
	/// \param b second source.
	/// \return a vector of the same lane type and count as the source vectors.
	template < uint32_t SELECTOR >
	inline f64x4 select(const f64x4 a, const f64x4 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 16 > assert_selector;

		return f64x4(_mm256_blend_pd(a.getn(), b.getn(), SELECTOR), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4)
	template < uint32_t SELECTOR >
	inline f32x8 select(const f32x8 a, const f32x8 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 256 > assert_selector;

		return f32x8(_mm256_blend_ps(a.getn(), b.getn(), SELECTOR), flag_native());
	}

#if __AVX2__ != 0
	/// \copydoc select(const f64x4, const f64x4)
	template < uint32_t SELECTOR >
	inline s32x8 select(const s32x8 a, const s32x8 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 256 > assert_selector;

		return s32x8(_mm256_blend_epi32(a.getn(), b.getn(), SELECTOR), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4)
	template < uint32_t SELECTOR >
	inline u32x8 select(const u32x8 a, const u32x8 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 256 > assert_selector;

		return u32x8(_mm256_blend_epi32(a.getn(), b.getn(), SELECTOR), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4)
	template < uint32_t SELECTOR >
	inline s16x16 select(const s16x16 a, const s16x16 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 65536 > assert_selector;

		// the instruction "vpblendw" re-uses the same 8-bit mask for the upper 8 lanes
		if (uint8_t(SELECTOR) == uint8_t(SELECTOR >> 8))
			return s16x16(_mm256_blend_epi16(a.getn(), b.getn(), SELECTOR & 0xff), flag_native());

		const int16_t u = -1;
		const __m256i vmsk = _mm256_set_epi16(
			SELECTOR & 0x8000 ? u : 0, SELECTOR & 0x4000 ? u : 0, SELECTOR & 0x2000 ? u : 0, SELECTOR & 0x1000 ? u : 0,
			SELECTOR & 0x0800 ? u : 0, SELECTOR & 0x0400 ? u : 0, SELECTOR & 0x0200 ? u : 0, SELECTOR & 0x0100 ? u : 0,
			SELECTOR & 0x0080 ? u : 0, SELECTOR & 0x0040 ? u : 0, SELECTOR & 0x0020 ? u : 0, SELECTOR & 0x0010 ? u : 0,
			SELECTOR & 0x0008 ? u : 0, SELECTOR & 0x0004 ? u : 0, SELECTOR & 0x0002 ? u : 0, SELECTOR & 0x0001 ? u : 0);
		return s16x16(_mm256_blendv_epi8(a.getn(), b.getn(), vmsk), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4)
	template < uint32_t SELECTOR >
	inline u16x16 select(const u16x16 a, const u16x16 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 65536 > assert_selector;

		// the instruction "vpblendw" re-uses the same 8-bit mask for the upper 8 lanes
		if (uint8_t(SELECTOR) == uint8_t(SELECTOR >> 8))
			return u16x16(_mm256_blend_epi16(a.getn(), b.getn(), SELECTOR & 0xff), flag_native());

		const int16_t u = -1;
		const __m256i vmsk = _mm256_set_epi16(
			SELECTOR & 0x8000 ? u : 0, SELECTOR & 0x4000 ? u : 0, SELECTOR & 0x2000 ? u : 0, SELECTOR & 0x1000 ? u : 0,
			SELECTOR & 0x0800 ? u : 0, SELECTOR & 0x0400 ? u : 0, SELECTOR & 0x0200 ? u : 0, SELECTOR & 0x0100 ? u : 0,
			SELECTOR & 0x0080 ? u : 0, SELECTOR & 0x0040 ? u : 0, SELECTOR & 0x0020 ? u : 0, SELECTOR & 0x0010 ? u : 0,
			SELECTOR & 0x0008 ? u : 0, SELECTOR & 0x0004 ? u : 0, SELECTOR & 0x0002 ? u : 0, SELECTOR & 0x0001 ? u : 0);
		return u16x16(_mm256_blendv_epi8(a.getn(), b.getn(), vmsk), flag_native());
	}

#else
	// internal helper - do not use directly
	template < uint32_t SELECTOR >
	inline __m256i select_base(const __m256i a, const __m256i b) {
		const __m128i a0 = _mm256_castsi256_si128(a);
		const __m128i b0 = _mm256_castsi256_si128(b);
		const __m128i a1 = _mm256_extractf128_si256(a, 1);
		const __m128i b1 = _mm256_extractf128_si256(b, 1);
		const __m128i r0 = _mm_blend_epi16(a0, b0, SELECTOR & 0xff);
		const __m128i r1 = _mm_blend_epi16(a1, b1, SELECTOR >> 8 & 0xff);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return r;
	}

	/// \copydoc select(const f64x4, const f64x4)
	template < uint32_t SELECTOR >
	inline s32x8 select(const s32x8 a, const s32x8 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 256 > assert_selector;

		// note: consider a "vblendps" instead - check if crossing the fp/int domain boundary is not actually faster
		const uint32_t x0 = SELECTOR | SELECTOR << 16;
		const uint32_t x1 = (x0 | (x0 << 4)) & 0x0f0f0f0f;
		const uint32_t x2 = (x1 | (x1 << 2)) & 0x33333333;
		const uint32_t x3 = (x2 | (x2 << 1)) & 0x55555555;
		const uint32_t interleavedBits8 = uint16_t(x3) | uint16_t(x3 >> 15);

		return s32x8(select_base< interleavedBits8 >(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4)
	template < uint32_t SELECTOR >
	inline u32x8 select(const u32x8 a, const u32x8 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 256 > assert_selector;

		// note: consider a "vblendps" instead - check if crossing the fp/int domain boundary is not actually faster
		const uint32_t x0 = SELECTOR | SELECTOR << 16;
		const uint32_t x1 = (x0 | (x0 << 4)) & 0x0f0f0f0f;
		const uint32_t x2 = (x1 | (x1 << 2)) & 0x33333333;
		const uint32_t x3 = (x2 | (x2 << 1)) & 0x55555555;
		const uint32_t interleavedBits8 = uint16_t(x3) | uint16_t(x3 >> 15);

		return u32x8(select_base< interleavedBits8 >(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4)
	template < uint32_t SELECTOR >
	inline s16x16 select(const s16x16 a, const s16x16 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 65536 > assert_selector;

		return s16x16(select_base< SELECTOR >(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4)
	template < uint32_t SELECTOR >
	inline u16x16 select(const u16x16 a, const u16x16 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 65536 > assert_selector;

		return u16x16(select_base< SELECTOR >(a.getn(), b.getn()), flag_native());
	}

#endif
	// combine lanes by lane mask //////////////////////////////////////////////////////////////////

	/// Combine the lanes from two arguments according to a lane-mask vector; a lane of 0 in the lane mask
	/// causes the selection of a lane from the 1st argument, a lane of -1 - from the 2nd argument.
	/// \param a first source.
	/// \param b second source.
	/// \param selector lane selection control.
	/// \return a vector of the same lane type and count as the source vectors.
	inline f64x4 select(const f64x4 a, const f64x4 b, const u64x4 selector) {
		return f64x4(_mm256_blendv_pd(a.getn(), b.getn(), _mm256_castsi256_pd(selector.getn())), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline f32x8 select(const f32x8 a, const f32x8 b, const u32x8 selector) {
		return f32x8(_mm256_blendv_ps(a.getn(), b.getn(), _mm256_castsi256_ps(selector.getn())), flag_native());
	}

#if __AVX2__ != 0
	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline s64x4 select(const s64x4 a, const s64x4 b, const u64x4 selector) {
		return s64x4(_mm256_blendv_epi8(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline u64x4 select(const u64x4 a, const u64x4 b, const u64x4 selector) {
		return u64x4(_mm256_blendv_epi8(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline s32x8 select(const s32x8 a, const s32x8 b, const u32x8 selector) {
		return s32x8(_mm256_blendv_epi8(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline u32x8 select(const u32x8 a, const u32x8 b, const u32x8 selector) {
		return u32x8(_mm256_blendv_epi8(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline s16x16 select(const s16x16 a, const s16x16 b, const u16x16 selector) {
		return s16x16(_mm256_blendv_epi8(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline u16x16 select(const u16x16 a, const u16x16 b, const u16x16 selector) {
		return u16x16(_mm256_blendv_epi8(a.getn(), b.getn(), selector.getn()), flag_native());
	}

#else
	// internal helper - do not use directly
	inline __m256i select_base(const __m256i a, const __m256i b, const __m256i selector) {
		const __m128i a0 = _mm256_castsi256_si128(a);
		const __m128i b0 = _mm256_castsi256_si128(b);
		const __m128i s0 = _mm256_castsi256_si128(selector);
		const __m128i a1 = _mm256_extractf128_si256(a, 1);
		const __m128i b1 = _mm256_extractf128_si256(b, 1);
		const __m128i s1 = _mm256_extractf128_si256(selector, 1);
		const __m128i r0 = _mm_blendv_epi8(a0, b0, s0);
		const __m128i r1 = _mm_blendv_epi8(a1, b1, s1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return r;
	}

	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline s64x4 select(const s64x4 a, const s64x4 b, const u64x4 selector) {
		return s64x4(select_base(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline u64x4 select(const u64x4 a, const u64x4 b, const u64x4 selector) {
		return u64x4(select_base(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline s32x8 select(const s32x8 a, const s32x8 b, const u32x8 selector) {
		return s32x8(select_base(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline u32x8 select(const u32x8 a, const u32x8 b, const u32x8 selector) {
		return u32x8(select_base(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline s16x16 select(const s16x16 a, const s16x16 b, const u16x16 selector) {
		return s16x16(select_base(a.getn(), b.getn(), selector.getn()), flag_native());
	}

	/// \copydoc select(const f64x4, const f64x4, const u64x4)
	inline u16x16 select(const u16x16 a, const u16x16 b, const u16x16 selector) {
		return u16x16(select_base(a.getn(), b.getn(), selector.getn()), flag_native());
	}

#endif
	// absolute value //////////////////////////////////////////////////////////////////////////////

	/// Compute the absolute value of the source vector.
	/// \param x source vector.
	/// \return a vector of the same lane type and count as the source,
	/// whose lanes contain the absolute values of the source lanes.
	inline f64x4 abs(const f64x4 x) {
		return f64x4(_mm256_andnot_pd(_mm256_set1_pd(-0.0), x.getn()), flag_native());
	}

	/// \copydoc abs(const f64x4)
	inline f32x8 abs(const f32x8 x) {
		return f32x8(_mm256_andnot_ps(_mm256_set1_ps(-0.f), x.getn()), flag_native());
	}

#if __AVX2__ != 0
	/// \copydoc abs(const f64x4)
	inline s32x8 abs(const s32x8 x) {
		return s32x8(_mm256_abs_epi32(x.getn()), flag_native());
	}

	/// \copydoc abs(const f64x4)
	inline s16x16 abs(const s16x16 x) {
		return s16x16(_mm256_abs_epi16(x.getn()), flag_native());
	}

#else
	/// \copydoc abs(const f64x4)
	inline s32x8 abs(const s32x8 x) {
		const __m128i x0 = _mm256_castsi256_si128(x.getn());
		const __m128i x1 = _mm256_extractf128_si256(x.getn(), 1);
		const __m128i r0 = _mm_abs_epi32(x0);
		const __m128i r1 = _mm_abs_epi32(x1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return s32x8(r, flag_native());
	}

	/// \copydoc abs(const f64x4)
	inline s16x16 abs(const s16x16 x) {
		const __m128i x0 = _mm256_castsi256_si128(x.getn());
		const __m128i x1 = _mm256_extractf128_si256(x.getn(), 1);
		const __m128i r0 = _mm_abs_epi16(x0);
		const __m128i r1 = _mm_abs_epi16(x1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return s16x16(r, flag_native());
	}

#endif
#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE != 0 || COMPILER_QUIRK_0003_RELATIONAL_OPS != 0
	/// Compare two vectors for equality.
	/// \param a first source vector.
	/// \param b second source vector.
	/// \return an integral-type vector of the same lane count and width as the sources,
	/// containing -1 in the lanes where the condition was satisfied, and 0 in the remaining lanes.
	inline u64x4 operator ==(const f64x4 a, const f64x4 b) {
		return u64x4(_mm256_castpd_si256(_mm256_cmp_pd(a.getn(), b.getn(), _CMP_EQ_OQ)), flag_native());
	}

	/// \copydoc operator ==(const f64x4, const f64x4)
	inline u32x8 operator ==(const f32x8 a, const f32x8 b) {
		return u32x8(_mm256_castps_si256(_mm256_cmp_ps(a.getn(), b.getn(), _CMP_EQ_OQ)), flag_native());
	}

#if __AVX2__ != 0
	/// \copydoc operator ==(const f64x4, const f64x4)
	inline u64x4 operator ==(const s64x4 a, const s64x4 b) {
		return u64x4(_mm256_cmpeq_epi64(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator ==(const f64x4, const f64x4)
	inline u32x8 operator ==(const s32x8 a, const s32x8 b) {
		return u32x8(_mm256_cmpeq_epi32(a.getn(), b.getn()), flag_native());
	}

	/// \copydoc operator ==(const f64x4, const f64x4)
	inline u16x16 operator ==(const s16x16 a, const s16x16 b) {
		return u16x16(_mm256_cmpeq_epi16(a.getn(), b.getn()), flag_native());
	}

#else
	/// \copydoc operator ==(const f64x4, const f64x4)
	inline u64x4 operator ==(const s64x4 a, const s64x4 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_cmpeq_epi64(a0, b0);
		const __m128i r1 = _mm_cmpeq_epi64(a1, b1);
		const __m256i r = _mm256_castsi128_si256(r0);
		return u64x4(_mm256_insertf128_si256(r, r1, 1), flag_native());
	}

	/// \copydoc operator ==(const f64x4, const f64x4)
	inline u32x8 operator ==(const s32x8 a, const s32x8 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_cmpeq_epi32(a0, b0);
		const __m128i r1 = _mm_cmpeq_epi32(a1, b1);
		const __m256i r = _mm256_castsi128_si256(r0);

		return u32x8(_mm256_insertf128_si256(r, r1, 1), flag_native());
	}

	/// \copydoc operator ==(const f64x4, const f64x4)
	inline u16x16 operator ==(const s16x16 a, const s16x16 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_cmpeq_epi16(a0, b0);
		const __m128i r1 = _mm_cmpeq_epi16(a1, b1);
		const __m256i r = _mm256_castsi128_si256(r0);

		return u16x16(_mm256_insertf128_si256(r, r1, 1), flag_native());
	}

#endif
	/// \copydoc operator ==(const f64x4, const f64x4)
	inline u64x4 operator ==(const u64x4 a, const u64x4 b) {
		return as_s64x4(a) == as_s64x4(b);
	}

	/// \copydoc operator ==(const f64x4, const f64x4)
	inline u32x8 operator ==(const u32x8 a, const u32x8 b) {
		return as_s32x8(a) == as_s32x8(b);
	}

	/// \copydoc operator ==(const f64x4, const f64x4)
	inline u16x16 operator ==(const u16x16 a, const u16x16 b) {
		return as_s16x16(a) == as_s16x16(b);
	}

	/// Compare two vectors for inequality.
	/// \param a first source vector.
	/// \param b second source vector.
	/// \return an integral-type vector of the same lane count and width as the sources,
	/// containing -1 in the lanes where the condition was satisfied, and 0 in the remaining lanes.
	inline u64x4 operator !=(const f64x4 a, const f64x4 b) {
		return u64x4(_mm256_castpd_si256(_mm256_cmp_pd(a.getn(), b.getn(), _CMP_NEQ_UQ)), flag_native());
	}

	/// \copydoc operator !=(const f64x4, const f64x4)
	inline u32x8 operator !=(const f32x8 a, const f32x8 b) {
		return u32x8(_mm256_castps_si256(_mm256_cmp_ps(a.getn(), b.getn(), _CMP_NEQ_UQ)), flag_native());
	}

#if __AVX2__ != 0
	/// \copydoc operator !=(const f64x4, const f64x4)
	inline u64x4 operator !=(const s64x4 a, const s64x4 b) {
		return u64x4(_mm256_xor_si256(_mm256_cmpeq_epi64(a.getn(), b.getn()), _mm256_set1_epi64x(-1)), flag_native());
	}

	/// \copydoc operator !=(const f64x4, const f64x4)
	inline u32x8 operator !=(const s32x8 a, const s32x8 b) {
		return u32x8(_mm256_xor_si256(_mm256_cmpeq_epi32(a.getn(), b.getn()), _mm256_set1_epi64x(-1)), flag_native());
	}

	/// \copydoc operator !=(const f64x4, const f64x4)
	inline u16x16 operator !=(const s16x16 a, const s16x16 b) {
		return u16x16(_mm256_xor_si256(_mm256_cmpeq_epi16(a.getn(), b.getn()), _mm256_set1_epi64x(-1)), flag_native());
	}

#else
	/// \copydoc operator !=(const f64x4, const f64x4)
	inline u64x4 operator !=(const s64x4 a, const s64x4 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_xor_si128(_mm_cmpeq_epi64(a0, b0), _mm_set1_epi64x(-1));
		const __m128i r1 = _mm_xor_si128(_mm_cmpeq_epi64(a1, b1), _mm_set1_epi64x(-1));
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return u64x4(r, flag_native());
	}

	/// \copydoc operator !=(const f64x4, const f64x4)
	inline u32x8 operator !=(const s32x8 a, const s32x8 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_xor_si128(_mm_cmpeq_epi32(a0, b0), _mm_set1_epi64x(-1));
		const __m128i r1 = _mm_xor_si128(_mm_cmpeq_epi32(a1, b1), _mm_set1_epi64x(-1));
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return u32x8(r, flag_native());
	}

	/// \copydoc operator !=(const f64x4, const f64x4)
	inline u16x16 operator !=(const s16x16 a, const s16x16 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_xor_si128(_mm_cmpeq_epi16(a0, b0), _mm_set1_epi64x(-1));
		const __m128i r1 = _mm_xor_si128(_mm_cmpeq_epi16(a1, b1), _mm_set1_epi64x(-1));
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return u16x16(r, flag_native());
	}

#endif
	/// \copydoc operator !=(const f64x4, const f64x4)
	inline u64x4 operator !=(const u64x4 a, const u64x4 b) {
		return as_s64x4(a) != as_s64x4(b);
	}

	/// \copydoc operator !=(const f64x4, const f64x4)
	inline u32x8 operator !=(const u32x8 a, const u32x8 b) {
		return as_s32x8(a) != as_s32x8(b);
	}

	/// \copydoc operator !=(const f64x4, const f64x4)
	inline u16x16 operator !=(const u16x16 a, const u16x16 b) {
		return as_s16x16(a) != as_s16x16(b);
	}

	/// Compare two vectors for less-than.
	/// \param a first source vector.
	/// \param b second source vector.
	/// \return an integral-type vector of the same lane count and width as the sources,
	/// containing -1 in the lanes where the condition was satisfied, and 0 in the remaining lanes.
	inline u64x4 operator <(const f64x4 a, const f64x4 b) {
		return u64x4(_mm256_castpd_si256(_mm256_cmp_pd(a.getn(), b.getn(), _CMP_LT_OQ)), flag_native());
	}

	/// \copydoc operator <(const f64x4, const f64x4)
	inline u32x8 operator <(const f32x8 a, const f32x8 b) {
		return u32x8(_mm256_castps_si256(_mm256_cmp_ps(a.getn(), b.getn(), _CMP_LT_OQ)), flag_native());
	}

#if __AVX2__ != 0
	/// \copydoc operator <(const f64x4, const f64x4)
	inline u64x4 operator <(const s64x4 a, const s64x4 b) {
		return u64x4(_mm256_cmpgt_epi64(b.getn(), a.getn()), flag_native());
	}

	/// \copydoc operator <(const f64x4, const f64x4)
	inline u32x8 operator <(const s32x8 a, const s32x8 b) {
		return u32x8(_mm256_cmpgt_epi32(b.getn(), a.getn()), flag_native());
	}

	/// \copydoc operator <(const f64x4, const f64x4)
	inline u16x16 operator <(const s16x16 a, const s16x16 b) {
		return u16x16(_mm256_cmpgt_epi16(b.getn(), a.getn()), flag_native());
	}

#else
#if __SSE4_2__ != 0
	/// \copydoc operator <(const f64x4, const f64x4)
	inline u64x4 operator <(const s64x4 a, const s64x4 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_cmpgt_epi64(b0, a0);
		const __m128i r1 = _mm_cmpgt_epi64(b1, a1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return u64x4(r, flag_native());
	}

#endif
	/// \copydoc operator <(const f64x4, const f64x4)
	inline u32x8 operator <(const s32x8 a, const s32x8 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_cmpgt_epi32(b0, a0);
		const __m128i r1 = _mm_cmpgt_epi32(b1, a1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return u32x8(r, flag_native());
	}

	/// \copydoc operator <(const f64x4, const f64x4)
	inline u16x16 operator <(const s16x16 a, const s16x16 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_cmpgt_epi16(b0, a0);
		const __m128i r1 = _mm_cmpgt_epi16(b1, a1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return u16x16(r, flag_native());
	}

#endif
	/// Compare two vectors for greater-than.
	/// \param a first source vector.
	/// \param b second source vector.
	/// \return an integral-type vector of the same lane count and width as the sources,
	/// containing -1 in the lanes where the condition was satisfied, and 0 in the remaining lanes.
	inline u64x4 operator >(const f64x4 a, const f64x4 b) {
		return b < a;
	}

	/// \copydoc operator >(const f64x4, const f64x4)
	inline u64x4 operator >(const s64x4 a, const s64x4 b) {
		return b < a;
	}

	/// \copydoc operator >(const f64x4, const f64x4)
	inline u32x8 operator >(const f32x8 a, const f32x8 b) {
		return b < a;
	}

	/// \copydoc operator >(const f64x4, const f64x4)
	inline u32x8 operator >(const s32x8 a, const s32x8 b) {
		return b < a;
	}

	/// \copydoc operator >(const f64x4, const f64x4)
	inline u16x16 operator >(const s16x16 a, const s16x16 b) {
		return b < a;
	}

	/// Compare two vectors for less-or-equal.
	/// \param a first source vector.
	/// \param b second source vector.
	/// \return an integral-type vector of the same lane count and width as the sources,
	/// containing -1 in the lanes where the condition was satisfied, and 0 in the remaining lanes.
	inline u64x4 operator <=(const f64x4 a, const f64x4 b) {
		return u64x4(_mm256_castpd_si256(_mm256_cmp_pd(a.getn(), b.getn(), _CMP_LE_OQ)), flag_native());
	}

	/// \copydoc operator <=(const f64x4, const f64x4)
	inline u32x8 operator <=(const f32x8 a, const f32x8 b) {
		return u32x8(_mm256_castps_si256(_mm256_cmp_ps(a.getn(), b.getn(), _CMP_LE_OQ)), flag_native());
	}

#if __AVX2__ != 0
	/// \copydoc operator <=(const f64x4, const f64x4)
	inline u64x4 operator <=(const s64x4 a, const s64x4 b) {
		return u64x4(_mm256_xor_si256(_mm256_cmpgt_epi64(a.getn(), b.getn()), _mm256_set1_epi64x(-1)), flag_native());
	}

	/// \copydoc operator <=(const f64x4, const f64x4)
	inline u32x8 operator <=(const s32x8 a, const s32x8 b) {
		return u32x8(_mm256_xor_si256(_mm256_cmpgt_epi32(a.getn(), b.getn()), _mm256_set1_epi64x(-1)), flag_native());
	}

	/// \copydoc operator <=(const f64x4, const f64x4)
	inline u16x16 operator <=(const s16x16 a, const s16x16 b) {
		return u16x16(_mm256_xor_si256(_mm256_cmpgt_epi16(a.getn(), b.getn()), _mm256_set1_epi64x(-1)), flag_native());
	}

#else
#if __SSE4_2__ != 0
	/// \copydoc operator <=(const f64x4, const f64x4)
	inline u64x4 operator <=(const s64x4 a, const s64x4 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_xor_si128(_mm_cmpgt_epi64(a0, b0), _mm_set1_epi64x(-1));
		const __m128i r1 = _mm_xor_si128(_mm_cmpgt_epi64(a1, b1), _mm_set1_epi64x(-1));
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return u64x4(r, flag_native());
	}

#endif
	/// \copydoc operator <=(const f64x4, const f64x4)
	inline u32x8 operator <=(const s32x8 a, const s32x8 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_xor_si128(_mm_cmpgt_epi32(a0, b0), _mm_set1_epi64x(-1));
		const __m128i r1 = _mm_xor_si128(_mm_cmpgt_epi32(a1, b1), _mm_set1_epi64x(-1));
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return u32x8(r, flag_native());
	}

	/// \copydoc operator <=(const f64x4, const f64x4)
	inline u16x16 operator <=(const s16x16 a, const s16x16 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_xor_si128(_mm_cmpgt_epi16(a0, b0), _mm_set1_epi64x(-1));
		const __m128i r1 = _mm_xor_si128(_mm_cmpgt_epi16(a1, b1), _mm_set1_epi64x(-1));
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return u16x16(r, flag_native());
	}

#endif
	/// Compare two vectors for greater-or-equal.
	/// \param a first source vector.
	/// \param b second source vector.
	/// \return an integral-type vector of the same lane count and width as the sources,
	/// containing -1 in the lanes where the condition was satisfied, and 0 in the remaining lanes.
	inline u64x4 operator >=(const f64x4 a, const f64x4 b) {
		return b <= a;
	}

	/// \copydoc operator >=(const f64x4, const f64x4)
	inline u64x4 operator >=(const s64x4 a, const s64x4 b) {
		return b <= a;
	}

	/// \copydoc operator >=(const f64x4, const f64x4)
	inline u32x8 operator >=(const f32x8 a, const f32x8 b) {
		return b <= a;
	}

	/// \copydoc operator >=(const f64x4, const f64x4)
	inline u32x8 operator >=(const s32x8 a, const s32x8 b) {
		return b <= a;
	}

	/// \copydoc operator >=(const f64x4, const f64x4)
	inline u16x16 operator >=(const s16x16 a, const s16x16 b) {
		return b <= a;
	}

#endif
#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE != 0
	inline f64x4 operator +(const f64x4 a, const f64x4 b) {
		return f64x4(_mm256_add_pd(a.getn(), b.getn()), flag_native());
	}

	inline f32x8 operator +(const f32x8 a, const f32x8 b) {
		return f32x8(_mm256_add_ps(a.getn(), b.getn()), flag_native());
	}

#if __AVX2__ != 0
	inline s64x4 operator +(const s64x4 a, const s64x4 b) {
		return s64x4(_mm256_add_epi64(a.getn(), b.getn()), flag_native());
	}

	inline u64x4 operator +(const u64x4 a, const u64x4 b) {
		return u64x4(_mm256_add_epi64(a.getn(), b.getn()), flag_native());
	}

	inline s32x8 operator +(const s32x8 a, const s32x8 b) {
		return s32x8(_mm256_add_epi32(a.getn(), b.getn()), flag_native());
	}

	inline u32x8 operator +(const u32x8 a, const u32x8 b) {
		return u32x8(_mm256_add_epi32(a.getn(), b.getn()), flag_native());
	}

	inline s16x16 operator +(const s16x16 a, const s16x16 b) {
		return s16x16(_mm256_add_epi16(a.getn(), b.getn()), flag_native());
	}

	inline u16x16 operator +(const u16x16 a, const u16x16 b) {
		return u16x16(_mm256_add_epi16(a.getn(), b.getn()), flag_native());
	}

#else
	inline s64x4 operator +(const s64x4 a, const s64x4 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_add_epi64(a0, b0);
		const __m128i r1 = _mm_add_epi64(a1, b1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return s64x4(r, flag_native());
	}

	inline u64x4 operator +(const u64x4 a, const u64x4 b) {
		return as_u64x4(as_s64x4(a) + as_s64x4(b));
	}

	inline s32x8 operator +(const s32x8 a, const s32x8 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_add_epi32(a0, b0);
		const __m128i r1 = _mm_add_epi32(a1, b1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return s32x8(r, flag_native());
	}

	inline u32x8 operator +(const u32x8 a, const u32x8 b) {
		return as_u32x8(as_s32x8(a) + as_s32x8(b));
	}

	inline s16x16 operator +(const s16x16 a, const s16x16 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_add_epi16(a0, b0);
		const __m128i r1 = _mm_add_epi16(a1, b1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return s16x16(r, flag_native());
	}

	inline u16x16 operator +(const u16x16 a, const u16x16 b) {
		return as_u16x16(as_s16x16(a) + as_s16x16(b));
	}

#endif
	inline f64x4 operator -(const f64x4 a, const f64x4 b) {
		return f64x4(_mm256_sub_pd(a.getn(), b.getn()), flag_native());
	}

	inline f32x8 operator -(const f32x8 a, const f32x8 b) {
		return f32x8(_mm256_sub_ps(a.getn(), b.getn()), flag_native());
	}

#if __AVX2__ != 0
	inline s64x4 operator -(const s64x4 a, const s64x4 b) {
		return s64x4(_mm256_sub_epi64(a.getn(), b.getn()), flag_native());
	}

	inline u64x4 operator -(const u64x4 a, const u64x4 b) {
		return u64x4(_mm256_sub_epi64(a.getn(), b.getn()), flag_native());
	}

	inline s32x8 operator -(const s32x8 a, const s32x8 b) {
		return s32x8(_mm256_sub_epi32(a.getn(), b.getn()), flag_native());
	}

	inline u32x8 operator -(const u32x8 a, const u32x8 b) {
		return u32x8(_mm256_sub_epi32(a.getn(), b.getn()), flag_native());
	}

	inline s16x16 operator -(const s16x16 a, const s16x16 b) {
		return s16x16(_mm256_sub_epi16(a.getn(), b.getn()), flag_native());
	}

	inline u16x16 operator -(const u16x16 a, const u16x16 b) {
		return u16x16(_mm256_sub_epi16(a.getn(), b.getn()), flag_native());
	}

#else
	inline s64x4 operator -(const s64x4 a, const s64x4 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_sub_epi64(a0, b0);
		const __m128i r1 = _mm_sub_epi64(a1, b1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return s64x4(r, flag_native());
	}

	inline u64x4 operator -(const u64x4 a, const u64x4 b) {
		return as_u64x4(as_s64x4(a) - as_s64x4(b));
	}

	inline s32x8 operator -(const s32x8 a, const s32x8 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_sub_epi32(a0, b0);
		const __m128i r1 = _mm_sub_epi32(a1, b1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return s32x8(r, flag_native());
	}

	inline u32x8 operator -(const u32x8 a, const u32x8 b) {
		return as_u32x8(as_s32x8(a) - as_s32x8(b));
	}

	inline s16x16 operator -(const s16x16 a, const s16x16 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_sub_epi16(a0, b0);
		const __m128i r1 = _mm_sub_epi16(a1, b1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return s16x16(r, flag_native());
	}

	inline u16x16 operator -(const u16x16 a, const u16x16 b) {
		return as_u16x16(as_s16x16(a) - as_s16x16(b));
	}

#endif
	inline f64x4 operator *(const f64x4 a, const f64x4 b) {
		return f64x4(_mm256_mul_pd(a.getn(), b.getn()), flag_native());
	}

	inline f32x8 operator *(const f32x8 a, const f32x8 b) {
		return f32x8(_mm256_mul_ps(a.getn(), b.getn()), flag_native());
	}

#if __AVX2__ != 0
	inline s32x8 operator *(const s32x8 a, const s32x8 b) {
		return s32x8(_mm256_mullo_epi32(a.getn(), b.getn()), flag_native());
	}

	inline u32x8 operator *(const u32x8 a, const u32x8 b) {
		return u32x8(_mm256_mullo_epi32(a.getn(), b.getn()), flag_native());
	}

	inline s16x16 operator *(const s16x16 a, const s16x16 b) {
		return s16x16(_mm256_mullo_epi16(a.getn(), b.getn()), flag_native());
	}

	inline u16x16 operator *(const u16x16 a, const u16x16 b) {
		return u16x16(_mm256_mullo_epi16(a.getn(), b.getn()), flag_native());
	}

#else
	inline s32x8 operator *(const s32x8 a, const s32x8 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_mullo_epi32(a0, b0);
		const __m128i r1 = _mm_mullo_epi32(a1, b1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return s32x8(r, flag_native());
	}

	inline u32x8 operator *(const u32x8 a, const u32x8 b) {
		return as_u32x8(as_s32x8(a) * as_s32x8(b));
	}

	inline s16x16 operator *(const s16x16 a, const s16x16 b) {
		const __m128i a0 = _mm256_castsi256_si128(a.getn());
		const __m128i b0 = _mm256_castsi256_si128(b.getn());
		const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i b1 = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i r0 = _mm_mullo_epi16(a0, b0);
		const __m128i r1 = _mm_mullo_epi16(a1, b1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return s16x16(r, flag_native());
	}

	inline u16x16 operator *(const u16x16 a, const u16x16 b) {
		return as_u16x16(as_s16x16(a) * as_s16x16(b));
	}

#endif
	inline f64x4 operator /(const f64x4 a, const f64x4 b) {
		return f64x4(_mm256_div_pd(a.getn(), b.getn()), flag_native());
	}

	inline f32x8 operator /(const f32x8 a, const f32x8 b) {
		return f32x8(_mm256_div_ps(a.getn(), b.getn()), flag_native());
	}

	inline f64x4 operator -(const f64x4 x) {
		return f64x4(_mm256_xor_pd(x.getn(), _mm256_set1_pd(-0.0)), flag_native());
	}

	inline f32x8 operator -(const f32x8 x) {
		return f32x8(_mm256_xor_ps(x.getn(), _mm256_set1_ps(-0.0)), flag_native());
	}

#if __AVX2__ != 0
	inline s32x8 operator -(const s32x8 x) {
		return s32x8(_mm256_sub_epi32(_mm256_setzero_si256(), x.getn()), flag_native());
	}

	inline s16x16 operator -(const s16x16 x) {
		return s16x16(_mm256_sub_epi16(_mm256_setzero_si256(), x.getn()), flag_native());
	}

#else
	inline s32x8 operator -(const s32x8 x) {
		const __m128i x0 = _mm256_castsi256_si128(x.getn());
		const __m128i x1 = _mm256_extractf128_si256(x.getn(), 1);
		const __m128i r0 = _mm_sub_epi32(_mm_setzero_si128(), x0);
		const __m128i r1 = _mm_sub_epi32(_mm_setzero_si128(), x1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return s32x8(r, flag_native());
	}

	inline s16x16 operator -(const s16x16 x) {
		const __m128i x0 = _mm256_castsi256_si128(x.getn());
		const __m128i x1 = _mm256_extractf128_si256(x.getn(), 1);
		const __m128i r0 = _mm_sub_epi16(_mm_setzero_si128(), x0);
		const __m128i r1 = _mm_sub_epi16(_mm_setzero_si128(), x1);
		const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
		return s16x16(r, flag_native());
	}

#endif
#endif
	/// Shift left logical /////////////////////////////////////////////////////////////////////////

	/// Shift left the argument by the count literal; if the count is larger or equal to the
	/// number of bits of the source argument, the result is undefined (as per the C++ standard)
	/// \param a source to be shifted
	/// \return vector where each lane is the result from the above-described op on the corresponding lane of the source
	template < uint32_t COUNT >
	inline s64x4 shl(const s64x4 a) {

#if __AVX2__ != 0
		return s64x4(_mm256_slli_epi64(a.getn(), int(COUNT)), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i rl = _mm_slli_epi64(al, int(COUNT));
		const __m128i rh = _mm_slli_epi64(ah, int(COUNT));

		return s64x4(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc shl(const s64x4 a)
	template < uint32_t COUNT >
	inline u64x4 shl(const u64x4 a) {
		return as_u64x4(shl< COUNT >(as_s64x4(a)));
	}

	/// \copydoc shl(const s64x2 a)
	template < uint32_t COUNT >
	inline s32x8 shl(const s32x8 a) {

#if __AVX2__ != 0
		return s32x8(_mm256_slli_epi32(a.getn(), int(COUNT)), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i rl = _mm_slli_epi32(al, int(COUNT));
		const __m128i rh = _mm_slli_epi32(ah, int(COUNT));

		return s32x8(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc shl(const s64x2 a)
	template < uint32_t COUNT >
	inline u32x8 shl(const u32x8 a) {
		return as_u32x8(shl< COUNT >(as_s32x8(a)));
	}

	/// \copydoc shl(const s64x2 a)
	template < uint32_t COUNT >
	inline s16x16 shl(const s16x16 a) {

#if __AVX2__ != 0
		return s16x16(_mm256_slli_epi16(a.getn(), int(COUNT)), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i rl = _mm_slli_epi16(al, int(COUNT));
		const __m128i rh = _mm_slli_epi16(ah, int(COUNT));

		return s16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc shl(const s64x2 a)
	template < uint32_t COUNT >
	inline u16x16 shl(const u16x16 a) {
		return as_u16x16(shl< COUNT >(as_s16x16(a)));
	}

	/// Shift left the first argument by the count in the second argument; if the count is larger or equal
	/// to the number of bits of the first argument, the result is undefined (as per the C++ standard)
	/// \param a source to be shifted
	/// \param c shift count
	/// \return vector where each lane is the result from the above-described op on the corresponding lane of the source
	inline s64x4 shl(const s64x4 a, const u64x4 c) {

#if __AVX2__ != 0
		return s64x4(_mm256_sllv_epi64(a.getn(), c.getn()), flag_native());

#else
		// Pre-AVX2 ISAs don't have independent lane shifts.
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i cl = _mm256_castsi256_si128(c.getn());
		const __m128i ch = _mm256_extractf128_si256(c.getn(), 1);

		const __m128i mask0 = _mm_set_epi64x(0, -1);
		const __m128i mask1 = _mm_slli_si128(mask0, 8);
		const __m128i cl0 =                cl;
		const __m128i cl1 = _mm_srli_si128(cl, 8);
		const __m128i ch0 =                ch;
		const __m128i ch1 = _mm_srli_si128(ch, 8);

		const __m128i rl0 = _mm_and_si128(_mm_sll_epi64(al, cl0), mask0);
		const __m128i rl1 = _mm_and_si128(_mm_sll_epi64(al, cl1), mask1);
		const __m128i rh0 = _mm_and_si128(_mm_sll_epi64(ah, ch0), mask0);
		const __m128i rh1 = _mm_and_si128(_mm_sll_epi64(ah, ch1), mask1);

		const __m128i rl = _mm_or_si128(rl0, rl1);
		const __m128i rh = _mm_or_si128(rh0, rh1);

		return s64x4(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc shl(const s64x4 a, u64x4 c)
	inline u64x4 shl(const u64x4 a, const u64x4 c) {
		return as_u64x4(shl(as_s64x4(a), c));
	}

	/// \copydoc shl(const s64x4 a, u64x4 c)
	inline s32x8 shl(const s32x8 a, const u32x8 c) {

#if __AVX2__ != 0
		return s32x8(_mm256_sllv_epi32(a.getn(), c.getn()), flag_native());

#else
		// Pre-AVX2 ISAs don't have independent lane shifts.
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i cl = _mm256_castsi256_si128(c.getn());
		const __m128i ch = _mm256_extractf128_si256(c.getn(), 1);

		const __m128i expl = _mm_add_epi32(_mm_slli_epi32(cl, 23), _mm_set1_epi32(0x3f800000));
		const __m128i exph = _mm_add_epi32(_mm_slli_epi32(ch, 23), _mm_set1_epi32(0x3f800000));
		const __m128i potl = _mm_cvttps_epi32(_mm_castsi128_ps(expl));
		const __m128i poth = _mm_cvttps_epi32(_mm_castsi128_ps(exph));
		const __m128i rl   = _mm_mullo_epi32(al, potl);
		const __m128i rh   = _mm_mullo_epi32(ah, poth);

		return s32x8(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc shl(const s64x4 a, u64x4 c)
	inline u32x8 shl(const u32x8 a, const u32x8 c) {
		return as_u32x8(shl(as_s32x8(a), c));
	}

	/// \copydoc shl(const s64x4 a, u64x4 c)
	inline s16x16 shl(const s16x16 a, const u16x16 c) {

#if __AVX2__ != 0
		const __m256i mask3 = _mm256_srai_epi16(_mm256_slli_epi16(c.getn(), 12), 15);
		const __m256i mask2 = _mm256_srai_epi16(_mm256_slli_epi16(c.getn(), 13), 15);
		const __m256i mask1 = _mm256_srai_epi16(_mm256_slli_epi16(c.getn(), 14), 15);
		const __m256i mask0 = _mm256_srai_epi16(_mm256_slli_epi16(c.getn(), 15), 15);
		__m256i r = a.getn();

		const __m256i r_s8 = _mm256_slli_epi16(r, 8);
		r = _mm256_blendv_epi8(r, r_s8, mask3);

		const __m256i r_s4 = _mm256_slli_epi16(r, 4);
		r = _mm256_blendv_epi8(r, r_s4, mask2);

		const __m256i r_s2 = _mm256_slli_epi16(r, 2);
		r = _mm256_blendv_epi8(r, r_s2, mask1);

		const __m256i r_s1 = _mm256_slli_epi16(r, 1);
		r = _mm256_blendv_epi8(r, r_s1, mask0);

		return s16x16(r, flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i cl = _mm256_castsi256_si128(c.getn());
		const __m128i ch = _mm256_extractf128_si256(c.getn(), 1);

		const __m128i maskl3 = _mm_or_si128(_mm_slli_epi16(cl, 12), _mm_slli_epi16(cl, 4));
		const __m128i maskh3 = _mm_or_si128(_mm_slli_epi16(ch, 12), _mm_slli_epi16(ch, 4));
		__m128i rl = al;
		__m128i rh = ah;

		const __m128i rl_s8 = _mm_slli_epi16(rl, 8);
		rl = _mm_blendv_epi8(rl, rl_s8, maskl3);
		const __m128i maskl2 = _mm_add_epi16(maskl3, maskl3);
		const __m128i rh_s8 = _mm_slli_epi16(rh, 8);
		rh = _mm_blendv_epi8(rh, rh_s8, maskh3);

		const __m128i maskh2 = _mm_add_epi16(maskh3, maskh3);
		const __m128i rl_s4 = _mm_slli_epi16(rl, 4);
		rl = _mm_blendv_epi8(rl, rl_s4, maskl2);
		const __m128i maskl1 = _mm_add_epi16(maskl2, maskl2);
		const __m128i rh_s4 = _mm_slli_epi16(rh, 4);
		rh = _mm_blendv_epi8(rh, rh_s4, maskh2);

		const __m128i maskh1 = _mm_add_epi16(maskh2, maskh2);
		const __m128i rl_s2 = _mm_slli_epi16(rl, 2);
		rl = _mm_blendv_epi8(rl, rl_s2, maskl1);
		const __m128i maskl0 = _mm_add_epi16(maskl1, maskl1);
		const __m128i rh_s2 = _mm_slli_epi16(rh, 2);
		rh = _mm_blendv_epi8(rh, rh_s2, maskh1);

		const __m128i maskh0 = _mm_add_epi16(maskh1, maskh1);
		const __m128i rl_s1 = _mm_slli_epi16(rl, 1);
		rl = _mm_blendv_epi8(rl, rl_s1, maskl0);
		const __m128i rh_s1 = _mm_slli_epi16(rh, 1);
		rh = _mm_blendv_epi8(rh, rh_s1, maskh0);

		return s16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc shl(const s64x4 a, u64x4 c)
	inline u16x16 shl(const u16x16 a, const u16x16 c) {
		return as_u16x16(shl(as_s16x16(a), c));
	}

	/// Shift right logical (unsigned) and arithmetic (signed) /////////////////////////////////////

	/// Shift right arithmetically the argument by the count literal; if the count is larger or equal to
	/// the number of bits of the source argument, the result is undefined (as per the C++ standard)
	/// \param a source to be shifted
	/// \return vector where each lane is the result from the above-described op on the corresponding lane of the source
	template < uint32_t COUNT >
	inline s64x4 shr(const s64x4 a) {

#if __AVX2__ != 0
		const __m256i sign = _mm256_slli_epi64(_mm256_cmpgt_epi64(_mm256_setzero_si256(), a.getn()), int(sizeof(s64) * 8 - COUNT));
		const __m256i r = _mm256_or_si256(_mm256_srli_epi64(a.getn(), int(COUNT)), sign);

		return s64x4(r, flag_native());

#else
		if (32 >= COUNT) {
			const __m128i a0 = _mm256_castsi256_si128(a.getn());
			const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
			const __m128i s0 = _mm_srai_epi32(a0, int(COUNT));
			const __m128i r0 = _mm_srli_epi64(a0, int(COUNT));
			const __m128i s1 = _mm_srai_epi32(a1, int(COUNT));
			const __m128i r1 = _mm_srli_epi64(a1, int(COUNT));
			const __m128i rl = _mm_blend_epi16(r0, s0, 0xcc);
			const __m128i rh = _mm_blend_epi16(r1, s1, 0xcc);
			return s64x4(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());
		}
		else {
			const __m128i a0 = _mm256_castsi256_si128(a.getn());
			const __m128i a1 = _mm256_extractf128_si256(a.getn(), 1);
			const __m128i s0 = _mm_srai_epi32(a0, 32);
			const __m128i r0 = _mm_srai_epi32(a0, int(COUNT - 32));
			const __m128i rl = _mm_blend_epi16(_mm_shuffle_epi32(r0, 0xf5), s0, 0xcc);
			const __m128i s1 = _mm_srai_epi32(a1, 32);
			const __m128i r1 = _mm_srai_epi32(a1, int(COUNT - 32));
			const __m128i rh = _mm_blend_epi16(_mm_shuffle_epi32(r1, 0xf5), s1, 0xcc);
			return s64x4(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());
		}

#endif
	}

	/// Shift right logically the argument by the count literal; if the count is larger or equal to
	/// the number of bits of the source argument, the result is undefined (as per the C++ standard)
	/// \param a source to be shifted
	/// \return vector where each lane is the result from the above-described op on the corresponding lane of the source
	template < uint32_t COUNT >
	inline u64x4 shr(const u64x4 a) {

#if __AVX2__ != 0
		return u64x4(_mm256_srli_epi64(a.getn(), int(COUNT)), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i rl = _mm_srli_epi64(al, int(COUNT));
		const __m128i rh = _mm_srli_epi64(ah, int(COUNT));

		return u64x4(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc shr(const s64x2 a)
	template < uint32_t COUNT >
	inline s32x8 shr(const s32x8 a) {

#if __AVX2__ != 0
		return s32x8(_mm256_srai_epi32(a.getn(), int(COUNT)), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i rl = _mm_srai_epi32(al, int(COUNT));
		const __m128i rh = _mm_srai_epi32(ah, int(COUNT));

		return s32x8(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc shr(const u64x2 a)
	template < uint32_t COUNT >
	inline u32x8 shr(const u32x8 a) {

#if __AVX2__ != 0
		return u32x8(_mm256_srli_epi32(a.getn(), int(COUNT)), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i rl = _mm_srli_epi32(al, int(COUNT));
		const __m128i rh = _mm_srli_epi32(ah, int(COUNT));

		return u32x8(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc shr(const s64x2 a)
	template < uint32_t COUNT >
	inline s16x16 shr(const s16x16 a) {

#if __AVX2__ != 0
		return s16x16(_mm256_srai_epi16(a.getn(), int(COUNT)), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i rl = _mm_srai_epi16(al, int(COUNT));
		const __m128i rh = _mm_srai_epi16(ah, int(COUNT));

		return s16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc shr(const u64x2 a)
	template < uint32_t COUNT >
	inline u16x16 shr(const u16x16 a) {

#if __AVX2__ != 0
		return u16x16(_mm256_srli_epi16(a.getn(), int(COUNT)), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i rl = _mm_srli_epi16(al, int(COUNT));
		const __m128i rh = _mm_srli_epi16(ah, int(COUNT));

		return u16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// Bitwise AND
	inline s64x4 operator & (const s64x4 a, const s64x4 b) {

#if __AVX2__ != 0
		return s64x4(_mm256_and_si256(a.getn(), b.getn()), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i bl = _mm256_castsi256_si128(b.getn());
		const __m128i bh = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i rl = _mm_and_si128(al, bl);
		const __m128i rh = _mm_and_si128(ah, bh);

		return s64x4(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc operator & (const s64x4 a, const s64x4 c)
	inline u64x4 operator & (const u64x4 a, const u64x4 b) {
		return as_u64x4(as_s64x4(a) & as_s64x4(b));
	}

	/// \copydoc operator & (const s64x4 a, const s64x4 c)
	inline s32x8 operator & (const s32x8 a, const s32x8 b) {

#if __AVX2__ != 0
		return s32x8(_mm256_and_si256(a.getn(), b.getn()), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i bl = _mm256_castsi256_si128(b.getn());
		const __m128i bh = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i rl = _mm_and_si128(al, bl);
		const __m128i rh = _mm_and_si128(ah, bh);

		return s32x8(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc operator & (const s64x4 a, const s64x4 c)
	inline u32x8 operator & (const u32x8 a, const u32x8 b) {
		return as_u32x8(as_s32x8(a) & as_s32x8(b));
	}

	/// \copydoc operator & (const s64x4 a, const s64x4 c)
	inline s16x16 operator & (const s16x16 a, const s16x16 b) {

#if __AVX2__ != 0
		return s16x16(_mm256_and_si256(a.getn(), b.getn()), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i bl = _mm256_castsi256_si128(b.getn());
		const __m128i bh = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i rl = _mm_and_si128(al, bl);
		const __m128i rh = _mm_and_si128(ah, bh);

		return s16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc operator & (const s64x4 a, const s64x4 c)
	inline u16x16 operator & (const u16x16 a, const u16x16 b) {
		return as_u16x16(as_s16x16(a) & as_s16x16(b));
	}

	/// Bitwise OR
	inline s64x4 operator | (const s64x4 a, const s64x4 b) {

#if __AVX2__ != 0
		return s64x4(_mm256_or_si256(a.getn(), b.getn()), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i bl = _mm256_castsi256_si128(b.getn());
		const __m128i bh = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i rl = _mm_or_si128(al, bl);
		const __m128i rh = _mm_or_si128(ah, bh);

		return s64x4(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc operator | (const s64x4 a, const s64x4 c)
	inline u64x4 operator | (const u64x4 a, const u64x4 b) {
		return as_u64x4(as_s64x4(a) | as_s64x4(b));
	}

	/// \copydoc operator | (const s64x4 a, const s64x4 c)
	inline s32x8 operator | (const s32x8 a, const s32x8 b) {

#if __AVX2__ != 0
		return s32x8(_mm256_or_si256(a.getn(), b.getn()), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i bl = _mm256_castsi256_si128(b.getn());
		const __m128i bh = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i rl = _mm_or_si128(al, bl);
		const __m128i rh = _mm_or_si128(ah, bh);

		return s32x8(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc operator | (const s64x4 a, const s64x4 c)
	inline u32x8 operator | (const u32x8 a, const u32x8 b) {
		return as_u32x8(as_s32x8(a) | as_s32x8(b));
	}

	/// \copydoc operator | (const s64x4 a, const s64x4 c)
	inline s16x16 operator | (const s16x16 a, const s16x16 b) {

#if __AVX2__ != 0
		return s16x16(_mm256_or_si256(a.getn(), b.getn()), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i bl = _mm256_castsi256_si128(b.getn());
		const __m128i bh = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i rl = _mm_or_si128(al, bl);
		const __m128i rh = _mm_or_si128(ah, bh);

		return s16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc operator | (const s64x4 a, const s64x4 c)
	inline u16x16 operator | (const u16x16 a, const u16x16 b) {
		return as_u16x16(as_s16x16(a) | as_s16x16(b));
	}

	/// Bitwise XOR
	inline s64x4 operator ^ (const s64x4 a, const s64x4 b) {

#if __AVX2__ != 0
		return s64x4(_mm256_xor_si256(a.getn(), b.getn()), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i bl = _mm256_castsi256_si128(b.getn());
		const __m128i bh = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i rl = _mm_xor_si128(al, bl);
		const __m128i rh = _mm_xor_si128(ah, bh);

		return s64x4(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc operator ^ (const s64x4 a, const s64x4 c)
	inline u64x4 operator ^ (const u64x4 a, const u64x4 b) {
		return as_u64x4(as_s64x4(a) ^ as_s64x4(b));
	}

	/// \copydoc operator ^ (const s64x4 a, const s64x4 c)
	inline s32x8 operator ^ (const s32x8 a, const s32x8 b) {

#if __AVX2__ != 0
		return s32x8(_mm256_xor_si256(a.getn(), b.getn()), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i bl = _mm256_castsi256_si128(b.getn());
		const __m128i bh = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i rl = _mm_xor_si128(al, bl);
		const __m128i rh = _mm_xor_si128(ah, bh);

		return s32x8(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc operator ^ (const s64x4 a, const s64x4 c)
	inline u32x8 operator ^ (const u32x8 a, const u32x8 b) {
		return as_u32x8(as_s32x8(a) ^ as_s32x8(b));
	}

	/// \copydoc operator ^ (const s64x4 a, const s64x4 c)
	inline s16x16 operator ^ (const s16x16 a, const s16x16 b) {

#if __AVX2__
		return s16x16(_mm256_xor_si256(a.getn(), b.getn()), flag_native());

#else
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i bl = _mm256_castsi256_si128(b.getn());
		const __m128i bh = _mm256_extractf128_si256(b.getn(), 1);
		const __m128i rl = _mm_xor_si128(al, bl);
		const __m128i rh = _mm_xor_si128(ah, bh);

		return s16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc operator ^ (const s64x4 a, const s64x4 c)
	inline u16x16 operator ^ (const u16x16 a, const u16x16 b) {
		return as_u16x16(as_s16x16(a) ^ as_s16x16(b));
	}

	/// Bitwise NOT
	inline s64x4 operator ~ (const s64x4 a) {

#if __AVX2__ != 0
		const __m256i nmask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
		return s64x4(_mm256_xor_si256(a.getn(), nmask), flag_native());

#else
		const __m128i nmask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i rl = _mm_xor_si128(al, nmask);
		const __m128i rh = _mm_xor_si128(ah, nmask);

		return s64x4(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc operator ~ (const s64x4 a)
	inline u64x4 operator ~ (const u64x4 a) {
		return as_u64x4(~ as_s64x4(a));
	}

	/// \copydoc operator ~ (const s64x4 a)
	inline s32x8 operator ~ (const s32x8 a) {

#if __AVX2__ != 0
		const __m256i nmask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
		return s32x8(_mm256_xor_si256(a.getn(), nmask), flag_native());

#else
		const __m128i nmask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i rl = _mm_xor_si128(al, nmask);
		const __m128i rh = _mm_xor_si128(ah, nmask);

		return s32x8(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc operator ~ (const s64x4 a)
	inline u32x8 operator ~ (const u32x8 a) {
		return as_u32x8(~ as_s32x8(a));
	}

	/// \copydoc operator ~ (const s64x4 a)
	inline s16x16 operator ~ (const s16x16 a) {

#if __AVX2__ != 0
		const __m256i nmask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
		return s16x16(_mm256_xor_si256(a.getn(), nmask), flag_native());

#else
		const __m128i nmask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
		const __m128i al = _mm256_castsi256_si128(a.getn());
		const __m128i ah = _mm256_extractf128_si256(a.getn(), 1);
		const __m128i rl = _mm_xor_si128(al, nmask);
		const __m128i rh = _mm_xor_si128(ah, nmask);

		return s16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(rl), rh, 1), flag_native());

#endif
	}

	/// \copydoc operator ~ (const s64x4 a)
	inline u16x16 operator ~ (const u16x16 a) {
		return as_u16x16(~ as_s16x16(a));
	}

	/// Transpose a 4x4 matrix. Slower than the f32x4 SSE version in all cases except reading-modifying-writing
	/// on SNB and IVB, and slower all around on HSW (increased-latency ops).
	/// \param src01 first and second rows of the source matrix
	/// \param src23 third and fourth rows of the source matrix
	/// \param[out] dst01 first and second rows of the result matrix
	/// \param[out] dst23 third and fourth rows of the result matrix
	inline void transpose4x4_ext(
		const f32x4 src0,
		const f32x4 src1,
		const f32x4 src2,
		const f32x4 src3,
		f32x4& dst0,
		f32x4& dst1,
		f32x4& dst2,
		f32x4& dst3) {

		const __m256 src02 = _mm256_insertf128_ps(_mm256_castps128_ps256(src0.getn()), src2.getn(), 1);
		const __m256 src13 = _mm256_insertf128_ps(_mm256_castps128_ps256(src1.getn()), src3.getn(), 1);
		const __m256 t01 = _mm256_unpacklo_ps(src02, src13);
		const __m256 t23 = _mm256_unpackhi_ps(src02, src13);

		// Since the basic permute (64-bit-element unpacklo256) that would allow us to efficiently output two 256-bit vectors is absent
		// from AVX, and the general 64-bit-element permute in AVX2 is slow (vpermpd), we revert to 128-bit extractions and shuffles.
		const __m128 t0 = _mm256_castps256_ps128(t01);
		const __m128 t2 = _mm256_castps256_ps128(t23);
		const __m128 t1 = _mm256_extractf128_ps(t01, 1);
		const __m128 t3 = _mm256_extractf128_ps(t23, 1);
		dst0 = f32x4(_mm_movelh_ps(t0, t1), flag_native());
		dst1 = f32x4(_mm_movehl_ps(t1, t0), flag_native());
		dst2 = f32x4(_mm_movelh_ps(t2, t3), flag_native());
		dst3 = f32x4(_mm_movehl_ps(t3, t2), flag_native());
	}

	/// Square root of the argument.
	/// \param x non-negative argument
	/// \return square root of argument; NaN if argument is negative.
	inline f64x4 sqrt(
		const f64x4 x) {

		return f64x4(_mm256_sqrt_pd(x.getn()), flag_native());
	}

	/// \copydoc sqrt(const f64x4)
	inline f32x8 sqrt(
		const f32x8 x) {

		return f32x8(_mm256_sqrt_ps(x.getn()), flag_native());
	}

	/// Natural logarithm of the argument.
	/// \param x a positive argument
	/// \return natural logarithm of the argument; NaN if argument is non-positive.
	inline f32x8 log(
		const f32x8 x) {

		return f32x8(cephes_log(x.getn()), flag_native());
	}

	/// Natural exponent of the argument.
	/// \param x argument
	/// \return natural exponent of the argument.
	inline f32x8 exp(
		const f32x8 x) {

		return f32x8(cephes_exp(x.getn()), flag_native());
	}

	/// Raise non-negative base to the given power. Function uses natural exp and log
	/// to carry its computation, and thus is less exact than some other methods.
	/// \param x base argument
	/// \param y power argument
	/// \return base argument raised to the power; NaN if base is negative.
	inline f32x8 pow(
		const f32x8 x,
		const f32x8 y) {

		return f32x8(cephes_pow(x.getn(), y.getn()), flag_native());
	}

	/// Sine of the argument.
	/// \param x angle in radians; keep less than 8192 for best precision.
	/// \return sine of the argument.
	inline f32x8 sin(
		const f32x8 x) {

		return f32x8(cephes_sin(x.getn()), flag_native());
	}

	/// Cosine of the argument.
	/// \param x angle in radians; keep less than 8192 for best precision.
	/// \return cosine of the argument.
	inline f32x8 cos(
		const f32x8 x) {

		return f32x8(cephes_cos(x.getn()), flag_native());
	}

	/// Simultaneous sine and cosine of the arument.
	/// \param[in] x angle in radians; keep less than 8192 for best precision.
	/// \param[out] sin sine of the argument.
	/// \param[out] cos cosine of the argument.
	inline void sincos(
		const f32x8 x,
		f32x8& sin,
		f32x8& cos) {

		f32x8::native s, c;
		cephes_sincos(x.getn(), &s, &c);
		sin = f32x8(s, flag_native());
		cos = f32x8(c, flag_native());
	}

#endif
#if __ARM_NEON != 0
#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE != 0
	// note: no generic vectors with this compiler - use the 'native' vector type as the underlying vector type
	typedef native2< f64, float64x2_t, float64x2_t > f64x2;
	typedef native2< s64, int64x2_t,   int64x2_t   > s64x2;
	typedef native2< u64, uint64x2_t,  uint64x2_t  > u64x2;
	typedef native4< f32, float32x4_t, float32x4_t > f32x4;
	typedef native4< s32, int32x4_t,   int32x4_t   > s32x4;
	typedef native4< u32, uint32x4_t,  uint32x4_t  > u32x4;
	typedef native2< f32, float32x2_t, float32x2_t > f32x2;
	typedef native2< s32, int32x2_t,   int32x2_t   > s32x2;
	typedef native2< u32, uint32x2_t,  uint32x2_t  > u32x2;
	typedef native8< s16, int16x8_t,   int16x8_t   > s16x8;
	typedef native8< u16, uint16x8_t,  uint16x8_t  > u16x8;
	typedef native4< s16, int16x4_t,   int16x4_t   > s16x4;
	typedef native4< u16, uint16x4_t,  uint16x4_t  > u16x4;

#else
	typedef native2< f64, f64 __attribute__ ((vector_size(2 * sizeof(f64)))), float64x2_t > f64x2;
	typedef native2< s64, s64 __attribute__ ((vector_size(2 * sizeof(s64)))), int64x2_t   > s64x2;
	typedef native2< u64, u64 __attribute__ ((vector_size(2 * sizeof(u64)))), uint64x2_t  > u64x2;
	typedef native4< f32, f32 __attribute__ ((vector_size(4 * sizeof(f32)))), float32x4_t > f32x4;
	typedef native4< s32, s32 __attribute__ ((vector_size(4 * sizeof(s32)))), int32x4_t   > s32x4;
	typedef native4< u32, u32 __attribute__ ((vector_size(4 * sizeof(u32)))), uint32x4_t  > u32x4;
	typedef native2< f32, f32 __attribute__ ((vector_size(2 * sizeof(f32)))), float32x2_t > f32x2;
	typedef native2< s32, s32 __attribute__ ((vector_size(2 * sizeof(s32)))), int32x2_t   > s32x2;
	typedef native2< u32, u32 __attribute__ ((vector_size(2 * sizeof(u32)))), uint32x2_t  > u32x2;
	typedef native8< s16, s16 __attribute__ ((vector_size(8 * sizeof(s16)))), int16x8_t   > s16x8;
	typedef native8< u16, u16 __attribute__ ((vector_size(8 * sizeof(u16)))), uint16x8_t  > u16x8;
	typedef native4< s16, s16 __attribute__ ((vector_size(4 * sizeof(s16)))), int16x4_t   > s16x4;
	typedef native4< u16, u16 __attribute__ ((vector_size(4 * sizeof(u16)))), uint16x4_t  > u16x4;

#endif
	#define NATIVE_F64X2    1
	#define NATIVE_S64X2    1
	#define NATIVE_U64X2    1
	#define NATIVE_F32X4    1
	#define NATIVE_S32X4    1
	#define NATIVE_U32X4    1
	#define NATIVE_F32X2    1
	#define NATIVE_S32X2    1
	#define NATIVE_U32X2    1
	#define NATIVE_S16X8    1
	#define NATIVE_U16X8    1
	#define NATIVE_S16X4    1
	#define NATIVE_U16X4    1

	// bitcasts ////////////////////////////////////////////////////////////////////////////////////
	inline f64x2 as_f64x2(const s64x2 x) {
		return f64x2(vreinterpretq_f64_s64(x.getn()), flag_native());
	}

	inline f64x2 as_f64x2(const u64x2 x) {
		return f64x2(vreinterpretq_f64_u64(x.getn()), flag_native());
	}

	inline s64x2 as_s64x2(const f64x2 x) {
		return s64x2(vreinterpretq_s64_f64(x.getn()), flag_native());
	}

	inline s64x2 as_s64x2(const u64x2 x) {
		return s64x2(vreinterpretq_s64_u64(x.getn()), flag_native());
	}

	inline u64x2 as_u64x2(const f64x2 x) {
		return u64x2(vreinterpretq_u64_f64(x.getn()), flag_native());
	}

	inline u64x2 as_u64x2(const s64x2 x) {
		return u64x2(vreinterpretq_u64_s64(x.getn()), flag_native());
	}

	inline f32x4 as_f32x4(const s32x4 x) {
		return f32x4(vreinterpretq_f32_s32(x.getn()), flag_native());
	}

	inline f32x4 as_f32x4(const u32x4 x) {
		return f32x4(vreinterpretq_f32_u32(x.getn()), flag_native());
	}

	inline s32x4 as_s32x4(const f32x4 x) {
		return s32x4(vreinterpretq_s32_f32(x.getn()), flag_native());
	}

	inline s32x4 as_s32x4(const u32x4 x) {
		return s32x4(vreinterpretq_s32_u32(x.getn()), flag_native());
	}

	inline u32x4 as_u32x4(const f32x4 x) {
		return u32x4(vreinterpretq_u32_f32(x.getn()), flag_native());
	}

	inline u32x4 as_u32x4(const s32x4 x) {
		return u32x4(vreinterpretq_u32_s32(x.getn()), flag_native());
	}

	inline f32x2 as_f32x2(const s32x2 x) {
		return f32x2(vreinterpret_f32_s32(x.getn()), flag_native());
	}

	inline f32x2 as_f32x2(const u32x2 x) {
		return f32x2(vreinterpret_f32_u32(x.getn()), flag_native());
	}

	inline s32x2 as_s32x2(const f32x2 x) {
		return s32x2(vreinterpret_s32_f32(x.getn()), flag_native());
	}

	inline s32x2 as_s32x2(const u32x2 x) {
		return s32x2(vreinterpret_s32_u32(x.getn()), flag_native());
	}

	inline u32x2 as_u32x2(const f32x2 x) {
		return u32x2(vreinterpret_u32_f32(x.getn()), flag_native());
	}

	inline u32x2 as_u32x2(const s32x2 x) {
		return u32x2(vreinterpret_u32_s32(x.getn()), flag_native());
	}

	inline s16x8 as_s16x8(const u16x8 x) {
		return s16x8(vreinterpretq_s16_u16(x.getn()), flag_native());
	}

	inline u16x8 as_u16x8(const s16x8 x) {
		return u16x8(vreinterpretq_u16_s16(x.getn()), flag_native());
	}

	inline s16x4 as_s16x4(const u16x4 x) {
		return s16x4(vreinterpret_s16_u16(x.getn()), flag_native());
	}

	inline u16x4 as_u16x4(const s16x4 x) {
		return u16x4(vreinterpret_u16_s16(x.getn()), flag_native());
	}

	// half-vector extraction //////////////////////////////////////////////////////////////////////
	template < uint32_t INDEX >
	inline f32x2 get_f32x2(const f32x4 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
			return f32x2(vget_high_f32(x.getn()), flag_native());

		return f32x2(vget_low_f32(x.getn()), flag_native());
	}

	template < uint32_t INDEX >
	inline s32x2 get_s32x2(const s32x4 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
			return s32x2(vget_high_s32(x.getn()), flag_native());

		return s32x2(vget_low_s32(x.getn()), flag_native());
	}

	template < uint32_t INDEX >
	inline u32x2 get_u32x2(const u32x4 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
			return u32x2(vget_high_u32(x.getn()), flag_native());

		return u32x2(vget_low_u32(x.getn()), flag_native());
	}

	template < uint32_t INDEX >
	inline s16x4 get_s16x4(const s16x8 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
			return s16x4(vget_high_s16(x.getn()), flag_native());

		return s16x4(vget_low_s16(x.getn()), flag_native());
	}

	template < uint32_t INDEX >
	inline u16x4 get_u16x4(const u16x8 x) {
		const compile_assert< 1 >= INDEX > assert_index;

		if (1 == INDEX)
			return u16x4(vget_high_u16(x.getn()), flag_native());

		return u16x4(vget_low_u16(x.getn()), flag_native());
	}

	// composition from half-vectors ///////////////////////////////////////////////////////////////
	inline f32x4 get_f32x4(const f32x2 x, const f32x2 y) {
		return f32x4(vcombine_f32(x.getn(), y.getn()), flag_native());
	}

	inline s32x4 get_s32x4(const s32x2 x, const s32x2 y) {
		return s32x4(vcombine_s32(x.getn(), y.getn()), flag_native());
	}

	inline u32x4 get_u32x4(const u32x2 x, const u32x2 y) {
		return u32x4(vcombine_u32(x.getn(), y.getn()), flag_native());
	}

	inline s16x8 get_s16x8(const s16x4 x, const s16x4 y) {
		return s16x8(vcombine_s16(x.getn(), y.getn()), flag_native());
	}

	inline u16x8 get_u16x8(const u16x4 x, const u16x4 y) {
		return u16x8(vcombine_u16(x.getn(), y.getn()), flag_native());
	}

	// reduction predicates ////////////////////////////////////////////////////////////////////////
	inline bool all(const u64x2 lane_mask, const uint64_t bit_submask = 3) {
		assert(bit_submask > 0 && bit_submask < 4);

		const uint64x2_t bitmask = vec2< uint64x2_t, u64 >(1, 2);
		return bit_submask == (bit_submask & vaddvq_u64(vandq_u64(bitmask, lane_mask.getn())));
	}

	inline bool all(const u32x4 lane_mask, const uint32_t bit_submask = 15) {
		assert(bit_submask > 0 && bit_submask < 16);

		const uint32x4_t bitmask = vec4< uint32x4_t, u32 >(1, 2, 4, 8);
		return bit_submask == (bit_submask & vaddvq_u32(vandq_u32(bitmask, lane_mask.getn())));
	}

	inline bool all(const u32x2 lane_mask, const uint32_t bit_submask = 3) {
		assert(bit_submask > 0 && bit_submask < 4);

		const uint32x2_t bitmask = vec2< uint32x2_t, u32 >(1, 2);
		return bit_submask == (bit_submask & vaddv_u32(vand_u32(bitmask, lane_mask.getn())));
	}

	inline bool all(const u16x8 lane_mask, const uint16_t bit_submask = 255) {
		assert(bit_submask > 0 && bit_submask < 256);

		const uint16x8_t bitmask = vec8< uint16x8_t, u16 >(1, 2, 4, 8, 16, 32, 64, 128);
		return bit_submask == (bit_submask & vaddvq_u16(vandq_u16(bitmask, lane_mask.getn())));
	}

	inline bool all(const u16x4 lane_mask, const uint16_t bit_submask = 15) {
		assert(bit_submask > 0 && bit_submask < 256);

		const uint16x4_t bitmask = vec4< uint16x4_t, u16 >(1, 2, 4, 8);
		return bit_submask == (bit_submask & vaddv_u16(vand_u16(bitmask, lane_mask.getn())));
	}

	inline bool any(const u64x2 lane_mask, const uint64_t bit_submask = 3) {
		assert(bit_submask > 0 && bit_submask < 4);

		const uint64x2_t bitmask = vec2< uint64x2_t, u64 >(1, 2);
		return 0 != (bit_submask & vaddvq_u64(vandq_u64(bitmask, lane_mask.getn())));
	}

	inline bool any(const u32x4 lane_mask, const uint32_t bit_submask = 15) {
		assert(bit_submask > 0 && bit_submask < 16);

		const uint32x4_t bitmask = vec4< uint32x4_t, u32 >(1, 2, 4, 8);
		return 0 != (bit_submask & vaddvq_u32(vandq_u32(bitmask, lane_mask.getn())));
	}

	inline bool any(const u32x2 lane_mask, const uint32_t bit_submask = 3) {
		assert(bit_submask > 0 && bit_submask < 4);

		const uint32x2_t bitmask = vec2< uint32x2_t, u32 >(1, 2);
		return 0 != (bit_submask & vaddv_u32(vand_u32(bitmask, lane_mask.getn())));
	}

	inline bool any(const u16x8 lane_mask, const uint16_t bit_submask = 255) {
		assert(bit_submask > 0 && bit_submask < 256);

		const uint16x8_t bitmask = vec8< uint16x8_t, u16 >(1, 2, 4, 8, 16, 32, 64, 128);
		return 0 != (bit_submask & vaddvq_u16(vandq_u16(bitmask, lane_mask.getn())));
	}

	inline bool any(const u16x4 lane_mask, const uint16_t bit_submask = 15) {
		assert(bit_submask > 0 && bit_submask < 256);

		const uint16x4_t bitmask = vec4< uint16x4_t, u16 >(1, 2, 4, 8);
		return 0 != (bit_submask & vaddv_u16(vand_u16(bitmask, lane_mask.getn())));
	}

	inline bool none(const u64x2 lane_mask, const uint64_t bit_submask = 3) {
		assert(bit_submask > 0 && bit_submask < 4);

		const uint64x2_t bitmask = vec2< uint64x2_t, u64 >(1, 2);
		return 0 == (bit_submask & vaddvq_u64(vandq_u64(bitmask, lane_mask.getn())));
	}

	inline bool none(const u32x4 lane_mask, const uint32_t bit_submask = 15) {
		assert(bit_submask > 0 && bit_submask < 16);

		const uint32x4_t bitmask = vec4< uint32x4_t, u32 >(1, 2, 4, 8);
		return 0 == (bit_submask & vaddvq_u32(vandq_u32(bitmask, lane_mask.getn())));
	}

	inline bool none(const u32x2 lane_mask, const uint32_t bit_submask = 3) {
		assert(bit_submask > 0 && bit_submask < 4);

		const uint32x2_t bitmask = vec2< uint32x2_t, u32 >(1, 2);
		return 0 == (bit_submask & vaddv_u32(vand_u32(bitmask, lane_mask.getn())));
	}

	inline bool none(const u16x8 lane_mask, const uint16_t bit_submask = 255) {
		assert(bit_submask > 0 && bit_submask < 256);

		const uint16x8_t bitmask = vec8< uint16x8_t, u16 >(1, 2, 4, 8, 16, 32, 64, 128);
		return 0 == (bit_submask & vaddvq_u16(vandq_u16(bitmask, lane_mask.getn())));
	}

	inline bool none(const u16x4 lane_mask, const uint16_t bit_submask = 15) {
		assert(bit_submask > 0 && bit_submask < 256);

		const uint16x4_t bitmask = vec4< uint16x4_t, u16 >(1, 2, 4, 8);
		return 0 == (bit_submask & vaddv_u16(vand_u16(bitmask, lane_mask.getn())));
	}

	// extrema /////////////////////////////////////////////////////////////////////////////////////
	inline f64x2 min(const f64x2 a, const f64x2 b) {
		return f64x2(vminq_f64(a.getn(), b.getn()), flag_native());
	}

	inline f64x2 max(const f64x2 a, const f64x2 b) {
		return f64x2(vmaxq_f64(a.getn(), b.getn()), flag_native());
	}

	inline f32x4 min(const f32x4 a, const f32x4 b) {
		return f32x4(vminq_f32(a.getn(), b.getn()), flag_native());
	}

	inline f32x4 max(const f32x4 a, const f32x4 b) {
		return f32x4(vmaxq_f32(a.getn(), b.getn()), flag_native());
	}

	inline f32x2 min(const f32x2 a, const f32x2 b) {
		return f32x2(vmin_f32(a.getn(), b.getn()), flag_native());
	}

	inline f32x2 max(const f32x2 a, const f32x2 b) {
		return f32x2(vmax_f32(a.getn(), b.getn()), flag_native());
	}

	inline s32x4 min(const s32x4 a, const s32x4 b) {
		return s32x4(vminq_s32(a.getn(), b.getn()), flag_native());
	}

	inline s32x4 max(const s32x4 a, const s32x4 b) {
		return s32x4(vmaxq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 min(const u32x4 a, const u32x4 b) {
		return u32x4(vminq_u32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 max(const u32x4 a, const u32x4 b) {
		return u32x4(vmaxq_u32(a.getn(), b.getn()), flag_native());
	}

	inline s32x2 min(const s32x2 a, const s32x2 b) {
		return s32x2(vmin_s32(a.getn(), b.getn()), flag_native());
	}

	inline s32x2 max(const s32x2 a, const s32x2 b) {
		return s32x2(vmax_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 min(const u32x2 a, const u32x2 b) {
		return u32x2(vmin_u32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 max(const u32x2 a, const u32x2 b) {
		return u32x2(vmax_u32(a.getn(), b.getn()), flag_native());
	}

	inline s16x8 min(const s16x8 a, const s16x8 b) {
		return s16x8(vminq_s16(a.getn(), b.getn()), flag_native());
	}

	inline s16x8 max(const s16x8 a, const s16x8 b) {
		return s16x8(vmaxq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 min(const u16x8 a, const u16x8 b) {
		return u16x8(vminq_u16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 max(const u16x8 a, const u16x8 b) {
		return u16x8(vmaxq_u16(a.getn(), b.getn()), flag_native());
	}

	inline s16x4 min(const s16x4 a, const s16x4 b) {
		return s16x4(vmin_s16(a.getn(), b.getn()), flag_native());
	}

	inline s16x4 max(const s16x4 a, const s16x4 b) {
		return s16x4(vmax_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 min(const u16x4 a, const u16x4 b) {
		return u16x4(vmin_u16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 max(const u16x4 a, const u16x4 b) {
		return u16x4(vmax_u16(a.getn(), b.getn()), flag_native());
	}

	// mask off lanes as nil ///////////////////////////////////////////////////////////////////////
	inline f64x2 mask(const f64x2 x, const u64x2 lane_mask) {
		return f64x2(vreinterpretq_f64_u64(vandq_u64(lane_mask.getn(), vreinterpretq_u64_f64(x.getn()))), flag_native());
	}

	inline s64x2 mask(const s64x2 x, const u64x2 lane_mask) {
		return s64x2(vreinterpretq_s64_u64(vandq_u64(lane_mask.getn(), vreinterpretq_u64_s64(x.getn()))), flag_native());
	}

	inline u64x2 mask(const u64x2 x, const u64x2 lane_mask) {
		return u64x2(vandq_u64(lane_mask.getn(), x.getn()), flag_native());
	}

	inline f32x4 mask(const f32x4 x, const u32x4 lane_mask) {
		return f32x4(vreinterpretq_f32_u32(vandq_u32(lane_mask.getn(), vreinterpretq_u32_f32(x.getn()))), flag_native());
	}

	inline s32x4 mask(const s32x4 x, const u32x4 lane_mask) {
		return s32x4(vreinterpretq_s32_u32(vandq_u32(lane_mask.getn(), vreinterpretq_u32_s32(x.getn()))), flag_native());
	}

	inline u32x4 mask(const u32x4 x, const u32x4 lane_mask) {
		return u32x4(vandq_u32(lane_mask.getn(), x.getn()), flag_native());
	}

	inline f32x2 mask(const f32x2 x, const u32x2 lane_mask) {
		return f32x2(vreinterpret_f32_u32(vand_u32(lane_mask.getn(), vreinterpret_u32_f32(x.getn()))), flag_native());
	}

	inline s32x2 mask(const s32x2 x, const u32x2 lane_mask) {
		return s32x2(vreinterpret_s32_u32(vand_u32(lane_mask.getn(), vreinterpret_u32_s32(x.getn()))), flag_native());
	}

	inline u32x2 mask(const u32x2 x, const u32x2 lane_mask) {
		return u32x2(vand_u32(lane_mask.getn(), x.getn()), flag_native());
	}

	inline s16x8 mask(const s16x8 x, const u16x8 lane_mask) {
		return s16x8(vreinterpretq_s16_u16(vandq_u16(lane_mask.getn(), vreinterpretq_u16_s16(x.getn()))), flag_native());
	}

	inline u16x8 mask(const u16x8 x, const u16x8 lane_mask) {
		return u16x8(vandq_u16(lane_mask.getn(), x.getn()), flag_native());
	}

	inline s16x4 mask(const s16x4 x, const u16x4 lane_mask) {
		return s16x4(vreinterpret_s16_u16(vand_u16(lane_mask.getn(), vreinterpret_u16_s16(x.getn()))), flag_native());
	}

	inline u16x4 mask(const u16x4 x, const u16x4 lane_mask) {
		return u16x4(vand_u16(lane_mask.getn(), x.getn()), flag_native());
	}

	// combine lanes by bitmask literal ////////////////////////////////////////////////////////////
	template < uint32_t SELECTOR >
	inline f64x2 select(const f64x2 a, const f64x2 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 4 > assert_selector;

		const uint64_t u = -1;
		const uint64x2_t vmsk = vec2< uint64x2_t, u64 >(
			SELECTOR & 0x1 ? u : 0, SELECTOR & 0x2 ? u : 0);
		return f64x2(vbslq_f64(vmsk, b.getn(), a.getn()), flag_native());
	}

	template < uint32_t SELECTOR >
	inline s64x2 select(const s64x2 a, const s64x2 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 4 > assert_selector;

		const uint64_t u = -1;
		const uint64x2_t vmsk = vec2< uint64x2_t, u64 >(
			SELECTOR & 0x1 ? u : 0, SELECTOR & 0x2 ? u : 0);
		return s64x2(vbslq_s64(vmsk, b.getn(), a.getn()), flag_native());
	}

	template < uint32_t SELECTOR >
	inline u64x2 select(const u64x2 a, const u64x2 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 4 > assert_selector;

		const uint64_t u = -1;
		const uint64x2_t vmsk = vec2< uint64x2_t, u64 >(
			SELECTOR & 0x1 ? u : 0, SELECTOR & 0x2 ? u : 0);
		return u64x2(vbslq_u64(vmsk, b.getn(), a.getn()), flag_native());
	}

	template < uint32_t SELECTOR >
	inline f32x4 select(const f32x4 a, const f32x4 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 16 > assert_selector;

		const uint32_t u = -1;
		const uint32x4_t vmsk = vec4< uint32x4_t, u32 >(
			SELECTOR & 0x1 ? u : 0, SELECTOR & 0x2 ? u : 0, SELECTOR & 0x4 ? u : 0, SELECTOR & 0x8 ? u : 0);
		return f32x4(vbslq_f32(vmsk, b.getn(), a.getn()), flag_native());
	}

	template < uint32_t SELECTOR >
	inline s32x4 select(const s32x4 a, const s32x4 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 16 > assert_selector;

		const uint32_t u = -1;
		const uint32x4_t vmsk = vec4< uint32x4_t, u32 >(
			SELECTOR & 0x1 ? u : 0, SELECTOR & 0x2 ? u : 0, SELECTOR & 0x4 ? u : 0, SELECTOR & 0x8 ? u : 0);
		return s32x4(vbslq_s32(vmsk, b.getn(), a.getn()), flag_native());
	}

	template < uint32_t SELECTOR >
	inline u32x4 select(const u32x4 a, const u32x4 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 16 > assert_selector;

		const uint32_t u = -1;
		const uint32x4_t vmsk = vec4< uint32x4_t, u32 >(
			SELECTOR & 0x1 ? u : 0, SELECTOR & 0x2 ? u : 0, SELECTOR & 0x4 ? u : 0, SELECTOR & 0x8 ? u : 0);
		return u32x4(vbslq_u32(vmsk, b.getn(), a.getn()), flag_native());
	}

	template < uint32_t SELECTOR >
	inline f32x2 select(const f32x2 a, const f32x2 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 4 > assert_selector;

		const uint32_t u = -1;
		const uint32x2_t vmsk = vec2< uint32x2_t, u32 >(
			SELECTOR & 0x1 ? u : 0, SELECTOR & 0x2 ? u : 0);
		return f32x2(vbsl_f32(vmsk, b.getn(), a.getn()), flag_native());
	}

	template < uint32_t SELECTOR >
	inline s32x2 select(const s32x2 a, const s32x2 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 4 > assert_selector;

		const uint32_t u = -1;
		const uint32x2_t vmsk = vec2< uint32x2_t, u32 >(
			SELECTOR & 0x1 ? u : 0, SELECTOR & 0x2 ? u : 0);
		return s32x2(vbsl_s32(vmsk, b.getn(), a.getn()), flag_native());
	}

	template < uint32_t SELECTOR >
	inline u32x2 select(const u32x2 a, const u32x2 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 4 > assert_selector;

		const uint32_t u = -1;
		const uint32x2_t vmsk = vec2< uint32x2_t, u32 >(
			SELECTOR & 0x1 ? u : 0, SELECTOR & 0x2 ? u : 0);
		return u32x2(vbsl_u32(vmsk, b.getn(), a.getn()), flag_native());
	}

	template < uint32_t SELECTOR >
	inline s16x8 select(const s16x8 a, const s16x8 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 256 > assert_selector;

		const uint16_t u = -1;
		const uint16x8_t vmsk = vec8< uint16x8_t, u16 >(
			SELECTOR & 0x01 ? u : 0, SELECTOR & 0x02 ? u : 0, SELECTOR & 0x04 ? u : 0, SELECTOR & 0x08 ? u : 0,
			SELECTOR & 0x10 ? u : 0, SELECTOR & 0x20 ? u : 0, SELECTOR & 0x40 ? u : 0, SELECTOR & 0x80 ? u : 0);
		return s16x8(vbslq_s16(vmsk, b.getn(), a.getn()), flag_native());
	}

	template < uint32_t SELECTOR >
	inline u16x8 select(const u16x8 a, const u16x8 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 256 > assert_selector;

		const uint16_t u = -1;
		const uint16x8_t vmsk = vec8< uint16x8_t, u16 >(
			SELECTOR & 0x01 ? u : 0, SELECTOR & 0x02 ? u : 0, SELECTOR & 0x04 ? u : 0, SELECTOR & 0x08 ? u : 0,
			SELECTOR & 0x10 ? u : 0, SELECTOR & 0x20 ? u : 0, SELECTOR & 0x40 ? u : 0, SELECTOR & 0x80 ? u : 0);
		return u16x8(vbslq_u16(vmsk, b.getn(), a.getn()), flag_native());
	}

	template < uint32_t SELECTOR >
	inline s16x4 select(const s16x4 a, const s16x4 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 16 > assert_selector;

		const uint16_t u = -1;
		const uint16x4_t vmsk = vec4< uint16x4_t, u16 >(
			SELECTOR & 0x1 ? u : 0, SELECTOR & 0x2 ? u : 0, SELECTOR & 0x4 ? u : 0, SELECTOR & 0x8 ? u : 0);
		return s16x4(vbsl_s16(vmsk, b.getn(), a.getn()), flag_native());
	}

	template < uint32_t SELECTOR >
	inline u16x4 select(const u16x4 a, const u16x4 b) {
		const compile_assert< SELECTOR >= 0 && SELECTOR < 16 > assert_selector;

		const uint16_t u = -1;
		const uint16x4_t vmsk = vec4< uint16x4_t, u16 >(
			SELECTOR & 0x1 ? u : 0, SELECTOR & 0x2 ? u : 0, SELECTOR & 0x4 ? u : 0, SELECTOR & 0x8 ? u : 0);
		return u16x4(vbsl_u16(vmsk, b.getn(), a.getn()), flag_native());
	}

	// combine lanes by lane mask //////////////////////////////////////////////////////////////////
	inline f64x2 select(const f64x2 a, const f64x2 b, const u64x2 selector) {
		return f64x2(vbslq_f64(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	inline s64x2 select(const s64x2 a, const s64x2 b, const u64x2 selector) {
		return s64x2(vbslq_s64(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	inline u64x2 select(const u64x2 a, const u64x2 b, const u64x2 selector) {
		return u64x2(vbslq_u64(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	inline f32x4 select(const f32x4 a, const f32x4 b, const u32x4 selector) {
		return f32x4(vbslq_f32(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	inline s32x4 select(const s32x4 a, const s32x4 b, const u32x4 selector) {
		return s32x4(vbslq_s32(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	inline u32x4 select(const u32x4 a, const u32x4 b, const u32x4 selector) {
		return u32x4(vbslq_u32(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	inline f32x2 select(const f32x2 a, const f32x2 b, const u32x2 selector) {
		return f32x2(vbsl_f32(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	inline s32x2 select(const s32x2 a, const s32x2 b, const u32x2 selector) {
		return s32x2(vbsl_s32(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	inline u32x2 select(const u32x2 a, const u32x2 b, const u32x2 selector) {
		return u32x2(vbsl_u32(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	inline s16x8 select(const s16x8 a, const s16x8 b, const u16x8 selector) {
		return s16x8(vbslq_s16(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	inline u16x8 select(const u16x8 a, const u16x8 b, const u16x8 selector) {
		return u16x8(vbslq_u16(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	inline s16x4 select(const s16x4 a, const s16x4 b, const u16x4 selector) {
		return s16x4(vbsl_s16(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	inline u16x4 select(const u16x4 a, const u16x4 b, const u16x4 selector) {
		return u16x4(vbsl_u16(selector.getn(), b.getn(), a.getn()), flag_native());
	}

	// absolute value //////////////////////////////////////////////////////////////////////////////
	inline f64x2 abs(const f64x2 x) {
		return f64x2(vabsq_f64(x.getn()), flag_native());
	}

	inline s64x2 abs(const s64x2 x) {
		return s64x2(vabsq_s64(x.getn()), flag_native());
	}

	inline f32x4 abs(const f32x4 x) {
		return f32x4(vabsq_f32(x.getn()), flag_native());
	}

	inline s32x4 abs(const s32x4 x) {
		return s32x4(vabsq_s32(x.getn()), flag_native());
	}

	inline f32x2 abs(const f32x2 x) {
		return f32x2(vabs_f32(x.getn()), flag_native());
	}

	inline s32x2 abs(const s32x2 x) {
		return s32x2(vabs_s32(x.getn()), flag_native());
	}

	inline s16x8 abs(const s16x8 x) {
		return s16x8(vabsq_s16(x.getn()), flag_native());
	}

	inline s16x4 abs(const s16x4 x) {
		return s16x4(vabs_s16(x.getn()), flag_native());
	}

#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE != 0 || COMPILER_QUIRK_0003_RELATIONAL_OPS != 0
	inline u64x2 operator ==(const f64x2 a, const f64x2 b) {
		return u64x2(vceqq_f64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator ==(const s64x2 a, const s64x2 b) {
		return u64x2(vceqq_s64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator ==(const u64x2 a, const u64x2 b) {
		return u64x2(vceqq_u64(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator ==(const f32x4 a, const f32x4 b) {
		return u32x4(vceqq_f32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator ==(const s32x4 a, const s32x4 b) {
		return u32x4(vceqq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator ==(const u32x4 a, const u32x4 b) {
		return u32x4(vceqq_u32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator ==(const f32x2 a, const f32x2 b) {
		return u32x2(vceq_f32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator ==(const s32x2 a, const s32x2 b) {
		return u32x2(vceq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator ==(const u32x2 a, const u32x2 b) {
		return u32x2(vceq_u32(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator ==(const s16x8 a, const s16x8 b) {
		return u16x8(vceqq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator ==(const u16x8 a, const u16x8 b) {
		return u16x8(vceqq_u16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator ==(const s16x4 a, const s16x4 b) {
		return u16x4(vceq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator ==(const u16x4 a, const u16x4 b) {
		return u16x4(vceq_u16(a.getn(), b.getn()), flag_native());
	}

#if COMPILER_QUIRK_0004_VMVNQ_U64 != 0
	inline uint64x2_t vmvnq_u64(uint64x2_t src) {
		return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(src)));
	}

#endif
	inline u64x2 operator !=(const f64x2 a, const f64x2 b) {
		return u64x2(vmvnq_u64(vceqq_f64(a.getn(), b.getn())), flag_native());
	}

	inline u64x2 operator !=(const s64x2 a, const s64x2 b) {
		return u64x2(vmvnq_u64(vceqq_s64(a.getn(), b.getn())), flag_native());
	}

	inline u64x2 operator !=(const u64x2 a, const u64x2 b) {
		return u64x2(vmvnq_u64(vceqq_u64(a.getn(), b.getn())), flag_native());
	}

	inline u32x4 operator !=(const f32x4 a, const f32x4 b) {
		return u32x4(vmvnq_u32(vceqq_f32(a.getn(), b.getn())), flag_native());
	}

	inline u32x4 operator !=(const s32x4 a, const s32x4 b) {
		return u32x4(vmvnq_u32(vceqq_s32(a.getn(), b.getn())), flag_native());
	}

	inline u32x4 operator !=(const u32x4 a, const u32x4 b) {
		return u32x4(vmvnq_u32(vceqq_u32(a.getn(), b.getn())), flag_native());
	}

	inline u32x2 operator !=(const f32x2 a, const f32x2 b) {
		return u32x2(vmvn_u32(vceq_f32(a.getn(), b.getn())), flag_native());
	}

	inline u32x2 operator !=(const s32x2 a, const s32x2 b) {
		return u32x2(vmvn_u32(vceq_s32(a.getn(), b.getn())), flag_native());
	}

	inline u32x2 operator !=(const u32x2 a, const u32x2 b) {
		return u32x2(vmvn_u32(vceq_u32(a.getn(), b.getn())), flag_native());
	}

	inline u16x8 operator !=(const s16x8 a, const s16x8 b) {
		return u16x8(vmvnq_u16(vceqq_s16(a.getn(), b.getn())), flag_native());
	}

	inline u16x8 operator !=(const u16x8 a, const u16x8 b) {
		return u16x8(vmvnq_u16(vceqq_u16(a.getn(), b.getn())), flag_native());
	}

	inline u16x4 operator !=(const s16x4 a, const s16x4 b) {
		return u16x4(vmvn_u16(vceq_s16(a.getn(), b.getn())), flag_native());
	}

	inline u16x4 operator !=(const u16x4 a, const u16x4 b) {
		return u16x4(vmvn_u16(vceq_u16(a.getn(), b.getn())), flag_native());
	}

	inline u64x2 operator <(const f64x2 a, const f64x2 b) {
		return u64x2(vcltq_f64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator <(const s64x2 a, const s64x2 b) {
		return u64x2(vcltq_s64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator <(const u64x2 a, const u64x2 b) {
		return u64x2(vcltq_u64(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator <(const f32x4 a, const f32x4 b) {
		return u32x4(vcltq_f32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator <(const s32x4 a, const s32x4 b) {
		return u32x4(vcltq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator <(const u32x4 a, const u32x4 b) {
		return u32x4(vcltq_u32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator <(const f32x2 a, const f32x2 b) {
		return u32x2(vclt_f32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator <(const s32x2 a, const s32x2 b) {
		return u32x2(vclt_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator <(const u32x2 a, const u32x2 b) {
		return u32x2(vclt_u32(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator <(const s16x8 a, const s16x8 b) {
		return u16x8(vcltq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator <(const u16x8 a, const u16x8 b) {
		return u16x8(vcltq_u16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator <(const s16x4 a, const s16x4 b) {
		return u16x4(vclt_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator <(const u16x4 a, const u16x4 b) {
		return u16x4(vclt_u16(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator >(const f64x2 a, const f64x2 b) {
		return u64x2(vcgtq_f64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator >(const s64x2 a, const s64x2 b) {
		return u64x2(vcgtq_s64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator >(const u64x2 a, const u64x2 b) {
		return u64x2(vcgtq_u64(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator >(const f32x4 a, const f32x4 b) {
		return u32x4(vcgtq_f32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator >(const s32x4 a, const s32x4 b) {
		return u32x4(vcgtq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator >(const u32x4 a, const u32x4 b) {
		return u32x4(vcgtq_u32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator >(const f32x2 a, const f32x2 b) {
		return u32x2(vcgt_f32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator >(const s32x2 a, const s32x2 b) {
		return u32x2(vcgt_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator >(const u32x2 a, const u32x2 b) {
		return u32x2(vcgt_u32(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator >(const s16x8 a, const s16x8 b) {
		return u16x8(vcgtq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator >(const u16x8 a, const u16x8 b) {
		return u16x8(vcgtq_u16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator >(const s16x4 a, const s16x4 b) {
		return u16x4(vcgt_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator >(const u16x4 a, const u16x4 b) {
		return u16x4(vcgt_u16(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator <=(const f64x2 a, const f64x2 b) {
		return u64x2(vcleq_f64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator <=(const s64x2 a, const s64x2 b) {
		return u64x2(vcleq_s64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator <=(const u64x2 a, const u64x2 b) {
		return u64x2(vcleq_u64(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator <=(const f32x4 a, const f32x4 b) {
		return u32x4(vcleq_f32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator <=(const s32x4 a, const s32x4 b) {
		return u32x4(vcleq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator <=(const u32x4 a, const u32x4 b) {
		return u32x4(vcleq_u32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator <=(const f32x2 a, const f32x2 b) {
		return u32x2(vcle_f32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator <=(const s32x2 a, const s32x2 b) {
		return u32x2(vcle_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator <=(const u32x2 a, const u32x2 b) {
		return u32x2(vcle_u32(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator <=(const s16x8 a, const s16x8 b) {
		return u16x8(vcleq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator <=(const u16x8 a, const u16x8 b) {
		return u16x8(vcleq_u16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator <=(const s16x4 a, const s16x4 b) {
		return u16x4(vcle_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator <=(const u16x4 a, const u16x4 b) {
		return u16x4(vcle_u16(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator >=(const f64x2 a, const f64x2 b) {
		return u64x2(vcgeq_f64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator >=(const s64x2 a, const s64x2 b) {
		return u64x2(vcgeq_s64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator >=(const u64x2 a, const u64x2 b) {
		return u64x2(vcgeq_u64(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator >=(const f32x4 a, const f32x4 b) {
		return u32x4(vcgeq_f32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator >=(const s32x4 a, const s32x4 b) {
		return u32x4(vcgeq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator >=(const u32x4 a, const u32x4 b) {
		return u32x4(vcgeq_u32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator >=(const f32x2 a, const f32x2 b) {
		return u32x2(vcge_f32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator >=(const s32x2 a, const s32x2 b) {
		return u32x2(vcge_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator >=(const u32x2 a, const u32x2 b) {
		return u32x2(vcge_u32(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator >=(const s16x8 a, const s16x8 b) {
		return u16x8(vcgeq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator >=(const u16x8 a, const u16x8 b) {
		return u16x8(vcgeq_u16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator >=(const s16x4 a, const s16x4 b) {
		return u16x4(vcge_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator >=(const u16x4 a, const u16x4 b) {
		return u16x4(vcge_u16(a.getn(), b.getn()), flag_native());
	}

#endif
#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE != 0
	inline f64x2 operator +(const f64x2 a, const f64x2 b) {
		return f64x2(vaddq_f64(a.getn(), b.getn()), flag_native());
	}

	inline s64x2 operator +(const s64x2 a, const s64x2 b) {
		return s64x2(vaddq_s64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator +(const u64x2 a, const u64x2 b) {
		return u64x2(vaddq_u64(a.getn(), b.getn()), flag_native());
	}

	inline f32x4 operator +(const f32x4 a, const f32x4 b) {
		return f32x4(vaddq_f32(a.getn(), b.getn()), flag_native());
	}

	inline s32x4 operator +(const s32x4 a, const s32x4 b) {
		return s32x4(vaddq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator +(const u32x4 a, const u32x4 b) {
		return u32x4(vaddq_u32(a.getn(), b.getn()), flag_native());
	}

	inline f32x2 operator +(const f32x2 a, const f32x2 b) {
		return f32x2(vadd_f32(a.getn(), b.getn()), flag_native());
	}

	inline s32x2 operator +(const s32x2 a, const s32x2 b) {
		return s32x2(vadd_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator +(const u32x2 a, const u32x2 b) {
		return u32x2(vadd_u32(a.getn(), b.getn()), flag_native());
	}

	inline s16x8 operator +(const s16x8 a, const s16x8 b) {
		return s16x8(vaddq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator +(const u16x8 a, const u16x8 b) {
		return u16x8(vaddq_u16(a.getn(), b.getn()), flag_native());
	}

	inline s16x4 operator +(const s16x4 a, const s16x4 b) {
		return s16x4(vadd_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator +(const u16x4 a, const u16x4 b) {
		return u16x4(vadd_u16(a.getn(), b.getn()), flag_native());
	}

	inline f64x2 operator -(const f64x2 a, const f64x2 b) {
		return f64x2(vsubq_f64(a.getn(), b.getn()), flag_native());
	}

	inline s64x2 operator -(const s64x2 a, const s64x2 b) {
		return s64x2(vsubq_s64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator -(const u64x2 a, const u64x2 b) {
		return u64x2(vsubq_u64(a.getn(), b.getn()), flag_native());
	}

	inline f32x4 operator -(const f32x4 a, const f32x4 b) {
		return f32x4(vsubq_f32(a.getn(), b.getn()), flag_native());
	}

	inline s32x4 operator -(const s32x4 a, const s32x4 b) {
		return s32x4(vsubq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator -(const u32x4 a, const u32x4 b) {
		return u32x4(vsubq_u32(a.getn(), b.getn()), flag_native());
	}

	inline f32x2 operator -(const f32x2 a, const f32x2 b) {
		return f32x2(vsub_f32(a.getn(), b.getn()), flag_native());
	}

	inline s32x2 operator -(const s32x2 a, const s32x2 b) {
		return s32x2(vsub_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator -(const u32x2 a, const u32x2 b) {
		return u32x2(vsub_u32(a.getn(), b.getn()), flag_native());
	}

	inline s16x8 operator -(const s16x8 a, const s16x8 b) {
		return s16x8(vsubq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator -(const u16x8 a, const u16x8 b) {
		return u16x8(vsubq_u16(a.getn(), b.getn()), flag_native());
	}

	inline s16x4 operator -(const s16x4 a, const s16x4 b) {
		return s16x4(vsub_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator -(const u16x4 a, const u16x4 b) {
		return u16x4(vsub_u16(a.getn(), b.getn()), flag_native());
	}

	inline f64x2 operator *(const f64x2 a, const f64x2 b) {
		return f64x2(vmulq_f64(a.getn(), b.getn()), flag_native());
	}

	inline f32x4 operator *(const f32x4 a, const f32x4 b) {
		return f32x4(vmulq_f32(a.getn(), b.getn()), flag_native());
	}

	inline s32x4 operator *(const s32x4 a, const s32x4 b) {
		return s32x4(vmulq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator *(const u32x4 a, const u32x4 b) {
		return u32x4(vmulq_u32(a.getn(), b.getn()), flag_native());
	}

	inline f32x2 operator *(const f32x2 a, const f32x2 b) {
		return f32x2(vmul_f32(a.getn(), b.getn()), flag_native());
	}

	inline s32x2 operator *(const s32x2 a, const s32x2 b) {
		return s32x2(vmul_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator *(const u32x2 a, const u32x2 b) {
		return u32x2(vmul_u32(a.getn(), b.getn()), flag_native());
	}

	inline s16x8 operator *(const s16x8 a, const s16x8 b) {
		return s16x8(vmulq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator *(const u16x8 a, const u16x8 b) {
		return u16x8(vmulq_u16(a.getn(), b.getn()), flag_native());
	}

	inline s16x4 operator *(const s16x4 a, const s16x4 b) {
		return s16x4(vmul_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator *(const u16x4 a, const u16x4 b) {
		return u16x4(vmul_u16(a.getn(), b.getn()), flag_native());
	}

	inline f64x2 operator /(const f64x2 a, const f64x2 b) {
		return f64x2(vdivq_f64(a.getn(), b.getn()), flag_native());
	}

	inline f32x4 operator /(const f32x4 a, const f32x4 b) {
		return f32x4(vdivq_f32(a.getn(), b.getn()), flag_native());
	}

	inline f32x2 operator /(const f32x2 a, const f32x2 b) {
		return f32x2(vdiv_f32(a.getn(), b.getn()), flag_native());
	}

	inline f64x2 operator -(const f64x2 x) {
		return f64x2(vnegq_f64(x.getn()), flag_native());
	}

	inline f32x4 operator -(const f32x4 x) {
		return f32x4(vnegq_f32(x.getn()), flag_native());
	}

	inline s32x4 operator -(const s32x4 x) {
		return s32x4(vnegq_s32(x.getn()), flag_native());
	}

	inline s16x8 operator -(const s16x8 x) {
		return s16x8(vnegq_s16(x.getn()), flag_native());
	}

#endif
	/// Shift left logical /////////////////////////////////////////////////////////////////////////
	template < uint32_t COUNT >
	inline s64x2 shl(const s64x2 a) {
		return s64x2(vshlq_n_s64(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline u64x2 shl(const u64x2 a) {
		return u64x2(vshlq_n_u64(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline s32x4 shl(const s32x4 a) {
		return s32x4(vshlq_n_s32(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline u32x4 shl(const u32x4 a) {
		return u32x4(vshlq_n_u32(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline s32x2 shl(const s32x2 a) {
		return s32x2(vshl_n_s32(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline u32x2 shl(const u32x2 a) {
		return u32x2(vshl_n_u32(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline s16x8 shl(const s16x8 a) {
		return s16x8(vshlq_n_s16(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline u16x8 shl(const u16x8 a) {
		return u16x8(vshlq_n_u16(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline s16x4 shl(const s16x4 a) {
		return s16x4(vshl_n_s16(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline u16x4 shl(const u16x4 a) {
		return u16x4(vshl_n_u16(a.getn(), int(COUNT)), flag_native());
	}

	inline s64x2 shl(const s64x2 a, const u64x2 c) {
		return s64x2(vshlq_s64(a.getn(), as_s64x2(c).getn()), flag_native());
	}

	inline u64x2 shl(const u64x2 a, const u64x2 c) {
		return u64x2(vshlq_u64(a.getn(), as_s64x2(c).getn()), flag_native());
	}

	inline s32x4 shl(const s32x4 a, const u32x4 c) {
		return s32x4(vshlq_s32(a.getn(), as_s32x4(c).getn()), flag_native());
	}

	inline u32x4 shl(const u32x4 a, const u32x4 c) {
		return u32x4(vshlq_u32(a.getn(), as_s32x4(a).getn()), flag_native());
	}

	inline s32x2 shl(const s32x2 a, const u32x2 c) {
		return s32x2(vshl_s32(a.getn(), as_s32x2(c).getn()), flag_native());
	}

	inline u32x2 shl(const u32x2 a, const u32x2 c) {
		return u32x2(vshl_u32(a.getn(), as_s32x2(c).getn()), flag_native());
	}

	inline s16x8 shl(const s16x8 a, const u16x8 c) {
		return s16x8(vshlq_s16(a.getn(), as_s16x8(c).getn()), flag_native());
	}

	inline u16x8 shl(const u16x8 a, const u16x8 c) {
		return u16x8(vshlq_u16(a.getn(), as_s16x8(c).getn()), flag_native());
	}

	inline s16x4 shl(const s16x4 a, const u16x4 c) {
		return s16x4(vshl_s16(a.getn(), as_s16x4(c).getn()), flag_native());
	}

	inline u16x4 shl(const u16x4 a, const u16x4 c) {
		return u16x4(vshl_u16(a.getn(), as_s16x4(c).getn()), flag_native());
	}

	/// Shift right logical (unsigned) and arithmetic (signed) /////////////////////////////////////
	template < uint32_t COUNT >
	inline s64x2 shr(const s64x2 a) {
		return s64x2(vshrq_n_s64(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline u64x2 shr(const u64x2 a) {
		return u64x2(vshrq_n_u64(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline s32x4 shr(const s32x4 a) {
		return s32x4(vshrq_n_s32(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline u32x4 shr(const u32x4 a) {
		return u32x4(vshrq_n_u32(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline s32x2 shr(const s32x2 a) {
		return s32x2(vshr_n_s32(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline u32x2 shr(const u32x2 a) {
		return u32x2(vshr_n_u32(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline s16x8 shr(const s16x8 a) {
		return s16x8(vshrq_n_s16(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline u16x8 shr(const u16x8 a) {
		return u16x8(vshrq_n_u16(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline s16x4 shr(const s16x4 a) {
		return s16x4(vshr_n_s16(a.getn(), int(COUNT)), flag_native());
	}

	template < uint32_t COUNT >
	inline u16x4 shr(const u16x4 a) {
		return u16x4(vshr_n_u16(a.getn(), int(COUNT)), flag_native());
	}

	inline s64x2 shr(const s64x2 a, const u64x2 c) {
		return s64x2(vshlq_s64(a.getn(), vnegq_s64(as_s64x2(c).getn())), flag_native());
	}

	inline u64x2 shr(const u64x2 a, const u64x2 c) {
		return u64x2(vshlq_u64(a.getn(), vnegq_s64(as_s64x2(c).getn())), flag_native());
	}

	inline s32x4 shr(const s32x4 a, const u32x4 c) {
		return s32x4(vshlq_s32(a.getn(), vnegq_s32(as_s32x4(c).getn())), flag_native());
	}

	inline u32x4 shr(const u32x4 a, const u32x4 c) {
		return u32x4(vshlq_u32(a.getn(), vnegq_s32(as_s32x4(c).getn())), flag_native());
	}

	inline s32x2 shr(const s32x2 a, const u32x2 c) {
		return s32x2(vshl_s32(a.getn(), vneg_s32(as_s32x2(c).getn())), flag_native());
	}

	inline u32x2 shr(const u32x2 a, const u32x2 c) {
		return u32x2(vshl_u32(a.getn(), vneg_s32(as_s32x2(c).getn())), flag_native());
	}

	inline s16x8 shr(const s16x8 a, const u16x8 c) {
		return s16x8(vshlq_s16(a.getn(), vnegq_s16(as_s16x8(c).getn())), flag_native());
	}

	inline u16x8 shr(const u16x8 a, const u16x8 c) {
		return u16x8(vshlq_u16(a.getn(), vnegq_s16(as_s16x8(c).getn())), flag_native());
	}

	inline s16x4 shr(const s16x4 a, const u16x4 c) {
		return s16x4(vshl_s16(a.getn(), vneg_s16(as_s16x4(c).getn())), flag_native());
	}

	inline u16x4 shr(const u16x4 a, const u16x4 c) {
		return u16x4(vshl_u16(a.getn(), vneg_s16(as_s16x4(c).getn())), flag_native());
	}

	inline s64x2 operator & (const s64x2 a, const s64x2 b) {
		return s64x2(vandq_s64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator & (const u64x2 a, const u64x2 b) {
		return u64x2(vandq_u64(a.getn(), b.getn()), flag_native());
	}

	inline s32x4 operator & (const s32x4 a, const s32x4 b) {
		return s32x4(vandq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator & (const u32x4 a, const u32x4 b) {
		return u32x4(vandq_u32(a.getn(), b.getn()), flag_native());
	}

	inline s32x2 operator & (const s32x2 a, const s32x2 b) {
		return s32x2(vand_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator & (const u32x2 a, const u32x2 b) {
		return u32x2(vand_u32(a.getn(), b.getn()), flag_native());
	}

	inline s16x8 operator & (const s16x8 a, const s16x8 b) {
		return s16x8(vandq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator & (const u16x8 a, const u16x8 b) {
		return u16x8(vandq_u16(a.getn(), b.getn()), flag_native());
	}

	inline s16x4 operator & (const s16x4 a, const s16x4 b) {
		return s16x4(vand_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator & (const u16x4 a, const u16x4 b) {
		return u16x4(vand_u16(a.getn(), b.getn()), flag_native());
	}

	inline s64x2 operator | (const s64x2 a, const s64x2 b) {
		return s64x2(vorrq_s64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator | (const u64x2 a, const u64x2 b) {
		return u64x2(vorrq_u64(a.getn(), b.getn()), flag_native());
	}

	inline s32x4 operator | (const s32x4 a, const s32x4 b) {
		return s32x4(vorrq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator | (const u32x4 a, const u32x4 b) {
		return u32x4(vorrq_u32(a.getn(), b.getn()), flag_native());
	}

	inline s32x2 operator | (const s32x2 a, const s32x2 b) {
		return s32x2(vorr_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator | (const u32x2 a, const u32x2 b) {
		return u32x2(vorr_u32(a.getn(), b.getn()), flag_native());
	}

	inline s16x8 operator | (const s16x8 a, const s16x8 b) {
		return s16x8(vorrq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator | (const u16x8 a, const u16x8 b) {
		return u16x8(vorrq_u16(a.getn(), b.getn()), flag_native());
	}

	inline s16x4 operator | (const s16x4 a, const s16x4 b) {
		return s16x4(vorr_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator | (const u16x4 a, const u16x4 b) {
		return u16x4(vorr_u16(a.getn(), b.getn()), flag_native());
	}

	inline s64x2 operator ^ (const s64x2 a, const s64x2 b) {
		return s64x2(veorq_s64(a.getn(), b.getn()), flag_native());
	}

	inline u64x2 operator ^ (const u64x2 a, const u64x2 b) {
		return u64x2(veorq_u64(a.getn(), b.getn()), flag_native());
	}

	inline s32x4 operator ^ (const s32x4 a, const s32x4 b) {
		return s32x4(veorq_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x4 operator ^ (const u32x4 a, const u32x4 b) {
		return u32x4(veorq_u32(a.getn(), b.getn()), flag_native());
	}

	inline s32x2 operator ^ (const s32x2 a, const s32x2 b) {
		return s32x2(veor_s32(a.getn(), b.getn()), flag_native());
	}

	inline u32x2 operator ^ (const u32x2 a, const u32x2 b) {
		return u32x2(veor_u32(a.getn(), b.getn()), flag_native());
	}

	inline s16x8 operator ^ (const s16x8 a, const s16x8 b) {
		return s16x8(veorq_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x8 operator ^ (const u16x8 a, const u16x8 b) {
		return u16x8(veorq_u16(a.getn(), b.getn()), flag_native());
	}

	inline s16x4 operator ^ (const s16x4 a, const s16x4 b) {
		return s16x4(veor_s16(a.getn(), b.getn()), flag_native());
	}

	inline u16x4 operator ^ (const u16x4 a, const u16x4 b) {
		return u16x4(veor_u16(a.getn(), b.getn()), flag_native());
	}

	inline s64x2 operator ~ (const s64x2 a) {
		return s64x2(vreinterpretq_s64_s32(vmvnq_s32(vreinterpretq_s32_s64(a.getn()))), flag_native());
	}

	inline u64x2 operator ~ (const u64x2 a) {
		return u64x2(vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(a.getn()))), flag_native());
	}

	inline s32x4 operator ~ (const s32x4 a) {
		return s32x4(vmvnq_s32(a.getn()), flag_native());
	}

	inline u32x4 operator ~ (const u32x4 a) {
		return u32x4(vmvnq_u32(a.getn()), flag_native());
	}

	inline s32x2 operator ~ (const s32x2 a) {
		return s32x2(vmvn_s32(a.getn()), flag_native());
	}

	inline u32x2 operator ~ (const u32x2 a) {
		return u32x2(vmvn_u32(a.getn()), flag_native());
	}

	inline s16x8 operator ~ (const s16x8 a) {
		return s16x8(vmvnq_s16(a.getn()), flag_native());
	}

	inline u16x8 operator ~ (const u16x8 a) {
		return u16x8(vmvnq_u16(a.getn()), flag_native());
	}

	inline s16x4 operator ~ (const s16x4 a) {
		return s16x4(vmvn_s16(a.getn()), flag_native());
	}

	inline u16x4 operator ~ (const u16x4 a) {
		return u16x4(vmvn_u16(a.getn()), flag_native());
	}

	/// Transpose a 4x4 matrix.
	/// \param src0 first row of the source matrix
	/// \param src1 second row of the source matrix
	/// \param src2 third row of the source matrix
	/// \param src3 fourth row of the source matrix
	/// \param[out] dst0 first row of the result matrix
	/// \param[out] dst1 second row of the result matrix
	/// \param[out] dst2 third row of the result matrix
	/// \param[out] dst3 fourth row of the result matrix
	inline void transpose4x4(
		const f32x4 src0,
		const f32x4 src1,
		const f32x4 src2,
		const f32x4 src3,
		f32x4& dst0,
		f32x4& dst1,
		f32x4& dst2,
		f32x4& dst3) {

		const float32x4x2_t t01 = vtrnq_f32(src0.getn(), src1.getn());
		const float32x4x2_t t23 = vtrnq_f32(src2.getn(), src3.getn());
		dst0 = f32x4(vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01.val[0]), vreinterpretq_f64_f32(t23.val[0]))), flag_native());
		dst1 = f32x4(vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01.val[1]), vreinterpretq_f64_f32(t23.val[1]))), flag_native());
		dst2 = f32x4(vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01.val[0]), vreinterpretq_f64_f32(t23.val[0]))), flag_native());
		dst3 = f32x4(vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01.val[1]), vreinterpretq_f64_f32(t23.val[1]))), flag_native());
	}

	/// Transpose a 3x3 matrix.
	/// \param src0 first row of the source matrix
	/// \param src1 second row of the source matrix
	/// \param src2 third row of the source matrix
	/// \param[out] dst0 first row of the result matrix
	/// \param[out] dst1 second row of the result matrix
	/// \param[out] dst2 third row of the result matrix
	inline void transpose3x3(
		const f32x4 src0,
		const f32x4 src1,
		const f32x4 src2,
		f32x4& dst0,
		f32x4& dst1,
		f32x4& dst2) {

		const float32x4x2_t t01 = vtrnq_f32(src0.getn(), src1.getn());
		const float32x4x2_t t23 = vtrnq_f32(src2.getn(), src2.getn());
		dst0 = f32x4(vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01.val[0]), vreinterpretq_f64_f32(t23.val[0]))), flag_native());
		dst1 = f32x4(vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01.val[1]), vreinterpretq_f64_f32(t23.val[1]))), flag_native());
		dst2 = f32x4(vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01.val[0]), vreinterpretq_f64_f32(t23.val[0]))), flag_native());
	}

	/// Square root of the argument.
	/// \param x non-negative argument
	/// \return square root of argument; NaN if argument is negative.
	inline f64x2 sqrt(
		const f64x2 x) {

		return f64x2(vsqrtq_f64(x.getn()), flag_native());
	}

	/// \copydoc sqrt(const f64x2)
	inline f32x4 sqrt(
		const f32x4 x) {

		return f32x4(vsqrtq_f32(x.getn()), flag_native());
	}

	/// \copydoc sqrt(const f64x2)
	inline f32x2 sqrt(
		const f32x2 x) {

		return f32x2(vsqrt_f32(x.getn()), flag_native());
	}

	/// Natural logarithm of the argument.
	/// \param x a positive argument
	/// \return natural logarithm of the argument; NaN if argument is non-positive.
	inline f32x4 log(
		const f32x4 x) {

		return f32x4(cephes_log(x.getn()), flag_native());
	}

	/// Natural exponent of the argument.
	/// \param x argument
	/// \return natural exponent of the argument.
	inline f32x4 exp(
		const f32x4 x) {

		return f32x4(cephes_exp(x.getn()), flag_native());
	}

	/// Raise non-negative base to the given power. Function uses natural exp and log
	/// to carry its computation, and thus is less exact than some other methods.
	/// \param x base argument
	/// \param y power argument
	/// \return base argument raised to the power; NaN if base is negative.
	inline f32x4 pow(
		const f32x4 x,
		const f32x4 y) {

		return f32x4(cephes_pow(x.getn(), y.getn()), flag_native());
	}

	/// Sine of the argument.
	/// \param x angle in radians; keep less than 8192 for best precision.
	/// \return sine of the argument.
	inline f32x4 sin(
		const f32x4 x) {

		return f32x4(cephes_sin(x.getn()), flag_native());
	}

	/// Cosine of the argument.
	/// \param x angle in radians; keep less than 8192 for best precision.
	/// \return cosine of the argument.
	inline f32x4 cos(
		const f32x4 x) {

		return f32x4(cephes_cos(x.getn()), flag_native());
	}

	/// Simultaneous sine and cosine of the arument.
	/// \param[in] x angle in radians; keep less than 8192 for best precision.
	/// \param[out] sin sine of the argument.
	/// \param[out] cos cosine of the argument.
	inline void sincos(
		const f32x4 x,
		f32x4& sin,
		f32x4& cos) {

		f32x4::native s, c;
		cephes_sincos(x.getn(), &s, &c);
		sin = f32x4(s, flag_native());
		cos = f32x4(c, flag_native());
	}

#endif
#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE == 0
// note: take care of operator ambiguities

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native2< SCALAR_T, GENERIC_T, NATIVE_T > operator +(
	const native2< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native2< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) + GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native2< SCALAR_T, GENERIC_T, NATIVE_T > operator -(
	const native2< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native2< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) - GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native2< SCALAR_T, GENERIC_T, NATIVE_T > operator *(
	const native2< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native2< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) * GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native2< SCALAR_T, GENERIC_T, NATIVE_T > operator /(
	const native2< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native2< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) / GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native2< SCALAR_T, GENERIC_T, NATIVE_T > operator -(
	const native2< SCALAR_T, GENERIC_T, NATIVE_T > a) {

	return -GENERIC_T(a);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native4< SCALAR_T, GENERIC_T, NATIVE_T > operator +(
	const native4< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native4< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) + GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native4< SCALAR_T, GENERIC_T, NATIVE_T > operator -(
	const native4< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native4< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) - GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native4< SCALAR_T, GENERIC_T, NATIVE_T > operator *(
	const native4< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native4< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) * GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native4< SCALAR_T, GENERIC_T, NATIVE_T > operator /(
	const native4< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native4< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) / GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native4< SCALAR_T, GENERIC_T, NATIVE_T > operator -(
	const native4< SCALAR_T, GENERIC_T, NATIVE_T > a) {

	return -GENERIC_T(a);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native8< SCALAR_T, GENERIC_T, NATIVE_T > operator +(
	const native8< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native8< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) + GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native8< SCALAR_T, GENERIC_T, NATIVE_T > operator -(
	const native8< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native8< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) - GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native8< SCALAR_T, GENERIC_T, NATIVE_T > operator *(
	const native8< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native8< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) * GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native8< SCALAR_T, GENERIC_T, NATIVE_T > operator /(
	const native8< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native8< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) / GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native8< SCALAR_T, GENERIC_T, NATIVE_T > operator -(
	const native8< SCALAR_T, GENERIC_T, NATIVE_T > a) {

	return -GENERIC_T(a);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native16< SCALAR_T, GENERIC_T, NATIVE_T > operator +(
	const native16< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native16< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) + GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native16< SCALAR_T, GENERIC_T, NATIVE_T > operator -(
	const native16< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native16< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) - GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native16< SCALAR_T, GENERIC_T, NATIVE_T > operator *(
	const native16< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native16< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) * GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native16< SCALAR_T, GENERIC_T, NATIVE_T > operator /(
	const native16< SCALAR_T, GENERIC_T, NATIVE_T > a,
	const native16< SCALAR_T, GENERIC_T, NATIVE_T > b) {

	return GENERIC_T(a) / GENERIC_T(b);
}

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline native16< SCALAR_T, GENERIC_T, NATIVE_T > operator -(
	const native16< SCALAR_T, GENERIC_T, NATIVE_T > a) {

	return -GENERIC_T(a);
}

#endif
#if COMPILER_QUIRK_0000_ARITHMETIC_TYPE == 0 && COMPILER_QUIRK_0003_RELATIONAL_OPS == 0
// note: take care of operator ambiguities

#if NATIVE_U64X2 != 0
inline u64x2 operator ==(const f64x2 a, const f64x2 b) {
	return u64x2(f64x2::generic(a) == f64x2::generic(b));
}

inline u64x2 operator ==(const s64x2 a, const s64x2 b) {
	return u64x2(s64x2::generic(a) == s64x2::generic(b));
}

inline u64x2 operator ==(const u64x2 a, const u64x2 b) {
	return u64x2(u64x2::generic(a) == u64x2::generic(b));
}

inline u64x2 operator !=(const f64x2 a, const f64x2 b) {
	return u64x2(f64x2::generic(a) != f64x2::generic(b));
}

inline u64x2 operator !=(const s64x2 a, const s64x2 b) {
	return u64x2(s64x2::generic(a) != s64x2::generic(b));
}

inline u64x2 operator !=(const u64x2 a, const u64x2 b) {
	return u64x2(u64x2::generic(a) != u64x2::generic(b));
}

inline u64x2 operator <(const f64x2 a, const f64x2 b) {
	return u64x2(f64x2::generic(a) < f64x2::generic(b));
}

inline u64x2 operator <(const s64x2 a, const s64x2 b) {
	return u64x2(s64x2::generic(a) < s64x2::generic(b));
}

inline u64x2 operator <(const u64x2 a, const u64x2 b) {
	return u64x2(u64x2::generic(a) < u64x2::generic(b));
}

inline u64x2 operator >(const f64x2 a, const f64x2 b) {
	return u64x2(f64x2::generic(a) > f64x2::generic(b));
}

inline u64x2 operator >(const s64x2 a, const s64x2 b) {
	return u64x2(s64x2::generic(a) > s64x2::generic(b));
}

inline u64x2 operator >(const u64x2 a, const u64x2 b) {
	return u64x2(u64x2::generic(a) > u64x2::generic(b));
}

inline u64x2 operator <=(const f64x2 a, const f64x2 b) {
	return u64x2(f64x2::generic(a) <= f64x2::generic(b));
}

inline u64x2 operator <=(const s64x2 a, const s64x2 b) {
	return u64x2(s64x2::generic(a) <= s64x2::generic(b));
}

inline u64x2 operator <=(const u64x2 a, const u64x2 b) {
	return u64x2(u64x2::generic(a) <= u64x2::generic(b));
}

inline u64x2 operator >=(const f64x2 a, const f64x2 b) {
	return u64x2(f64x2::generic(a) >= f64x2::generic(b));
}

inline u64x2 operator >=(const s64x2 a, const s64x2 b) {
	return u64x2(s64x2::generic(a) >= s64x2::generic(b));
}

inline u64x2 operator >=(const u64x2 a, const u64x2 b) {
	return u64x2(u64x2::generic(a) >= u64x2::generic(b));
}

#endif
#if NATIVE_U32X2 != 0
inline u32x2 operator ==(const f32x2 a, const f32x2 b) {
	return u32x2(f32x2::generic(a) == f32x2::generic(b));
}

inline u32x2 operator ==(const s32x2 a, const s32x2 b) {
	return u32x2(s32x2::generic(a) == s32x2::generic(b));
}

inline u32x2 operator ==(const u32x2 a, const u32x2 b) {
	return u32x2(u32x2::generic(a) == u32x2::generic(b));
}

inline u32x2 operator !=(const f32x2 a, const f32x2 b) {
	return u32x2(f32x2::generic(a) != f32x2::generic(b));
}

inline u32x2 operator !=(const s32x2 a, const s32x2 b) { 
	return u32x2(s32x2::generic(a) != s32x2::generic(b));
}

inline u32x2 operator !=(const u32x2 a, const u32x2 b) { 
	return u32x2(u32x2::generic(a) != u32x2::generic(b));
}

inline u32x2 operator <(const f32x2 a, const f32x2 b) {
	return u32x2(f32x2::generic(a) < f32x2::generic(b));
}

inline u32x2 operator <(const s32x2 a, const s32x2 b) {
	return u32x2(s32x2::generic(a) < s32x2::generic(b));
}

inline u32x2 operator <(const u32x2 a, const u32x2 b) { 
	return u32x2(u32x2::generic(a) < u32x2::generic(b));
}

inline u32x2 operator >(const f32x2 a, const f32x2 b) {
	return u32x2(f32x2::generic(a) > f32x2::generic(b));
}

inline u32x2 operator >(const s32x2 a, const s32x2 b) {
	return u32x2(s32x2::generic(a) > s32x2::generic(b));
}

inline u32x2 operator >(const u32x2 a, const u32x2 b) {
	return u32x2(u32x2::generic(a) > u32x2::generic(b));
}

inline u32x2 operator <=(const f32x2 a, const f32x2 b) {
	return u32x2(f32x2::generic(a) <= f32x2::generic(b));
}

inline u32x2 operator <=(const s32x2 a, const s32x2 b) {
	return u32x2(s32x2::generic(a) <= s32x2::generic(b));
}

inline u32x2 operator <=(const u32x2 a, const u32x2 b) {
	return u32x2(u32x2::generic(a) <= u32x2::generic(b));
}

inline u32x2 operator >=(const f32x2 a, const f32x2 b) {
	return u32x2(f32x2::generic(a) >= f32x2::generic(b));
}

inline u32x2 operator >=(const s32x2 a, const s32x2 b) {
	return u32x2(s32x2::generic(a) >= s32x2::generic(b));
}

inline u32x2 operator >=(const u32x2 a, const u32x2 b) {
	return u32x2(u32x2::generic(a) >= u32x2::generic(b));
}

#endif
#if NATIVE_U64X4 != 0
inline u64x4 operator ==(const f64x4 a, const f64x4 b) {
	return u64x4(f64x4::generic(a) == f64x4::generic(b));
}

inline u64x4 operator ==(const s64x4 a, const s64x4 b) {
	return u64x4(s64x4::generic(a) == s64x4::generic(b));
}

inline u64x4 operator ==(const u64x4 a, const u64x4 b) {
	return u64x4(u64x4::generic(a) == u64x4::generic(b));
}

inline u64x4 operator !=(const f64x4 a, const f64x4 b) {
	return u64x4(f64x4::generic(a) != f64x4::generic(b));
}

inline u64x4 operator !=(const s64x4 a, const s64x4 b) {
	return u64x4(s64x4::generic(a) != s64x4::generic(b));
}

inline u64x4 operator !=(const u64x4 a, const u64x4 b) {
	return u64x4(u64x4::generic(a) != u64x4::generic(b));
}

inline u64x4 operator <(const f64x4 a, const f64x4 b) {
	return u64x4(f64x4::generic(a) < f64x4::generic(b));
}

inline u64x4 operator <(const s64x4 a, const s64x4 b) {
	return u64x4(s64x4::generic(a) < s64x4::generic(b));
}

inline u64x4 operator <(const u64x4 a, const u64x4 b) {
	return u64x4(u64x4::generic(a) < u64x4::generic(b));
}

inline u64x4 operator >(const f64x4 a, const f64x4 b) {
	return u64x4(f64x4::generic(a) > f64x4::generic(b));
}

inline u64x4 operator >(const s64x4 a, const s64x4 b) {
	return u64x4(s64x4::generic(a) > s64x4::generic(b));
}

inline u64x4 operator >(const u64x4 a, const u64x4 b) {
	return u64x4(u64x4::generic(a) > u64x4::generic(b));
}

inline u64x4 operator <=(const f64x4 a, const f64x4 b) {
	return u64x4(f64x4::generic(a) <= f64x4::generic(b));
}

inline u64x4 operator <=(const s64x4 a, const s64x4 b) {
	return u64x4(s64x4::generic(a) <= s64x4::generic(b));
}

inline u64x4 operator <=(const u64x4 a, const u64x4 b) {
	return u64x4(u64x4::generic(a) <= u64x4::generic(b));
}

inline u64x4 operator >=(const f64x4 a, const f64x4 b) {
	return u64x4(f64x4::generic(a) >= f64x4::generic(b));
}

inline u64x4 operator >=(const s64x4 a, const s64x4 b) {
	return u64x4(s64x4::generic(a) >= s64x4::generic(b));
}

inline u64x4 operator >=(const u64x4 a, const u64x4 b) {
	return u64x4(u64x4::generic(a) >= u64x4::generic(b));
}

#endif
#if NATIVE_U32X4 != 0
inline u32x4 operator ==(const f32x4 a, const f32x4 b) {
	return u32x4(f32x4::generic(a) == f32x4::generic(b));
}

inline u32x4 operator ==(const s32x4 a, const s32x4 b) {
	return u32x4(s32x4::generic(a) == s32x4::generic(b));
}

inline u32x4 operator ==( const u32x4 a, const u32x4 b) {
	return u32x4(u32x4::generic(a) == u32x4::generic(b));
}

inline u32x4 operator !=(const f32x4 a, const f32x4 b) {
	return u32x4(f32x4::generic(a) != f32x4::generic(b));
}

inline u32x4 operator !=(const s32x4 a, const s32x4 b) {
	return u32x4(s32x4::generic(a) != s32x4::generic(b));
}

inline u32x4 operator !=(const u32x4 a, const u32x4 b) {
	return u32x4(u32x4::generic(a) != u32x4::generic(b));
}

inline u32x4 operator <(const f32x4 a, const f32x4 b) {
	return u32x4(f32x4::generic(a) < f32x4::generic(b));
}

inline u32x4 operator <(const s32x4 a, const s32x4 b) {
	return u32x4(s32x4::generic(a) < s32x4::generic(b));
}

inline u32x4 operator <(const u32x4 a, const u32x4 b) {
	return u32x4(u32x4::generic(a) < u32x4::generic(b));
}

inline u32x4 operator >(const f32x4 a, const f32x4 b) {
	return u32x4(f32x4::generic(a) > f32x4::generic(b));
}

inline u32x4 operator >(const s32x4 a, const s32x4 b) {
	return u32x4(s32x4::generic(a) > s32x4::generic(b));
}

inline u32x4 operator >(const u32x4 a, const u32x4 b) {
	return u32x4(u32x4::generic(a) > u32x4::generic(b));
}

inline u32x4 operator <=(const f32x4 a, const f32x4 b) {
	return u32x4(f32x4::generic(a) <= f32x4::generic(b));
}

inline u32x4 operator <=(const s32x4 a, const s32x4 b) {
	return u32x4(s32x4::generic(a) <= s32x4::generic(b));
}

inline u32x4 operator <=(const u32x4 a, const u32x4 b) {
	return u32x4(u32x4::generic(a) <= u32x4::generic(b));
}

inline u32x4 operator >=(const f32x4 a, const f32x4 b) {
	return u32x4(f32x4::generic(a) >= f32x4::generic(b));
}

inline u32x4 operator >=(const s32x4 a, const s32x4 b) {
	return u32x4(s32x4::generic(a) >= s32x4::generic(b));
}

inline u32x4 operator >=(const u32x4 a, const u32x4 b) {
	return u32x4(u32x4::generic(a) >= u32x4::generic(b));
}

#endif
#if NATIVE_U16X4 != 0
inline u16x4 operator ==(const s16x4 a, const s16x4 b) {
	return u16x4(s16x4::generic(a) == s16x4::generic(b));
}

inline u16x4 operator ==(const u16x4 a, const u16x4 b) {
	return u16x4(u16x4::generic(a) == u16x4::generic(b));
}

inline u16x4 operator !=(const s16x4 a, const s16x4 b) {
	return u16x4(s16x4::generic(a) != s16x4::generic(b));
}

inline u16x4 operator !=(const u16x4 a, const u16x4 b) {
	return u16x4(u16x4::generic(a) != u16x4::generic(b));
}

inline u16x4 operator <(const s16x4 a, const s16x4 b) {
	return u16x4(s16x4::generic(a) < s16x4::generic(b));
}

inline u16x4 operator <(const u16x4 a, const u16x4 b) {
	return u16x4(u16x4::generic(a) < u16x4::generic(b));
}

inline u16x4 operator >(const s16x4 a, const s16x4 b) {
	return u16x4(s16x4::generic(a) > s16x4::generic(b));
}

inline u16x4 operator >(const u16x4 a, const u16x4 b) {
	return u16x4(u16x4::generic(a) > u16x4::generic(b));
}

inline u16x4 operator <=(const s16x4 a, const s16x4 b) {
	return u16x4(s16x4::generic(a) <= s16x4::generic(b));
}

inline u16x4 operator <=(const u16x4 a, const u16x4 b) {
	return u16x4(u16x4::generic(a) <= u16x4::generic(b));
}

inline u16x4 operator >=(const s16x4 a, const s16x4 b) {
	return u16x4(s16x4::generic(a) >= s16x4::generic(b));
}

inline u16x4 operator >=(const u16x4 a, const u16x4 b) {
	return u16x4(u16x4::generic(a) >= u16x4::generic(b));
}

#endif
#if NATIVE_U32X8 != 0
inline u32x8 operator ==(const f32x8 a, const f32x8 b) {
	return u32x8(f32x8::generic(a) == f32x8::generic(b));
}

inline u32x8 operator ==(const s32x8 a, const s32x8 b) {
	return u32x8(s32x8::generic(a) == s32x8::generic(b));
}

inline u32x8 operator ==(const u32x8 a, const u32x8 b) {
	return u32x8(u32x8::generic(a) == u32x8::generic(b));
}

inline u32x8 operator !=(const f32x8 a, const f32x8 b) {
	return u32x8(f32x8::generic(a) != f32x8::generic(b));
}

inline u32x8 operator !=(const s32x8 a, const s32x8 b) {
	return u32x8(s32x8::generic(a) != s32x8::generic(b));
}

inline u32x8 operator !=(const u32x8 a, const u32x8 b) {
	return u32x8(u32x8::generic(a) != u32x8::generic(b));
}

inline u32x8 operator <(const f32x8 a, const f32x8 b) {
	return u32x8(f32x8::generic(a) < f32x8::generic(b));
}

inline u32x8 operator <(const s32x8 a, const s32x8 b) {
	return u32x8(s32x8::generic(a) < s32x8::generic(b));
}

inline u32x8 operator <(const u32x8 a, const u32x8 b) {
	return u32x8(u32x8::generic(a) < u32x8::generic(b));
}

inline u32x8 operator >(const f32x8 a, const f32x8 b) {
	return u32x8(f32x8::generic(a) > f32x8::generic(b));
}

inline u32x8 operator >(const s32x8 a, const s32x8 b) {
	return u32x8(s32x8::generic(a) > s32x8::generic(b));
}

inline u32x8 operator >(const u32x8 a, const u32x8 b) {
	return u32x8(u32x8::generic(a) > u32x8::generic(b));
}

inline u32x8 operator <=(const f32x8 a, const f32x8 b) {
	return u32x8(f32x8::generic(a) <= f32x8::generic(b));
}

inline u32x8 operator <=(const s32x8 a, const s32x8 b) {
	return u32x8(s32x8::generic(a) <= s32x8::generic(b));
}

inline u32x8 operator <=(const u32x8 a, const u32x8 b) {
	return u32x8(u32x8::generic(a) <= u32x8::generic(b));
}

inline u32x8 operator >=(const f32x8 a, const f32x8 b) {
	return u32x8(f32x8::generic(a) >= f32x8::generic(b));
}

inline u32x8 operator >=(const s32x8 a, const s32x8 b) {
	return u32x8(s32x8::generic(a) >= s32x8::generic(b));
}

inline u32x8 operator >=(const u32x8 a, const u32x8 b) {
	return u32x8(u32x8::generic(a) >= u32x8::generic(b));
}

#endif
#if NATIVE_U16X8 != 0
inline u16x8 operator ==(const s16x8 a, const s16x8 b) {
	return u16x8(s16x8::generic(a) == s16x8::generic(b));
}

inline u16x8 operator ==(const u16x8 a, const u16x8 b) {
	return u16x8(u16x8::generic(a) == u16x8::generic(b));
}

inline u16x8 operator !=(const s16x8 a, const s16x8 b) {
	return u16x8(s16x8::generic(a) != s16x8::generic(b));
}

inline u16x8 operator !=(const u16x8 a, const u16x8 b) {
	return u16x8(u16x8::generic(a) != u16x8::generic(b));
}

inline u16x8 operator <(const s16x8 a, const s16x8 b) {
	return u16x8(s16x8::generic(a) < s16x8::generic(b));
}

inline u16x8 operator <(const u16x8 a, const u16x8 b) {
	return u16x8(u16x8::generic(a) < u16x8::generic(b));
}

inline u16x8 operator >(const s16x8 a, const s16x8 b) {
	return u16x8(s16x8::generic(a) > s16x8::generic(b));
}

inline u16x8 operator >(const u16x8 a, const u16x8 b) {
	return u16x8(u16x8::generic(a) > u16x8::generic(b));
}

inline u16x8 operator <=(const s16x8 a, const s16x8 b) {
	return u16x8(s16x8::generic(a) <= s16x8::generic(b));
}

inline u16x8 operator <=(const u16x8 a, const u16x8 b) {
	return u16x8(u16x8::generic(a) <= u16x8::generic(b));
}

inline u16x8 operator >=(const s16x8 a, const s16x8 b) {
	return u16x8(s16x8::generic(a) >= s16x8::generic(b));
}

inline u16x8 operator >=(const u16x8 a, const u16x8 b) {
	return u16x8(u16x8::generic(a) >= u16x8::generic(b));
}

#endif
#if NATIVE_U16X16 != 0
inline u16x16 operator ==(const s16x16 a, const s16x16 b) {
	return u16x16(s16x16::generic(a) == s16x16::generic(b));
}

inline u16x16 operator ==(const u16x16 a, const u16x16 b) {
	return u16x16(u16x16::generic(a) == u16x16::generic(b));
}

inline u16x16 operator !=(const s16x16 a, const s16x16 b) {
	return u16x16(s16x16::generic(a) != s16x16::generic(b));
}

inline u16x16 operator !=(const u16x16 a, const u16x16 b) {
	return u16x16(u16x16::generic(a) != u16x16::generic(b));
}

inline u16x16 operator <(const s16x16 a, const s16x16 b) {
	return u16x16(s16x16::generic(a) < s16x16::generic(b));
}

inline u16x16 operator <(const u16x16 a, const u16x16 b) {
	return u16x16(u16x16::generic(a) < u16x16::generic(b));
}

inline u16x16 operator >(const s16x16 a, const s16x16 b) {
	return u16x16(s16x16::generic(a) > s16x16::generic(b));
}

inline u16x16 operator >(const u16x16 a, const u16x16 b) {
	return u16x16(u16x16::generic(a) > u16x16::generic(b));
}

inline u16x16 operator <=(const s16x16 a, const s16x16 b) {
	return u16x16(s16x16::generic(a) <= s16x16::generic(b));
}

inline u16x16 operator <=(const u16x16 a, const u16x16 b) {
	return u16x16(u16x16::generic(a) <= u16x16::generic(b));
}

inline u16x16 operator >=(const s16x16 a, const s16x16 b) {
	return u16x16(s16x16::generic(a) >= s16x16::generic(b));
}

inline u16x16 operator >=(const u16x16 a, const u16x16 b) {
	return u16x16(u16x16::generic(a) >= u16x16::generic(b));
}

#endif
#endif
} // namespace simd

#endif // vectnative_H__
