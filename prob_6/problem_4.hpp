#ifndef prob_4_H__
#define prob_4_H__

#if __AVX__ != 0
	#include <immintrin.h>
#elif __SSE4_1__ != 0
	#include <smmintrin.h>
#else
	#include <emmintrin.h>
#endif
#include <assert.h>
#include <stdint.h>
#include <istream>
#include <ostream>
#include <limits>
#include "vectsimd_sse.hpp"
#include "array.hpp"
#include "scoped.hpp"

template < bool >
struct compile_assert;

template <>
struct compile_assert< true >
{
	compile_assert()
	{
	}
};


class Ray
{
	simd::vect3 m_origin;
	simd::vect3 m_direction;
	simd::vect3 m_rcpdir;

#if __AVX__ == 0
	// pre-AVX Intels don't have efficient m32 broadcast ops,
	// so store pre-baked x, y and z broadcasts as members
	__m128 m_origin_x;
	__m128 m_origin_y;
	__m128 m_origin_z;

	__m128 m_rcpdir_x;
	__m128 m_rcpdir_y;
	__m128 m_rcpdir_z;

#endif
public:
	Ray()
	{
	}

	Ray(
		const simd::vect3& origin,
		const simd::vect3& direction)
	: m_origin(origin)
	, m_direction(direction)
	{
#if RAY_HIGH_PRECISION_RCP_DIR == 1
		const __m128 rcp = _mm_div_ps(_mm_set1_ps(1.f), direction.getn());

#else
		const __m128 rcp = _mm_rcp_ps(direction.getn());

#endif
		// substitute infinities for max floats in the reciprocal direction
		m_rcpdir.setn(0, _mm_max_ps(_mm_min_ps(rcp,
			_mm_set1_ps( std::numeric_limits< float >::max())),
			_mm_set1_ps(-std::numeric_limits< float >::max())));

#if __AVX__ == 0
		m_origin_x = _mm_shuffle_ps(origin.getn(), origin.getn(), 0);
		m_origin_y = _mm_shuffle_ps(origin.getn(), origin.getn(), 0x55);
		m_origin_z = _mm_shuffle_ps(origin.getn(), origin.getn(), 0xaa);

		m_rcpdir_x = _mm_shuffle_ps(m_rcpdir.getn(), m_rcpdir.getn(), 0);
		m_rcpdir_y = _mm_shuffle_ps(m_rcpdir.getn(), m_rcpdir.getn(), 0x55);
		m_rcpdir_z = _mm_shuffle_ps(m_rcpdir.getn(), m_rcpdir.getn(), 0xaa);

#endif
	}

	const simd::vect3&
	get_origin() const
	{
		return m_origin;
	}

	const simd::vect3&
	get_direction() const
	{
		return m_direction;
	}

	const simd::vect3&
	get_rcpdir() const
	{
		return m_rcpdir;
	}

#if __AVX__ != 0
	__m256 get_origin_x() const
	{
		return _mm256_set1_ps(m_origin[0]);
	}

	__m256 get_origin_y() const
	{
		return _mm256_set1_ps(m_origin[1]);
	}

	__m256 get_origin_z() const
	{
		return _mm256_set1_ps(m_origin[2]);
	}

	__m256 get_rcpdir_x() const
	{
		return _mm256_set1_ps(m_rcpdir[0]);
	}

	__m256 get_rcpdir_y() const
	{
		return _mm256_set1_ps(m_rcpdir[1]);
	}

	__m256 get_rcpdir_z() const
	{
		return _mm256_set1_ps(m_rcpdir[2]);
	}

#else
	__m128 get_origin_x() const
	{
		return m_origin_x;
	}

	__m128 get_origin_y() const
	{
		return m_origin_y;
	}

	__m128 get_origin_z() const
	{
		return m_origin_z;
	}

	__m128 get_rcpdir_x() const
	{
		return m_rcpdir_x;
	}

	__m128 get_rcpdir_y() const
	{
		return m_rcpdir_y;
	}

	__m128 get_rcpdir_z() const
	{
		return m_rcpdir_z;
	}

#endif
	bool
	flatten(
		std::ostream& out) const;

	bool
	deflatten(
		std::istream& in);
};


class BBox
{
	union {
		__m128 m_min;
		uint32_t umin[4];
	};
	union {
		__m128 m_max;
		uint32_t umax[4];
	};

public:
	BBox(
		const BBox& oth)
	{
		m_min = oth.m_min;
		m_max = oth.m_max;
	}

	BBox& operator =(
		const BBox& oth)
	{
		m_min = oth.m_min;
		m_max = oth.m_max;
		return *this;
	}

	enum flag_direct {};
	enum flag_noinit {};

	BBox()
	: m_min(_mm_set1_ps( std::numeric_limits< float >::infinity()))
	, m_max(_mm_set1_ps(-std::numeric_limits< float >::infinity()))
	{
	}

	BBox(
		const flag_noinit)
	{
	}

	bool
	is_valid() const
	{
		return 7 == (7 & _mm_movemask_ps(_mm_cmple_ps(m_min, m_max)));
	}

	BBox(
		const __m128 min,
		const __m128 max,
		const flag_direct)
	: m_min(min)
	, m_max(max)
	{
		assert(is_valid());
	}

	BBox(
		const simd::vect3& min,
		const simd::vect3& max,
		const flag_direct)
	{
		*this = BBox(min.getn(), max.getn(), flag_direct());
	}

	BBox(
		const __m128 v0,
		const __m128 v1)
	{
		m_min = _mm_min_ps(v0, v1);
		m_max = _mm_max_ps(v0, v1);

		assert(is_valid());
	}

	BBox(
		const simd::vect3& v0,
		const simd::vect3& v1)
	{
		*this = BBox(v0.getn(), v1.getn());
	}

	const __m128
	get_min() const
	{
		return m_min;
	}

	const __m128
	get_max() const
	{
		return m_max;
	}

	BBox&
	grow(
		const simd::vect3& v)
	{
		const BBox min = BBox(m_min, v.getn());
		const BBox max = BBox(m_max, v.getn());

		m_min = min.get_min();
		m_max = max.get_max();

		return *this;
	}

	BBox&
	grow(
		const BBox& oth)
	{
		const BBox min = BBox(m_min, oth.m_min);
		const BBox max = BBox(m_max, oth.m_max);

		m_min = min.get_min();
		m_max = max.get_max();

		return *this;
	}

	BBox&
	overlap(
		const BBox& oth)
	{
		m_min = _mm_max_ps(m_min, oth.m_min);
		m_max = _mm_min_ps(m_max, oth.m_max);

		assert(is_valid());

		return *this;
	}

	// warning: use this mutator only after the box is done with grow/overlap!
	void set_min_cookie(
		const uint32_t cookie)
	{
		umin[3] = cookie;
	}

	// warning: use this mutator only after the box is done with grow/overlap!
	void set_max_cookie(
		const uint32_t cookie)
	{
		umax[3] = cookie;
	}

	uint32_t get_min_cookie() const
	{
		return umin[3];
	}

	uint32_t get_max_cookie() const
	{
		return umax[3];
	}

	bool
	intersect(
		const Ray& ray,
		float (&t)[2]) const;

	bool
	intersect(
		const Ray& ray,
		__m128& min_mask,
		int& a_mask,
		int& b_mask,
		float& t) const;

	bool
	has_overlap_open(
		const BBox& oth) const;

	bool
	has_overlap_closed(
		const BBox& oth) const;

	bool
	contains_open(
		const simd::vect3& v) const;

	bool
	contains_closed(
		const simd::vect3& v) const;

	bool
	contains_open(
		const BBox& oth) const;

	bool
	contains_closed(
		const BBox& oth) const;

	template < size_t SUBDIM_T >
	bool
	has_overlap_open_subdim(
		const BBox& oth) const;

	template < size_t SUBDIM_T >
	bool
	has_overlap_closed_subdim(
		const BBox& oth) const;

	template < size_t SUBDIM_T >
	bool
	contains_open_subdim(
		const simd::vect3& v) const;

	template < size_t SUBDIM_T >
	bool
	contains_closed_subdim(
		const simd::vect3& v) const;

	template < size_t SUBDIM_T >
	bool
	contains_open_subdim(
		const BBox& oth) const;

	template < size_t SUBDIM_T >
	bool
	contains_closed_subdim(
		const BBox& oth) const;
};


template < size_t SUBDIM_T >
inline bool
BBox::has_overlap_open_subdim(
	const BBox& oth) const
{
	const compile_assert< 3 >= SUBDIM_T > assert_subdim;

	for (size_t i = 0; i < SUBDIM_T; ++i)
		if (m_min[i] >= oth.m_max[i])
			return false;

	for (size_t i = 0; i < SUBDIM_T; ++i)
		if (m_max[i] <= oth.m_min[i])
			return false;

	return true;
}


template < size_t SUBDIM_T >
inline bool
BBox::has_overlap_closed_subdim(
	const BBox& oth) const
{
	const compile_assert< 3 >= SUBDIM_T > assert_subdim;

	for (size_t i = 0; i < SUBDIM_T; ++i)
		if (m_min[i] > oth.m_max[i])
			return false;

	for (size_t i = 0; i < SUBDIM_T; ++i)
		if (m_max[i] < oth.m_min[i])
			return false;

	return true;
}


template < size_t SUBDIM_T >
inline bool
BBox::contains_open_subdim(
	const simd::vect3& v) const
{
	const compile_assert< 3 >= SUBDIM_T > assert_subdim;

	for (size_t i = 0; i < SUBDIM_T; ++i)
		if (m_min[i] >= v[i])
			return false;

	for (size_t i = 0; i < SUBDIM_T; ++i)
		if (m_max[i] <= v[i])
			return false;

	return true;
}


template < size_t SUBDIM_T >
inline bool
BBox::contains_closed_subdim(
	const simd::vect3& v) const
{
	const compile_assert< 3 >= SUBDIM_T > assert_subdim;

	for (size_t i = 0; i < SUBDIM_T; ++i)
		if (m_min[i] > v[i])
			return false;

	for (size_t i = 0; i < SUBDIM_T; ++i)
		if (m_max[i] < v[i])
			return false;

	return true;
}


template < size_t SUBDIM_T >
inline bool
BBox::contains_open_subdim(
	const BBox& oth) const
{
	const compile_assert< 3 >= SUBDIM_T > assert_subdim;

	for (size_t i = 0; i < SUBDIM_T; ++i)
		if (m_min[i] >= oth.m_min[i])
			return false;

	for (size_t i = 0; i < SUBDIM_T; ++i)
		if (m_max[i] <= oth.m_max[i])
			return false;

	return true;
}


template < size_t SUBDIM_T >
inline bool
BBox::contains_closed_subdim(
	const BBox& oth) const
{
	const compile_assert< 3 >= SUBDIM_T > assert_subdim;

	for (size_t i = 0; i < SUBDIM_T; ++i)
		if (m_min[i] > oth.m_min[i])
			return false;

	for (size_t i = 0; i < SUBDIM_T; ++i)
		if (m_max[i] < oth.m_max[i])
			return false;

	return true;
}


inline bool
BBox::has_overlap_open(
	const BBox& oth) const
{
	return has_overlap_open_subdim< 3 >(oth);
}


inline bool
BBox::has_overlap_closed(
	const BBox& oth) const
{
	return has_overlap_closed_subdim< 3 >(oth);
}


inline bool
BBox::contains_open(
	const simd::vect3& v) const
{
	return contains_open_subdim< 3 >(v);
}


inline bool
BBox::contains_closed(
	const simd::vect3& v) const
{
	return contains_closed_subdim< 3 >(v);
}


inline bool
BBox::contains_open(
	const BBox& oth) const
{
	return contains_open_subdim< 3 >(oth);
}


inline bool
BBox::contains_closed(
	const BBox& oth) const
{
	return contains_closed_subdim< 3 >(oth);
}

//
// ray/box intersection yielding both boolean and numeric results
//

inline bool __attribute__ ((always_inline))
BBox::intersect(
	const Ray& ray,
	float (& t)[2]) const
{
	// the parametric dot(normal, origin + t * direction) = distance
	// yields t = (distance - dot(normal, origin)) / dot(normal, direction)

	const __m128 t0 = _mm_mul_ps(_mm_sub_ps(m_min, ray.get_origin().getn()), ray.get_rcpdir().getn());
	const __m128 t1 = _mm_mul_ps(_mm_sub_ps(m_max, ray.get_origin().getn()), ray.get_rcpdir().getn());

	const __m128 x_min = _mm_min_ps(t0, t1);
	const __m128 x_max = _mm_max_ps(t0, t1);
	const __m128 y_min = _mm_shuffle_ps(x_min, x_min, 0xe5);
	const __m128 y_max = _mm_shuffle_ps(x_max, x_max, 0xe5);
	const __m128 z_min = _mm_shuffle_ps(x_min, x_min, 0xe6);
	const __m128 z_max = _mm_shuffle_ps(x_max, x_max, 0xe6);

	const __m128 min = _mm_max_ss(_mm_max_ss(x_min, y_min), z_min);
	const __m128 max = _mm_min_ss(_mm_min_ss(x_max, y_max), z_max);

	t[0] = _mm_cvtss_f32(min);
	t[1] = _mm_cvtss_f32(max);

	// discard non-intersections (min >= max) and intersections at non-positive distances
	return _mm_comilt_ss(min, max) & _mm_comilt_ss(_mm_setzero_ps(), max);
}

//
// ray/box intersection yielding both boolean and numeric results
//

inline bool __attribute__ ((always_inline))
BBox::intersect(
	const Ray& ray,
	__m128& min_mask,
	int& a_mask,
	int& b_mask,
	float& t) const
{
	// the parametric dot(normal, origin + t * direction) = distance
	// yields t = (distance - dot(normal, origin)) / dot(normal, direction)

	const __m128 t0 = _mm_mul_ps(_mm_sub_ps(m_min, ray.get_origin().getn()), ray.get_rcpdir().getn());
	const __m128 t1 = _mm_mul_ps(_mm_sub_ps(m_max, ray.get_origin().getn()), ray.get_rcpdir().getn());
	min_mask = _mm_cmple_ps(t0, t1);

	const __m128 x_min = _mm_min_ps(t0, t1);
	const __m128 x_max = _mm_max_ps(t0, t1);
	const __m128 y_min = _mm_shuffle_ps(x_min, x_min, 0xe5);
	const __m128 y_max = _mm_shuffle_ps(x_max, x_max, 0xe5);
	const __m128 z_min = _mm_shuffle_ps(x_min, x_min, 0xe6);
	const __m128 z_max = _mm_shuffle_ps(x_max, x_max, 0xe6);

	const __m128 min_a_mask = _mm_cmpge_ss(x_min, y_min);
	const __m128 min_b_mask = _mm_cmpge_ss(_mm_max_ss(x_min, y_min), z_min);

	a_mask = _mm_cvtsi128_si32(_mm_castps_si128(min_a_mask));
	b_mask = _mm_cvtsi128_si32(_mm_castps_si128(min_b_mask));

	const __m128 min = _mm_max_ss(_mm_max_ss(x_min, y_min), z_min);
	const __m128 max = _mm_min_ss(_mm_min_ss(x_max, y_max), z_max);

	t = _mm_cvtss_f32(min);

	// discard non-intersections (min > max) and intersections at negative entry distances
	return _mm_comile_ss(min, max) & _mm_comile_ss(_mm_setzero_ps(), min);
}

//
// ray/dual-box intersection yielding boolean results
//

inline void __attribute__ ((always_inline))
intersect2(
	const BBox* const bbox,
	const Ray& ray,
	uint32_t (& r)[2])
{
	assert(0 != bbox);

	// the parametric dot(normal, origin + t * direction) = distance
	// yields t = (distance - dot(normal, origin)) / dot(normal, direction)

	const __m128 t0 = _mm_mul_ps(_mm_sub_ps(bbox[0].get_min(), ray.get_origin().getn()), ray.get_rcpdir().getn());
	const __m128 t1 = _mm_mul_ps(_mm_sub_ps(bbox[0].get_max(), ray.get_origin().getn()), ray.get_rcpdir().getn());
	const __m128 t2 = _mm_mul_ps(_mm_sub_ps(bbox[1].get_min(), ray.get_origin().getn()), ray.get_rcpdir().getn());
	const __m128 t3 = _mm_mul_ps(_mm_sub_ps(bbox[1].get_max(), ray.get_origin().getn()), ray.get_rcpdir().getn());

	const __m128 x_min0 = _mm_min_ps(t0, t1);
	const __m128 x_max0 = _mm_max_ps(t0, t1);
	const __m128 x_min1 = _mm_min_ps(t2, t3);
	const __m128 x_max1 = _mm_max_ps(t2, t3);

	// AoS -> SoA
	const __m128 x_min = _mm_unpacklo_ps(x_min0, x_min1);
	const __m128 x_max = _mm_unpacklo_ps(x_max0, x_max1);
	const __m128 z_min = _mm_unpackhi_ps(x_min0, x_min1);
	const __m128 z_max = _mm_unpackhi_ps(x_max0, x_max1);
	const __m128 y_min = _mm_movehl_ps(x_min, x_min);
	const __m128 y_max = _mm_movehl_ps(x_max, x_max);

	const __m128 min = _mm_max_ps(_mm_max_ps(x_min, z_min), y_min);
	const __m128 max = _mm_min_ps(_mm_min_ps(x_max, z_max), y_max);

	// discard non-intersections (min >= max) and intersections at non-positive distances
	const __m128 res = _mm_and_ps(_mm_cmplt_ps(min, max), _mm_cmplt_ps(_mm_setzero_ps(), max));

	r[0] = uint32_t(_mm_castps_si128(res)[0] >>  0);
	r[1] = uint32_t(_mm_castps_si128(res)[0] >> 32);
}

//
// ray/octo-box intersection yielding both boolean and numeric results (max t)
//

#if __AVX__ != 0
inline void __attribute__ ((always_inline))
intersect8(
	const __m256 & bbox_min_x,
	const __m256 & bbox_min_y,
	const __m256 & bbox_min_z,
	const __m256 & bbox_max_x,
	const __m256 & bbox_max_y,
	const __m256 & bbox_max_z,
	const Ray& ray,
	float (& t)[8],
	uint32_t (& r)[8])
{
	// the parametric dot(normal, origin + t * direction) = distance
	// yields t = (distance - dot(normal, origin)) / dot(normal, direction)

	const __m256 tmin_x = _mm256_mul_ps(_mm256_sub_ps(bbox_min_x, ray.get_origin_x()), ray.get_rcpdir_x());
	const __m256 tmax_x = _mm256_mul_ps(_mm256_sub_ps(bbox_max_x, ray.get_origin_x()), ray.get_rcpdir_x());
	const __m256 tmin_y = _mm256_mul_ps(_mm256_sub_ps(bbox_min_y, ray.get_origin_y()), ray.get_rcpdir_y());
	const __m256 tmax_y = _mm256_mul_ps(_mm256_sub_ps(bbox_max_y, ray.get_origin_y()), ray.get_rcpdir_y());
	const __m256 tmin_z = _mm256_mul_ps(_mm256_sub_ps(bbox_min_z, ray.get_origin_z()), ray.get_rcpdir_z());
	const __m256 tmax_z = _mm256_mul_ps(_mm256_sub_ps(bbox_max_z, ray.get_origin_z()), ray.get_rcpdir_z());

	const __m256 x_min = _mm256_min_ps(tmin_x, tmax_x);
	const __m256 x_max = _mm256_max_ps(tmin_x, tmax_x);
	const __m256 y_min = _mm256_min_ps(tmin_y, tmax_y);
	const __m256 y_max = _mm256_max_ps(tmin_y, tmax_y);
	const __m256 z_min = _mm256_min_ps(tmin_z, tmax_z);
	const __m256 z_max = _mm256_max_ps(tmin_z, tmax_z);

	const __m256 min = _mm256_max_ps(_mm256_max_ps(x_min, y_min), z_min);
	const __m256 max = _mm256_min_ps(_mm256_min_ps(x_max, y_max), z_max);

	// store t_max results
	_mm256_store_ps(t, max);

	// discard non-intersections (min >= max) and intersections at non-positive distances
	const __m256 msk = _mm256_and_ps(_mm256_cmp_ps(min, max, _CMP_LT_OQ), _mm256_cmp_ps(_mm256_setzero_ps(), max, _CMP_LT_OQ));

	// store logical (mask) results
	_mm256_store_si256((__m256i*) r, _mm256_castps_si256(msk));
}

#else // __AVX__ == 0
inline void __attribute__ ((always_inline))
intersect8(
	const __m128 (& bbox_min_x)[2],
	const __m128 (& bbox_min_y)[2],
	const __m128 (& bbox_min_z)[2],
	const __m128 (& bbox_max_x)[2],
	const __m128 (& bbox_max_y)[2],
	const __m128 (& bbox_max_z)[2],
	const Ray& ray,
	float (& t)[8],
	uint32_t (& r)[8])
{
	// the parametric dot(normal, origin + t * direction) = distance
	// yields t = (distance - dot(normal, origin)) / dot(normal, direction)

	const __m128 tmin_x0 = _mm_mul_ps(_mm_sub_ps(bbox_min_x[0], ray.get_origin_x()), ray.get_rcpdir_x());
	const __m128 tmax_x0 = _mm_mul_ps(_mm_sub_ps(bbox_max_x[0], ray.get_origin_x()), ray.get_rcpdir_x());
	const __m128 tmin_x1 = _mm_mul_ps(_mm_sub_ps(bbox_min_x[1], ray.get_origin_x()), ray.get_rcpdir_x());
	const __m128 tmax_x1 = _mm_mul_ps(_mm_sub_ps(bbox_max_x[1], ray.get_origin_x()), ray.get_rcpdir_x());
	const __m128 tmin_y0 = _mm_mul_ps(_mm_sub_ps(bbox_min_y[0], ray.get_origin_y()), ray.get_rcpdir_y());
	const __m128 tmax_y0 = _mm_mul_ps(_mm_sub_ps(bbox_max_y[0], ray.get_origin_y()), ray.get_rcpdir_y());
	const __m128 tmin_y1 = _mm_mul_ps(_mm_sub_ps(bbox_min_y[1], ray.get_origin_y()), ray.get_rcpdir_y());
	const __m128 tmax_y1 = _mm_mul_ps(_mm_sub_ps(bbox_max_y[1], ray.get_origin_y()), ray.get_rcpdir_y());
	const __m128 tmin_z0 = _mm_mul_ps(_mm_sub_ps(bbox_min_z[0], ray.get_origin_z()), ray.get_rcpdir_z());
	const __m128 tmax_z0 = _mm_mul_ps(_mm_sub_ps(bbox_max_z[0], ray.get_origin_z()), ray.get_rcpdir_z());
	const __m128 tmin_z1 = _mm_mul_ps(_mm_sub_ps(bbox_min_z[1], ray.get_origin_z()), ray.get_rcpdir_z());
	const __m128 tmax_z1 = _mm_mul_ps(_mm_sub_ps(bbox_max_z[1], ray.get_origin_z()), ray.get_rcpdir_z());

	const __m128 x_min0 = _mm_min_ps(tmin_x0, tmax_x0);
	const __m128 x_max0 = _mm_max_ps(tmin_x0, tmax_x0);
	const __m128 x_min1 = _mm_min_ps(tmin_x1, tmax_x1);
	const __m128 x_max1 = _mm_max_ps(tmin_x1, tmax_x1);
	const __m128 y_min0 = _mm_min_ps(tmin_y0, tmax_y0);
	const __m128 y_max0 = _mm_max_ps(tmin_y0, tmax_y0);
	const __m128 y_min1 = _mm_min_ps(tmin_y1, tmax_y1);
	const __m128 y_max1 = _mm_max_ps(tmin_y1, tmax_y1);
	const __m128 z_min0 = _mm_min_ps(tmin_z0, tmax_z0);
	const __m128 z_max0 = _mm_max_ps(tmin_z0, tmax_z0);
	const __m128 z_min1 = _mm_min_ps(tmin_z1, tmax_z1);
	const __m128 z_max1 = _mm_max_ps(tmin_z1, tmax_z1);

	const __m128 min0 = _mm_max_ps(_mm_max_ps(x_min0, y_min0), z_min0);
	const __m128 max0 = _mm_min_ps(_mm_min_ps(x_max0, y_max0), z_max0);
	const __m128 min1 = _mm_max_ps(_mm_max_ps(x_min1, y_min1), z_min1);
	const __m128 max1 = _mm_min_ps(_mm_min_ps(x_max1, y_max1), z_max1);

	// store t_max results
	_mm_store_ps(t + 0, max0);
	_mm_store_ps(t + 4, max1);

	// discard non-intersections (min >= max) and intersections at non-positive distances
	const __m128 msk0 = _mm_and_ps(_mm_cmplt_ps(min0, max0), _mm_cmplt_ps(_mm_setzero_ps(), max0));
	const __m128 msk1 = _mm_and_ps(_mm_cmplt_ps(min1, max1), _mm_cmplt_ps(_mm_setzero_ps(), max1));

	// store logical (mask) results
	_mm_store_si128((__m128i*)(r + 0), _mm_castps_si128(msk0));
	_mm_store_si128((__m128i*)(r + 4), _mm_castps_si128(msk1));
}

#endif // __AVX__ != 0

class Voxel
{
	BBox m_bbox;

public:
	Voxel()
	: m_bbox(BBox::flag_noinit())
	{
	}

	Voxel(
		const simd::vect3& min,
		const simd::vect3& max)
	: m_bbox(min, max, BBox::flag_direct())
	{
	}

	enum flag_unordered {};

	Voxel(
		const simd::vect3& p0,
		const simd::vect3& p1,
		const flag_unordered)
	: m_bbox(p0, p1)
	{
	}

	Voxel(
		const BBox& bbox,
		const uint32_t id)
	: m_bbox(bbox)
	{
		m_bbox.set_min_cookie(id);
	}

	const BBox&
	get_bbox() const
	{
		return m_bbox;
	}

	const simd::vect3
	get_min() const
	{
		simd::vect3 r;
		r.setn(0, m_bbox.get_min());
		return r;
	}

	const simd::vect3
	get_max() const
	{
		simd::vect3 r;
		r.setn(0, m_bbox.get_max());
		return r;
	}

	uint32_t get_id() const
	{
		return m_bbox.get_min_cookie();
	}

	bool
	flatten(
		std::ostream& out) const;

	bool
	deflatten(
		std::istream& in);
};


enum {
	octree_level_root,

#if MINIMAL_TREE != 0
	octree_level_last_but_one = octree_level_root,

#else
	octree_level_interior_0,

#if BIG_TREE != 0
	octree_level_interior_1,
	octree_level_last_but_one = octree_level_interior_1,

#else
	octree_level_last_but_one = octree_level_interior_0,

#endif
#endif
	octree_level_leaf,
	octree_level_count
};


enum {
	octree_interior_count =

#if MINIMAL_TREE == 0
#if BIG_TREE != 0
		(1 << 3 * octree_level_interior_1) +

#endif
		(1 << 3 * octree_level_interior_0) +

#endif
		(1 << 3 * octree_level_root),

	octree_leaf_count = 1 << 3 * (octree_level_count - 1),
	octree_cell_count = 1 << 3 * (octree_level_count - 0),
	octree_axis_granularity = 1 << octree_level_count
};


inline unsigned
local2index(
	const unsigned x,
	const unsigned y,
	const unsigned z)
{
	assert(2 > x);
	assert(2 > y);
	assert(2 > z);

	return z << 2 | y << 1 | x;
}


inline void
index2local(
	const unsigned index,
	unsigned& x,
	unsigned& y,
	unsigned& z)
{
	assert(8 > index);

	x = index >> 0 & 1;
	y = index >> 1 & 1;
	z = index >> 2 & 1;
}


inline void
global2local(
	const unsigned level,
	const unsigned x,
	const unsigned y,
	const unsigned z,
	unsigned& local_x,
	unsigned& local_y,
	unsigned& local_z)
{
	assert(octree_axis_granularity > x);
	assert(octree_axis_granularity > y);
	assert(octree_axis_granularity > z);

	local_x = x >> (octree_level_count - level - 1) & 1;
	local_y = y >> (octree_level_count - level - 1) & 1;
	local_z = z >> (octree_level_count - level - 1) & 1;
}


typedef uint16_t OctetId; // integral type capable of holding the amount of leaves
typedef uint16_t PayloadId; // integral type capable of holding the amount of leaf payload

enum {
	cell_capacity = 64 // octree cell capacity during building
};

enum {
	octree_payload_count = octree_cell_count * cell_capacity
};

static const compile_assert< (size_t(1) << sizeof(OctetId) * 8 > octree_leaf_count) > assert_octet_id;
static const compile_assert< (size_t(1) << sizeof(PayloadId) * 8 > octree_payload_count) > assert_payload_id;


class __attribute__ ((aligned(16))) Octet
{
	enum { capacity = 8 };

	OctetId m_child[capacity];

public:
	Octet()
	{
		for (size_t i = 0; i < capacity; ++i)
			m_child[i] = OctetId(-1);
	}

	void
	set(
		const size_t index,
		const OctetId child)
	{
		assert(capacity > index);
		m_child[index] = child;
	}

	OctetId
	get(
		const size_t index) const
	{
		assert(capacity > index);
		return m_child[index];
	}

	bool
	empty(
		const size_t index) const
	{
		assert(capacity > index);
		return OctetId(-1) == m_child[index];
	}

	__m128i
	get_occupancy() const
	{
		const compile_assert< sizeof(*this) == sizeof(__m128i) > assert_octet_size;
		return _mm_cmpeq_epi16(_mm_set1_epi16(-1), reinterpret_cast< const __m128i* >(this)[0]);
	}
};


class __attribute__ ((aligned(16))) Leaf
{
	enum { capacity = 8 };

	PayloadId m_start[capacity];
	PayloadId m_count[capacity];

public:
	void init(
		const PayloadId start)
	{
		for (size_t i = 0; i < capacity; ++i)
		{
			m_start[i] = start + PayloadId(i * cell_capacity);
			m_count[i] = 0;
		}
	}

	PayloadId
	get_start(
		const size_t index) const
	{
		assert(capacity > index);
		return m_start[index];
	}

	PayloadId
	get_count(
		const size_t index) const
	{
		assert(capacity > index);
		return m_count[index];
	}

	bool
	add(
		const size_t index,
		Array< Voxel >& payload,
		const Voxel& item)
	{
		assert(capacity > index);

		const size_t cell_start = m_start[index];
		const size_t cell_count = m_count[index];

		if (cell_capacity == cell_count)
			return false;

		m_count[index] = PayloadId(cell_count + 1);
		payload.getMutable(cell_start + cell_count) = item;
		return true;
	}

	bool
	empty(
		const size_t index) const
	{
		assert(capacity > index);
		return 0 == m_count[index];
	}

	__m128i
	get_occupancy() const
	{
		const compile_assert< sizeof(*this) == sizeof(__m128i) * 2 > assert_leaf_size;
		return _mm_cmpeq_epi16(_mm_set1_epi16(0), reinterpret_cast< const __m128i* >(this)[1]);
	}

	void
	compact(
		size_t& cursor,
		Array< Voxel >& payload)
	{
		for (size_t i = 0; i < capacity; ++i)
		{
			const size_t cell_start = m_start[i];
			const size_t cell_count = m_count[i];

			if (0 == cell_count)
				continue;

			for (size_t j = 0; j < cell_count; ++j)
				payload.getMutable(cursor + j) = payload.getElement(cell_start + j);

			m_start[i] = PayloadId(cursor);
			cursor += cell_count;
		}
	}
};


struct __attribute__ ((aligned(64))) ChildIndex
{
	uint32_t index[8];
	float distance[8];
};

static const compile_assert< 64 == sizeof(ChildIndex) > assert_child_index_size;

#if __SSE4_1__ == 0
inline __m128 __attribute__ ((always_inline))
_nn_blend_ps(
	const __m128 a,
	const __m128 b,
	const int8_t mask)
{
	assert(mask >= 0);
	assert(16 > mask);

	const int32_t u = -1;
	const __m128i vmsk[] =
	{
		_mm_set_epi32(0, 0, 0, 0),
		_mm_set_epi32(0, 0, 0, u),
		_mm_set_epi32(0, 0, u, 0),
		_mm_set_epi32(0, 0, u, u),
		_mm_set_epi32(0, u, 0, 0),
		_mm_set_epi32(0, u, 0, u),
		_mm_set_epi32(0, u, u, 0),
		_mm_set_epi32(0, u, u, u),
		_mm_set_epi32(u, 0, 0, 0),
		_mm_set_epi32(u, 0, 0, u),
		_mm_set_epi32(u, 0, u, 0),
		_mm_set_epi32(u, 0, u, u),
		_mm_set_epi32(u, u, 0, 0),
		_mm_set_epi32(u, u, 0, u),
		_mm_set_epi32(u, u, u, 0),
		_mm_set_epi32(u, u, u, u)
	};
	const __m128 vmask = _mm_castsi128_ps(vmsk[mask]);

	return _mm_or_ps(_mm_andnot_ps(vmask, a), _mm_and_ps(vmask, b));
}

inline __m128 __attribute__ ((always_inline))
_nn_blendv_ps(
	const __m128 a,
	const __m128 b,
	const __m128 mask)
{
	return _mm_or_ps(_mm_andnot_ps(mask, a), _mm_and_ps(mask, b));
}

#else // __SSE4_1__
#define _nn_blend_ps _mm_blend_ps
#define _nn_blendv_ps _mm_blendv_ps

#endif // __SSE4_1__

struct HitInfo
{
	__m128 min_mask;
	int a_mask;
	int b_mask;

	float dist;
	PayloadId target;
};

//
// A sparse regular octree
//

class Timeslice
{
	BBox             m_root_bbox;
	Octet            m_root;
	Array< Octet >   m_interior;
	Array< Leaf >    m_leaf;
	Array< Voxel >   m_payload;

	// following data members are thread-local and valid only for the duration of a traversal
	static __thread const Ray* m_ray;
	static __thread HitInfo* m_hit;

	template < unsigned OCTREE_LEVEL_T >
	bool
	traverse(
		const Octet& octet,
		const BBox& bbox) const;

	template < unsigned OCTREE_LEVEL_T >
	bool
	traverse_lite(
		const Octet& octet,
		const BBox& bbox) const;

	template < unsigned OCTREE_LEVEL_T >
	bool
	traverse_litest(
		const Octet& octet,
		const BBox& bbox) const;

	bool
	traverse(
		const Leaf& leaf,
		const BBox& bbox) const;

	bool
	traverse_lite(
		const Leaf& leaf,
		const BBox& bbox) const;

	bool
	traverse_litest(
		const Leaf& leaf,
		const BBox& bbox) const;

	template < unsigned OCTREE_LEVEL_T >
	bool
	add_payload(
		Octet& octet,
		const BBox& bbox,
		const Voxel& payload);

	bool
	add_payload(
		Leaf& leaf,
		const BBox& bbox,
		const Voxel& payload);

public:
	Timeslice()
	{
	}

	bool
	set_payload_array(
		const Array< Voxel >& arr);

	const BBox&
	get_root_bbox() const
	{
		return m_root_bbox;
	}

#if CLANG_QUIRK_0001 != 0
	// we want the following methods always inlined, yet for some reason keeping their definitions in the translation
	// unit, tagging them 'always_inline' here, and leaving the inlining to the LTO yields faster code; file a report?

	bool __attribute__ ((always_inline))
	traverse(
		const Ray& ray,
		HitInfo& hit) const;

	bool __attribute__ ((always_inline))
	traverse_lite(
		const Ray& ray,
		HitInfo& hit) const;

	bool __attribute__ ((always_inline))
	traverse_litest(
		const Ray& ray,
		HitInfo& hit) const;

#else
	bool
	traverse(
		const Ray& ray,
		HitInfo& hit) const;

	bool
	traverse_lite(
		const Ray& ray,
		HitInfo& hit) const;

	bool
	traverse_litest(
		const Ray& ray,
		HitInfo& hit) const;

#endif // CLANG_QUIRK_0001
};

#include "octet_intersect_wide.hpp"
#include "octlf_intersect_wide.hpp"

template < unsigned OCTREE_LEVEL_T >
inline bool
Timeslice::traverse(
	const Octet& octet,
	const BBox& bbox) const
{
	assert(bbox.is_valid());
	assert(0 != m_ray);

	const Ray& ray = *m_ray;

	ChildIndex child_index;
	BBox child_bbox[8] __attribute__ ((aligned(64))) =
	{
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit())
	};

	const size_t hit_count = octet_intersect_wide(
		octet,
		bbox,
		ray,
		child_index,
		child_bbox);

	for (size_t i = 0; i < hit_count; ++i)
	{
		const size_t index = child_index.index[i];
		const OctetId child_id = octet.get(index);

		if (traverse< OCTREE_LEVEL_T + 1 >(
				m_interior.getElement(child_id),
				child_bbox[index]))
		{
			return true;
		}
	}

	return false;
}


template <>
inline bool
Timeslice::traverse< octree_level_last_but_one >(
	const Octet& octet,
	const BBox& bbox) const
{
	assert(bbox.is_valid());
	assert(0 != m_ray);

	const Ray& ray = *m_ray;

	ChildIndex child_index;
	BBox child_bbox[8] __attribute__ ((aligned(64))) =
	{
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit())
	};

	const size_t hit_count = octet_intersect_wide(
		octet,
		bbox,
		ray,
		child_index,
		child_bbox);

	for (size_t i = 0; i < hit_count; ++i)
	{
		const size_t index = child_index.index[i];
		const OctetId child_id = octet.get(index);

		if (traverse(
				m_leaf.getElement(child_id),
				child_bbox[index]))
		{
			return true;
		}
	}

	return false;
}


inline bool
Timeslice::traverse(
	const Leaf& leaf,
	const BBox& bbox) const
{
	assert(bbox.is_valid());
	assert(0 != m_ray);
	assert(0 != m_hit);

	const Ray& ray = *m_ray;

	ChildIndex child_index;

	const size_t hit_count = octet_intersect_wide(
		leaf,
		bbox,
		ray,
		child_index);

	const PayloadId prior_target = m_hit->target;

#if DRAW_TREE_CELLS == 1
	if (0 != hit_count)
	{
		m_hit->target = child_index.index[0];
		return true;
	}

#endif
	for (size_t i = 0; i < hit_count; ++i)
	{
		const size_t payload_start = leaf.get_start(child_index.index[i]);
		const size_t payload_count = leaf.get_count(child_index.index[i]);

		assert(0 != payload_count);

		float nearest_dist = child_index.distance[i];

		for (size_t j = payload_start; j < payload_start + payload_count; ++j)
		{
			const Voxel& voxel = m_payload.getElement(j);
			const PayloadId id = PayloadId(voxel.get_id());

			if (id == prior_target)
				continue;

			__m128 min_mask;
			int a_mask;
			int b_mask;
			float dist;

			if (voxel.get_bbox().intersect(ray, min_mask, a_mask, b_mask, dist) &&
				dist < nearest_dist)
			{
				nearest_dist = dist;

				m_hit->min_mask = min_mask;
				m_hit->a_mask = a_mask;
				m_hit->b_mask = b_mask;
				m_hit->dist = dist;
				m_hit->target = id;
			}
		}

		if (m_hit->target != prior_target)
			return true;
	}

	return false;
}


template < unsigned OCTREE_LEVEL_T >
inline bool
Timeslice::traverse_lite(
	const Octet& octet,
	const BBox& bbox) const
{
	assert(bbox.is_valid());
	assert(0 != m_ray);

	const Ray& ray = *m_ray;

	ChildIndex child_index;
	BBox child_bbox[8] __attribute__ ((aligned(64))) =
	{
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit())
	};

	const size_t hit_count = octet_intersect_wide(
		octet,
		bbox,
		ray,
		child_index,
		child_bbox);

	for (size_t i = 0; i < hit_count; ++i)
	{
		const size_t index = child_index.index[i];
		const OctetId child_id = octet.get(index);

		if (traverse_lite< OCTREE_LEVEL_T + 1 >(
				m_interior.getElement(child_id),
				child_bbox[index]))
		{
			return true;
		}
	}

	return false;
}


template <>
inline bool
Timeslice::traverse_lite< octree_level_last_but_one >(
	const Octet& octet,
	const BBox& bbox) const
{
	assert(bbox.is_valid());
	assert(0 != m_ray);

	const Ray& ray = *m_ray;

	ChildIndex child_index;
	BBox child_bbox[8] __attribute__ ((aligned(64))) =
	{
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit())
	};

	const size_t hit_count = octet_intersect_wide(
		octet,
		bbox,
		ray,
		child_index,
		child_bbox);

	for (size_t i = 0; i < hit_count; ++i)
	{
		const size_t index = child_index.index[i];
		const OctetId child_id = octet.get(index);

		if (traverse_lite(
				m_leaf.getElement(child_id),
				child_bbox[index]))
		{
			return true;
		}
	}

	return false;
}


inline bool
Timeslice::traverse_lite(
	const Leaf& leaf,
	const BBox& bbox) const
{
	assert(bbox.is_valid());
	assert(0 != m_ray);
	assert(0 != m_hit);

	const Ray& ray = *m_ray;

	ChildIndex child_index;

	const size_t hit_count = octet_intersect_wide(
		leaf,
		bbox,
		ray,
		child_index);

	const PayloadId prior_target = m_hit->target;

	for (size_t i = 0; i < hit_count; ++i)
	{
		const size_t payload_start = leaf.get_start(child_index.index[i]);
		const size_t payload_count = leaf.get_count(child_index.index[i]);

		assert(0 != payload_count);

		float nearest_dist = child_index.distance[i];

		for (size_t j = payload_start; j < payload_start + payload_count; ++j)
		{
			const Voxel& voxel = m_payload.getElement(j);
			const PayloadId id = PayloadId(voxel.get_id());

			if (id == prior_target)
				continue;

			float dist[2];

			if (voxel.get_bbox().intersect(ray, dist) &&
				dist[0] < nearest_dist)
			{
				nearest_dist = dist[0];

				m_hit->target = id;
				m_hit->dist = dist[0];
			}
		}

		if (m_hit->target != prior_target)
			return true;
	}

	return false;
}


template < unsigned OCTREE_LEVEL_T >
inline bool
Timeslice::traverse_litest(
	const Octet& octet,
	const BBox& bbox) const
{
	assert(bbox.is_valid());
	assert(0 != m_ray);

	const Ray& ray = *m_ray;

	ChildIndex child_index;
	BBox child_bbox[8] __attribute__ ((aligned(64))) =
	{
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit())
	};

	const size_t hit_count = octet_intersect_wide(
		octet,
		bbox,
		ray,
		child_index,
		child_bbox);

	for (size_t i = 0; i < hit_count; ++i)
	{
		const size_t index = child_index.index[i];
		const OctetId child_id = octet.get(index);

		if (traverse_litest< OCTREE_LEVEL_T + 1 >(
				m_interior.getElement(child_id),
				child_bbox[index]))
		{
			return true;
		}
	}

	return false;
}


template <>
inline bool
Timeslice::traverse_litest< octree_level_last_but_one >(
	const Octet& octet,
	const BBox& bbox) const
{
	assert(bbox.is_valid());
	assert(0 != m_ray);

	const Ray& ray = *m_ray;

	ChildIndex child_index;
	BBox child_bbox[8] __attribute__ ((aligned(64))) =
	{
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit()),
		BBox(BBox::flag_noinit())
	};

	const size_t hit_count = octet_intersect_wide(
		octet,
		bbox,
		ray,
		child_index,
		child_bbox);

	for (size_t i = 0; i < hit_count; ++i)
	{
		const size_t index = child_index.index[i];
		const OctetId child_id = octet.get(index);

		if (traverse_litest(
				m_leaf.getElement(child_id),
				child_bbox[index]))
		{
			return true;
		}
	}

	return false;
}


inline bool
Timeslice::traverse_litest(
	const Leaf& leaf,
	const BBox& bbox) const
{
	assert(bbox.is_valid());
	assert(0 != m_ray);
	assert(0 != m_hit);

	const Ray& ray = *m_ray;

	ChildIndex child_index;

	const size_t hit_count = octet_intersect_wide(
		leaf,
		bbox,
		ray,
		child_index);

	const PayloadId prior_target = m_hit->target;

	for (size_t i = 0; i < hit_count; ++i)
	{
		const size_t payload_start = leaf.get_start(child_index.index[i]);
		const size_t payload_count = leaf.get_count(child_index.index[i]);

		assert(0 != payload_count);

		const size_t unroll_by_2 = payload_count & size_t(-2);

		for (size_t j = payload_start; j < payload_start + unroll_by_2; j += 2)
		{
			const Voxel& voxel0 = m_payload.getElement(j + 0);
			const Voxel& voxel1 = m_payload.getElement(j + 1);
			const PayloadId id0 = PayloadId(voxel0.get_id());
			const PayloadId id1 = PayloadId(voxel1.get_id());

			unsigned r[2];
			intersect2(&voxel0.get_bbox(), ray, r);

			if (id0 != prior_target & r[0] |
				id1 != prior_target & r[1])
			{
				return true;
			}
		}

		if (0 == (payload_count & 1))
			continue;

		const size_t j = payload_start + unroll_by_2;
		const Voxel& voxel = m_payload.getElement(j);
		const PayloadId id = PayloadId(voxel.get_id());

		float dist[2];

		if (id != prior_target && voxel.get_bbox().intersect(ray, dist))
			return true;
	}

	return false;
}


#if CLANG_QUIRK_0001 == 0
inline bool __attribute__ ((always_inline))
Timeslice::traverse(
	const Ray& ray,
	HitInfo& hit) const
{
	assert(m_root_bbox.is_valid());

	float dummy[2];

	if (!m_root_bbox.intersect(ray, dummy))
		return false;

	m_ray = &ray;
	m_hit = &hit;

	return traverse< octree_level_root >(
		m_root,
		m_root_bbox);
}


inline bool __attribute__ ((always_inline))
Timeslice::traverse_lite(
	const Ray& ray,
	HitInfo& hit) const
{
	assert(m_root_bbox.is_valid());

	float dummy[2];

	if (!m_root_bbox.intersect(ray, dummy))
		return false;

	m_ray = &ray;
	m_hit = &hit;

	return traverse_lite< octree_level_root >(
		m_root,
		m_root_bbox);
}


inline bool __attribute__ ((always_inline))
Timeslice::traverse_litest(
	const Ray& ray,
	HitInfo& hit) const
{
	assert(m_root_bbox.is_valid());

	float dummy[2];

	if (!m_root_bbox.intersect(ray, dummy))
		return false;

	m_ray = &ray;
	m_hit = &hit;

	return traverse_litest< octree_level_root >(
		m_root,
		m_root_bbox);
}

#endif // CLANG_QUIRK_0001

#endif // prob_4_H__
