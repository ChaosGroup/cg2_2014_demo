#ifndef prob_6_H__
#define prob_6_H__

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
#include "array_extern.hpp"
#include "scoped.hpp"

template < bool >
struct compile_assert;

template <>
struct compile_assert< true > {
	compile_assert() {
	}
};


class BBox {
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
		const BBox& oth) {

		m_min = oth.m_min;
		m_max = oth.m_max;
	}

	BBox& operator =(
		const BBox& oth) {

		m_min = oth.m_min;
		m_max = oth.m_max;
		return *this;
	}

	enum flag_direct {};
	enum flag_noinit {};

	BBox()
	: m_min(_mm_set1_ps( std::numeric_limits< float >::infinity()))
	, m_max(_mm_set1_ps(-std::numeric_limits< float >::infinity())) {
	}

	BBox(
		const flag_noinit) {
	}

	bool
	is_valid() const {
		return 7 == (7 & _mm_movemask_ps(_mm_cmple_ps(m_min, m_max)));
	}

	BBox(
		const __m128 min,
		const __m128 max,
		const flag_direct)
	: m_min(min)
	, m_max(max) {
		assert(is_valid());
	}

	BBox(
		const simd::vect3& min,
		const simd::vect3& max,
		const flag_direct) {

		*this = BBox(min.getn(), max.getn(), flag_direct());
	}

	BBox(
		const __m128 v0,
		const __m128 v1) {

		m_min = _mm_min_ps(v0, v1);
		m_max = _mm_max_ps(v0, v1);

		assert(is_valid());
	}

	BBox(
		const simd::vect3& v0,
		const simd::vect3& v1) {

		*this = BBox(v0.getn(), v1.getn());
	}

	const __m128
	get_min() const {
		return m_min;
	}

	const __m128
	get_max() const {
		return m_max;
	}

	BBox&
	grow(
		const simd::vect3& v) {

		const BBox min = BBox(m_min, v.getn());
		const BBox max = BBox(m_max, v.getn());

		m_min = min.get_min();
		m_max = max.get_max();

		return *this;
	}

	BBox&
	grow(
		const BBox& oth) {

		const BBox min = BBox(m_min, oth.m_min);
		const BBox max = BBox(m_max, oth.m_max);

		m_min = min.get_min();
		m_max = max.get_max();

		return *this;
	}

	BBox&
	overlap(
		const BBox& oth) {

		m_min = _mm_max_ps(m_min, oth.m_min);
		m_max = _mm_min_ps(m_max, oth.m_max);

		assert(is_valid());

		return *this;
	}

	// warning: use this mutator only after the box is done with grow/overlap!
	void set_min_cookie(
		const uint32_t cookie) {

		umin[3] = cookie;
	}

	// warning: use this mutator only after the box is done with grow/overlap!
	void set_max_cookie(
		const uint32_t cookie) {

		umax[3] = cookie;
	}

	uint32_t get_min_cookie() const {
		return umin[3];
	}

	uint32_t get_max_cookie() const {
		return umax[3];
	}

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
	const BBox& oth) const {

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
	const BBox& oth) const {

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
	const simd::vect3& v) const {

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
	const simd::vect3& v) const {

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
	const BBox& oth) const {

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
	const BBox& oth) const {

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
	const BBox& oth) const {

	return has_overlap_open_subdim< 3 >(oth);
}


inline bool
BBox::has_overlap_closed(
	const BBox& oth) const {

	return has_overlap_closed_subdim< 3 >(oth);
}


inline bool
BBox::contains_open(
	const simd::vect3& v) const {

	return contains_open_subdim< 3 >(v);
}


inline bool
BBox::contains_closed(
	const simd::vect3& v) const {

	return contains_closed_subdim< 3 >(v);
}


inline bool
BBox::contains_open(
	const BBox& oth) const {

	return contains_open_subdim< 3 >(oth);
}


inline bool
BBox::contains_closed(
	const BBox& oth) const {

	return contains_closed_subdim< 3 >(oth);
}

class Voxel {
	BBox m_bbox;

public:
	Voxel()
	: m_bbox(BBox::flag_noinit()) {
	}

	Voxel(
		const simd::vect3& min,
		const simd::vect3& max)
	: m_bbox(min, max, BBox::flag_direct()) {
	}

	enum flag_unordered {};

	Voxel(
		const simd::vect3& p0,
		const simd::vect3& p1,
		const flag_unordered)
	: m_bbox(p0, p1) {
	}

	Voxel(
		const BBox& bbox,
		const uint32_t id)
	: m_bbox(bbox) {
		m_bbox.set_min_cookie(id);
	}

	const BBox&
	get_bbox() const {
		return m_bbox;
	}

	const simd::vect3
	get_min() const {
		simd::vect3 r;
		r.setn(0, m_bbox.get_min());
		return r;
	}

	const simd::vect3
	get_max() const {
		simd::vect3 r;
		r.setn(0, m_bbox.get_max());
		return r;
	}

	uint32_t get_id() const {
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
	const unsigned z) {

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
	unsigned& z) {

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
	unsigned& local_z) {

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
	cell_capacity = 32 // octree cell capacity during building
};

enum {
	octree_payload_count = octree_cell_count * cell_capacity
};

static const compile_assert< (size_t(1) << sizeof(OctetId) * 8 > octree_leaf_count) > assert_octet_id;
static const compile_assert< (size_t(1) << sizeof(PayloadId) * 8 > octree_payload_count) > assert_payload_id;


class __attribute__ ((aligned(16))) Octet {
	enum { capacity = 8 };

	OctetId m_child[capacity];

public:
	Octet() {
		for (size_t i = 0; i < capacity; ++i)
			m_child[i] = OctetId(-1);
	}

	void
	set(
		const size_t index,
		const OctetId child) {

		assert(capacity > index);
		m_child[index] = child;
	}

	OctetId
	get(
		const size_t index) const {

		assert(capacity > index);
		return m_child[index];
	}

	bool
	empty(
		const size_t index) const {

		assert(capacity > index);
		return OctetId(-1) == m_child[index];
	}

	__m128i
	get_occupancy() const {
		const compile_assert< sizeof(*this) == sizeof(__m128i) > assert_octet_size;
		return _mm_cmpeq_epi16(_mm_set1_epi16(-1), reinterpret_cast< const __m128i* >(this)[0]);
	}
};


class __attribute__ ((aligned(16))) Leaf {
	enum { capacity = 8 };

	PayloadId m_start[capacity];
	PayloadId m_count[capacity];

public:
	void init(
		const PayloadId start) {

		for (size_t i = 0; i < capacity; ++i) {
			m_start[i] = start + PayloadId(i * cell_capacity);
			m_count[i] = 0;
		}
	}

	PayloadId
	get_start(
		const size_t index) const {

		assert(capacity > index);
		return m_start[index];
	}

	PayloadId
	get_count(
		const size_t index) const {

		assert(capacity > index);
		return m_count[index];
	}

	bool
	add(
		const size_t index,
		ArrayExtern< Voxel >& payload,
		const Voxel& item) {

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
		const size_t index) const {

		assert(capacity > index);
		return 0 == m_count[index];
	}

	__m128i
	get_occupancy() const {
		const compile_assert< sizeof(*this) == sizeof(__m128i) * 2 > assert_leaf_size;
		return _mm_cmpeq_epi16(_mm_set1_epi16(0), reinterpret_cast< const __m128i* >(this)[1]);
	}

	void
	compact(
		size_t& cursor,
		ArrayExtern< Voxel >& payload) {

		for (size_t i = 0; i < capacity; ++i) {
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

//
// A sparse regular octree - external pointer version
//

class Timeslice {
	BBox m_root_bbox;
	ArrayExtern< Octet > m_interior;
	ArrayExtern< Leaf >  m_leaf;
	ArrayExtern< Voxel > m_payload;

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
	Timeslice() {
	}

	void
	set_extrnal_storage(
		const size_t interiorCapacity,
		void* const interior,
		const size_t leafCapacity,
		void* const leaf,
		const size_t payloadCapacity,
		void* const payload);

	bool
	set_payload_array(
		const Array< Voxel >& arr);

	const BBox&
	get_root_bbox() const {
		return m_root_bbox;
	}
};

#endif // prob_6_H__
