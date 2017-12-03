#ifndef prob_6_H__
#define prob_6_H__

#include <assert.h>
#include <stdint.h>
#include <istream>
#include <ostream>
#include <limits>
#include "vectnative.hpp"
#include "array.hpp"
#include "array_extern.hpp"
#include "scoped.hpp"

template < bool >
struct compile_assert;

template <>
struct compile_assert< true > {
	compile_assert() {}
};

class vect3 : public simd::f32x4 {
public:
	// indices for subscript op
	enum axis {
		axis_x,
		axis_y,
		axis_z
	};

	enum flag_zero {};
	vect3() {}

	vect3(
		const float x,
		const float y,
		const float z)
	: simd::f32x4(x, y, z) {
	}

	vect3(
		const float x,
		const float y,
		const float z,
		const flag_zero)
	: simd::f32x4(x, y, z, simd::flag_zero()) {
	}

	explicit vect3(const float c)
	: simd::f32x4(c) {
	}

	explicit vect3(const float (& arr)[3])
	: simd::f32x4(arr[0], arr[1], arr[2]) {
	}

	vect3(const float (& arr)[3], const flag_zero)
	: simd::f32x4(arr[0], arr[1], arr[2], simd::flag_zero()) {
	}

	vect3(const simd::f32x4 c)
	: simd::f32x4(c) {
	}
};

class vect4 : public simd::f32x4 {
public:
	// indices for subscript op
	enum axis {
		axis_x,
		axis_y,
		axis_z,
		axis_w
	};

	vect4() {}

	vect4(
		const float x,
		const float y,
		const float z,
		const float w)
	: simd::f32x4(x, y, z, w) {
	}

	explicit vect4(const float c)
	: simd::f32x4(c) {
	}

	explicit vect4(const float (& arr)[4])
	: simd::f32x4(arr[0], arr[1], arr[2], arr[3]) {
	}

	vect4(const simd::f32x4 c)
	: simd::f32x4(c) {
	}
};

class matx3 {
protected:
	simd::f32x4 m[3];

public:
	matx3() {}
	matx3(
		const float c00, const float c01, const float c02,
		const float c10, const float c11, const float c12,
		const float c20, const float c21, const float c22) {

		m[0] = simd::f32x4(c00, c01, c02, simd::flag_zero());
		m[1] = simd::f32x4(c10, c11, c12, simd::flag_zero());
		m[2] = simd::f32x4(c20, c21, c22, simd::flag_zero());
	}

	matx3(
		const simd::f32x4 row0,
		const simd::f32x4 row1,
		const simd::f32x4 row2) {
		m[0] = row0;
		m[1] = row1;
		m[2] = row2;
	}

	vect3 get(const size_t rowIdx) const {
		assert(3 > rowIdx);
		return m[rowIdx];
	}

	void set(const size_t rowIdx, const vect3& row) {
		assert(3 > rowIdx);
		m[rowIdx] = row;
	}

	vect3 operator[](const size_t rowIdx) const {
		return get(rowIdx);
	}
};

class matx4 {
protected:
	simd::f32x4 m[4];

public:
	matx4() {}
	matx4(
		const float c00, const float c01, const float c02, const float c03,
		const float c10, const float c11, const float c12, const float c13,
		const float c20, const float c21, const float c22, const float c23,
		const float c30, const float c31, const float c32, const float c33) {

		m[0] = simd::f32x4(c00, c01, c02, c03);
		m[1] = simd::f32x4(c10, c11, c12, c13);
		m[2] = simd::f32x4(c20, c21, c22, c23);
		m[3] = simd::f32x4(c30, c31, c32, c33);
	}

	matx4(
		const simd::f32x4 row0,
		const simd::f32x4 row1,
		const simd::f32x4 row2,
		const simd::f32x4 row3) {
		m[0] = row0;
		m[1] = row1;
		m[2] = row2;
		m[3] = row3;
	}

	vect4 get(const size_t rowIdx) const {
		assert(4 > rowIdx);
		return m[rowIdx];
	}

	void set(const size_t rowIdx, const vect4& row) {
		assert(4 > rowIdx);
		m[rowIdx] = row;
	}

	vect4 operator[](const size_t rowIdx) const {
		return get(rowIdx);
	}
};

inline vect3 operator *(
	const vect3& v,
	const matx3& m) {

	using simd::f32x4;
	return vect3(
		f32x4(v[0]) * m[0] +
		f32x4(v[1]) * m[1] +
		f32x4(v[2]) * m[2]);
}

inline vect3 operator *(
	const vect3& v,
	const matx4& m) {

	using simd::f32x4;
	return vect3(
		f32x4(v[0]) * m[0] +
		f32x4(v[1]) * m[1] +
		f32x4(v[2]) * m[2] + m[3]);
}

inline matx3 operator *(
	const matx3& a,
	const matx3& b) {

	using simd::f32x4;
	const f32x4 r0 =
		f32x4(a[0][0]) * b[0] +
		f32x4(a[0][1]) * b[1] +
		f32x4(a[0][2]) * b[2];

	const f32x4 r1 =
		f32x4(a[1][0]) * b[0] +
		f32x4(a[1][1]) * b[1] +
		f32x4(a[1][2]) * b[2];

	const f32x4 r2 =
		f32x4(a[2][0]) * b[0] +
		f32x4(a[2][1]) * b[1] +
		f32x4(a[2][2]) * b[2];

	return matx3(r0, r1, r2);
}

inline matx4 operator *(
	const matx4& a,
	const matx4& b) {

	using simd::f32x4;
	const f32x4 r0 =
		f32x4(a[0][0]) * b[0] +
		f32x4(a[0][1]) * b[1] +
		f32x4(a[0][2]) * b[2] +
		f32x4(a[0][3]) * b[3];

	const f32x4 r1 =
		f32x4(a[1][0]) * b[0] +
		f32x4(a[1][1]) * b[1] +
		f32x4(a[1][2]) * b[2] +
		f32x4(a[1][3]) * b[3];

	const f32x4 r2 =
		f32x4(a[2][0]) * b[0] +
		f32x4(a[2][1]) * b[1] +
		f32x4(a[2][2]) * b[2] +
		f32x4(a[2][3]) * b[3];

	const f32x4 r3 =
		f32x4(a[3][0]) * b[0] +
		f32x4(a[3][1]) * b[1] +
		f32x4(a[3][2]) * b[2] +
		f32x4(a[3][3]) * b[3];

	return matx4(r0, r1, r2, r3);
}

class BBox {
	vect3 m_min;
	vect3 m_max;

public:
	BBox(
		const BBox& oth)
	: m_min(oth.m_min)
	, m_max(oth.m_max) {
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
	: m_min( std::numeric_limits< float >::infinity())
	, m_max(-std::numeric_limits< float >::infinity()) {
	}

	BBox(
		const flag_noinit) {
	}

	enum { axis_mask_all = 7 };

	bool
	is_valid() const {
		return simd::all(simd::u32x4(m_min <= m_max), axis_mask_all);
	}

	BBox(
		const vect3& min,
		const vect3& max,
		const flag_direct)
	: m_min(min)
	, m_max(max) {
		assert(is_valid());
	}

	BBox(
		const vect3& v0,
		const vect3& v1)
	: m_min(simd::min(v0, v1))
	, m_max(simd::max(v0, v1)) {
		assert(is_valid());
	}

	const vect3&
	get_min() const {
		return m_min;
	}

	const vect3&
	get_max() const {
		return m_max;
	}

	BBox&
	grow(
		const vect3& v) {

		const BBox min = BBox(m_min, v);
		const BBox max = BBox(m_max, v);

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

		m_min = simd::max(m_min, oth.m_min);
		m_max = simd::min(m_max, oth.m_max);

		assert(is_valid());
		return *this;
	}

	// warning: use this mutator only after the box is done with grow/overlap!
	void set_min_cookie(
		const uint32_t cookie) {

		m_min.set(3, reinterpret_cast< const float& >(cookie));
	}

	// warning: use this mutator only after the box is done with grow/overlap!
	void set_max_cookie(
		const uint32_t cookie) {

		m_max.set(3, reinterpret_cast< const float& >(cookie));
	}

	uint32_t get_min_cookie() const {
		const float cookie = m_min[3];
		return reinterpret_cast< const uint32_t& >(cookie);
	}

	uint32_t get_max_cookie() const {
		const float cookie = m_max[3];
		return reinterpret_cast< const uint32_t& >(cookie);
	}

	bool
	has_overlap_open(
		const BBox& oth) const;

	bool
	has_overlap_closed(
		const BBox& oth) const;

	bool
	contains_open(
		const vect3& v) const;

	bool
	contains_closed(
		const vect3& v) const;

	bool
	contains_open(
		const BBox& oth) const;

	bool
	contains_closed(
		const BBox& oth) const;

	template < uint32_t AXIS_MASK >
	bool
	has_overlap_open_subdim(
		const BBox& oth) const;

	template < uint32_t AXIS_MASK >
	bool
	has_overlap_closed_subdim(
		const BBox& oth) const;

	template < uint32_t AXIS_MASK >
	bool
	contains_open_subdim(
		const vect3& v) const;

	template < uint32_t AXIS_MASK >
	bool
	contains_closed_subdim(
		const vect3& v) const;

	template < uint32_t AXIS_MASK >
	bool
	contains_open_subdim(
		const BBox& oth) const;

	template < uint32_t AXIS_MASK >
	bool
	contains_closed_subdim(
		const BBox& oth) const;
};


template < uint32_t AXIS_MASK >
inline bool
BBox::has_overlap_open_subdim(
	const BBox& oth) const {

	if (any(m_min >= oth.m_max, AXIS_MASK))
		return false;

	if (any(m_max <= oth.m_min, AXIS_MASK))
		return false;

	return true;
}


template < uint32_t AXIS_MASK >
inline bool
BBox::has_overlap_closed_subdim(
	const BBox& oth) const {

	if (any(m_min > oth.m_max, AXIS_MASK))
		return false;

	if (any(m_max < oth.m_min, AXIS_MASK))
		return false;

	return true;
}


template < uint32_t AXIS_MASK >
inline bool
BBox::contains_open_subdim(
	const vect3& v) const {

	if (any(m_min >= v, AXIS_MASK))
		return false;

	if (any(m_max <= v, AXIS_MASK))
		return false;

	return true;
}


template < uint32_t AXIS_MASK >
inline bool
BBox::contains_closed_subdim(
	const vect3& v) const {

	if (any(m_min > v, AXIS_MASK))
		return false;

	if (any(m_max < v, AXIS_MASK))
		return false;

	return true;
}


template < uint32_t AXIS_MASK >
inline bool
BBox::contains_open_subdim(
	const BBox& oth) const {

	if (any(m_min >= oth.m_min, AXIS_MASK))
		return false;

	if (any(m_max <= oth.m_max, AXIS_MASK))
		return false;

	return true;
}


template < uint32_t AXIS_MASK >
inline bool
BBox::contains_closed_subdim(
	const BBox& oth) const {

	if (any(m_min > oth.m_min, AXIS_MASK))
		return false;

	if (any(m_max < oth.m_max, AXIS_MASK))
		return false;

	return true;
}


inline bool
BBox::has_overlap_open(
	const BBox& oth) const {

	return has_overlap_open_subdim< axis_mask_all >(oth);
}


inline bool
BBox::has_overlap_closed(
	const BBox& oth) const {

	return has_overlap_closed_subdim< axis_mask_all >(oth);
}


inline bool
BBox::contains_open(
	const vect3& v) const {

	return contains_open_subdim< axis_mask_all >(v);
}


inline bool
BBox::contains_closed(
	const vect3& v) const {

	return contains_closed_subdim< axis_mask_all >(v);
}


inline bool
BBox::contains_open(
	const BBox& oth) const {

	return contains_open_subdim< axis_mask_all >(oth);
}


inline bool
BBox::contains_closed(
	const BBox& oth) const {

	return contains_closed_subdim< axis_mask_all >(oth);
}

class Voxel {
	BBox m_bbox;

public:
	Voxel()
	: m_bbox(BBox::flag_noinit()) {
	}

	Voxel(
		const vect3& min,
		const vect3& max)
	: m_bbox(min, max, BBox::flag_direct()) {
	}

	enum flag_unordered {};

	Voxel(
		const vect3& p0,
		const vect3& p1,
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

	const vect3&
	get_min() const {
		return m_bbox.get_min();
	}

	const vect3&
	get_max() const {
		return m_bbox.get_max();
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

enum { octet_empty = -1 };
enum { cell_capacity = 32 }; // octree cell capacity during building
enum { octree_payload_count = octree_cell_count * cell_capacity };

static const compile_assert< (size_t(1) << sizeof(OctetId) * 8 > octree_leaf_count) > assert_octet_id;
static const compile_assert< (size_t(1) << sizeof(PayloadId) * 8 > octree_payload_count) > assert_payload_id;


class __attribute__ ((aligned(16))) Octet {
	enum { capacity = 8 };

	OctetId m_child[capacity];

public:
	Octet() {
		for (size_t i = 0; i < capacity; ++i)
			m_child[i] = OctetId(octet_empty);
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
		return OctetId(octet_empty) == m_child[index];
	}

	simd::u16x8
	get_occupancy() const {
		const compile_assert< sizeof(*this) == sizeof(simd::u16x8) > assert_octet_size;
		return simd::u16x8(OctetId(octet_empty)) == reinterpret_cast< const simd::u16x8* >(this)[0];
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

	simd::u16x8
	get_occupancy() const {
		const compile_assert< sizeof(*this) == sizeof(simd::u16x8) * 2 > assert_leaf_size;
		return simd::u16x8(0) == reinterpret_cast< const simd::u16x8* >(this)[1];
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
		const Array< Voxel >& arr,
		const BBox& root_bbox);

	const BBox&
	get_root_bbox() const {
		return m_root_bbox;
	}
};

#endif // prob_6_H__
