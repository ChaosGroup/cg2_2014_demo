#include <istream>
#include <ostream>
#include <limits>
#include "vectsimd_sse.hpp"
#include "array.hpp"
#include "isfinite.hpp"
#include "stream.hpp"
#include "problem_6.hpp"

// verify iostream-free status
#if _GLIBCXX_IOSTREAM
#error rogue iostream acquired
#endif

template < size_t DIMENSION_T, typename NATIVE_T >
inline std::ostream&
operator <<(
	std::ostream& str,
	const simd::vect< DIMENSION_T, NATIVE_T >& v)
{
	str << '(';

	size_t i = 0;

	for (; i < DIMENSION_T - 1 && str.good(); ++i)
		str << v.get(i) << ", ";

	str << v.get(i) << ')';

	return str;
}


template < size_t DIMENSION_T, typename NATIVE_T >
inline std::istream&
operator >>(
	std::istream& str,
	simd::vect< DIMENSION_T, NATIVE_T >& v)
{
	float temp[DIMENSION_T];

	for (size_t i = 0; i < DIMENSION_T; ++i)
		temp[i] = std::numeric_limits< float >::quiet_NaN();

	char sep;
	str >> sep;

	if ('(' != sep)
		str.setstate(std::istream::failbit);

	size_t i = 0;

	for (; i < DIMENSION_T - 1 && str.good(); ++i)
	{
		str >> temp[i];
		str >> sep;

		if (',' != sep)
			str.setstate(std::istream::failbit);
	}

	str >> temp[i];
	str >> sep;

	if (')' != sep)
		str.setstate(std::istream::failbit);

	v = simd::vect< DIMENSION_T, NATIVE_T >(temp);

	return str;
}


bool
Voxel::flatten(
	std::ostream& str) const
{
	if (!str.good())
		return false;

	str << get_min();
	str << get_max();

	return str.good();
}

bool
Voxel::deflatten(
	std::istream& str)
{
	if (!str.good())
		return false;

	simd::vect3 temp[2];
	str >> temp[0];

	if (!str.good())
		return false;

	for (size_t i = 0; i < temp[0].dimension; ++i)
		if (!isfinite(temp[0].get(i)))
			return false;

	str >> temp[1];

	if (!str.good())
		return false;

	for (size_t i = 0; i < temp[1].dimension; ++i)
		if (!isfinite(temp[1].get(i)))
			return false;

	*this = Voxel(temp[0], temp[1], flag_unordered());

	return true;
}

template < unsigned OCTREE_LEVEL_T >
bool
Timeslice::add_payload(
	Octet& octet,
	const BBox& bbox,
	const Voxel& payload) {

	const __m128 bbox_min = bbox.get_min();
	const __m128 bbox_max = bbox.get_max();
	const __m128 bbox_mid = _mm_mul_ps(
		_mm_add_ps(bbox_min, bbox_max),
		_mm_set1_ps(.5f));

	const BBox child_bbox[8] __attribute__ ((aligned(64))) = {
		BBox(          bbox_min,                                          bbox_mid,                                BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_min[1], bbox_min[2] }, (__m128){ bbox_max[0], bbox_mid[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_mid[1], bbox_min[2] }, (__m128){ bbox_mid[0], bbox_max[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_mid[1], bbox_min[2] }, (__m128){ bbox_max[0], bbox_max[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_min[1], bbox_mid[2] }, (__m128){ bbox_mid[0], bbox_mid[1], bbox_max[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_min[1], bbox_mid[2] }, (__m128){ bbox_max[0], bbox_mid[1], bbox_max[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_mid[1], bbox_mid[2] }, (__m128){ bbox_mid[0], bbox_max[1], bbox_max[2] }, BBox::flag_direct()),
		BBox(          bbox_mid,                                          bbox_max,                                BBox::flag_direct())
	};

	for (size_t i = 0; i < 8; ++i) {

		if (!child_bbox[i].has_overlap_open(payload.get_bbox()))
			continue;

		OctetId child_id = octet.get(i);

		if (OctetId(-1) == child_id) {

			child_id = OctetId(m_interior.getCount());

			if (!m_interior.addElement())
				return false;

			octet.set(i, child_id);
		}

		if (!add_payload< OCTREE_LEVEL_T + 1 >(m_interior.getMutable(child_id), child_bbox[i], payload))
			return false;
	}

	return true;
}


template <>
bool
Timeslice::add_payload< octree_level_last_but_one >(
	Octet& octet,
	const BBox& bbox,
	const Voxel& payload) {

	const __m128 bbox_min = bbox.get_min();
	const __m128 bbox_max = bbox.get_max();
	const __m128 bbox_mid = _mm_mul_ps(
		_mm_add_ps(bbox_min, bbox_max),
		_mm_set1_ps(.5f));

	const BBox child_bbox[8] __attribute__ ((aligned(64))) = {
		BBox(          bbox_min,                                          bbox_mid,                                BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_min[1], bbox_min[2] }, (__m128){ bbox_max[0], bbox_mid[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_mid[1], bbox_min[2] }, (__m128){ bbox_mid[0], bbox_max[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_mid[1], bbox_min[2] }, (__m128){ bbox_max[0], bbox_max[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_min[1], bbox_mid[2] }, (__m128){ bbox_mid[0], bbox_mid[1], bbox_max[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_min[1], bbox_mid[2] }, (__m128){ bbox_max[0], bbox_mid[1], bbox_max[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_mid[1], bbox_mid[2] }, (__m128){ bbox_mid[0], bbox_max[1], bbox_max[2] }, BBox::flag_direct()),
		BBox(          bbox_mid,                                          bbox_max,                                BBox::flag_direct())
	};

	for (size_t i = 0; i < 8; ++i) {

		if (!child_bbox[i].has_overlap_open(payload.get_bbox()))
			continue;

		OctetId child_id = octet.get(i);

		if (OctetId(-1) == child_id) {

			child_id = OctetId(m_leaf.getCount());

			if (!m_leaf.addElement())
				return false;

			m_leaf.getMutable(child_id).init(m_payload.getCount());

			if (!m_payload.addMultiElement(8 * cell_capacity))
				return false;

			octet.set(i, child_id);
		}

		if (!add_payload(m_leaf.getMutable(child_id), child_bbox[i], payload))
			return false;
	}

	return true;
}


bool
Timeslice::add_payload(
	Leaf& leaf,
	const BBox& bbox,
	const Voxel& payload) {

	const __m128 bbox_min = bbox.get_min();
	const __m128 bbox_max = bbox.get_max();
	const __m128 bbox_mid = _mm_mul_ps(
		_mm_add_ps(bbox_min, bbox_max),
		_mm_set1_ps(.5f));

	const BBox child_bbox[8] __attribute__ ((aligned(64))) = {
		BBox(          bbox_min,                                          bbox_mid,                                BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_min[1], bbox_min[2] }, (__m128){ bbox_max[0], bbox_mid[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_mid[1], bbox_min[2] }, (__m128){ bbox_mid[0], bbox_max[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_mid[1], bbox_min[2] }, (__m128){ bbox_max[0], bbox_max[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_min[1], bbox_mid[2] }, (__m128){ bbox_mid[0], bbox_mid[1], bbox_max[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_min[1], bbox_mid[2] }, (__m128){ bbox_max[0], bbox_mid[1], bbox_max[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_mid[1], bbox_mid[2] }, (__m128){ bbox_mid[0], bbox_max[1], bbox_max[2] }, BBox::flag_direct()),
		BBox(          bbox_mid,                                          bbox_max,                                BBox::flag_direct())
	};

	for (size_t i = 0; i < 8; ++i) {

		if (!child_bbox[i].has_overlap_open(payload.get_bbox()))
			continue;

		if (!leaf.add(i, m_payload, payload)) {
			stream::cerr << "failure adding content to leaf: insufficient cell capacity?\n";
			return false;
		}
	}

	return true;
}


void
Timeslice::set_extrnal_storage(
	const size_t interiorCapacity,
	void* const interior,
	const size_t leafCapacity,
	void* const leaf,
	const size_t payloadCapacity,
	void* const payload) {

	assert(0 == interiorCapacity || 0 != interior);
	assert(0 == leafCapacity || 0 != leaf);
	assert(0 == payloadCapacity || 0 != payload);

	m_interior.setCapacity(interiorCapacity, reinterpret_cast< Octet* >(interior));
	m_leaf.setCapacity(leafCapacity, reinterpret_cast< Leaf* >(leaf));
	m_payload.setCapacity(payloadCapacity, reinterpret_cast< Voxel* >(payload));
}


bool
Timeslice::set_payload_array(
	const Array< Voxel >& payload) {

	m_root_bbox = BBox();
	m_interior.resetCount();
	m_leaf.resetCount();
	m_payload.resetCount();

	const size_t item_count = payload.getCount();

	if (item_count == 0)
		return true;

	if (item_count > PayloadId(-1))
		return false;

	// building the octree requires tree's payload bbox in advance
	for (size_t i = 0; i < item_count; ++i) {
		const Voxel& item = payload.getElement(i);

		m_root_bbox.grow(item.get_bbox());
	}

	if (!m_root_bbox.is_valid())
		return false;

	m_interior.addElement(); // root octet

	// feed payload item by item to the tree, building up the tree in the process
	for (size_t i = 0; i < item_count; ++i) {
		const Voxel item = Voxel(payload.getElement(i).get_bbox(), i);

		if (!add_payload< 0 >(m_interior.getMutable(0), m_root_bbox, item))
			return false;
	}

	// compact payload for better locality
	const size_t leaf_count = m_leaf.getCount();
	size_t cursor = 0;

	for (size_t i = 0; i < leaf_count; ++i)
		m_leaf.getMutable(i).compact(cursor, m_payload);

	// check if duplicates cause payload overflow
	if (cursor > PayloadId(-1))
		return false;

	return true;
}
