#include <istream>
#include <ostream>
#include <limits>
#include "vectnative.hpp"
#include "array.hpp"
#include "isfinite.hpp"
#include "stream.hpp"
#include "problem_6.hpp"

// verify iostream-free status
#if _GLIBCXX_IOSTREAM
#error rogue iostream acquired
#endif

template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline std::ostream&
operator <<(
	std::ostream& str,
	const simd::native4< SCALAR_T, GENERIC_T, NATIVE_T >& v)
{
	enum { dimension = simd::native4< SCALAR_T, GENERIC_T, NATIVE_T >::dimension };

	str << '(';

	size_t i = 0;

	for (; i < dimension - 1 && str.good(); ++i)
		str << v.get(i) << ", ";

	str << v.get(i) << ')';

	return str;
}


template < typename SCALAR_T, typename GENERIC_T, typename NATIVE_T >
inline std::istream&
operator >>(
	std::istream& str,
	simd::native4< SCALAR_T, GENERIC_T, NATIVE_T >& v)
{
	enum { dimension = simd::native4< SCALAR_T, GENERIC_T, NATIVE_T >::dimension };

	float temp[dimension];

	for (size_t i = 0; i < dimension; ++i)
		temp[i] = std::numeric_limits< float >::quiet_NaN();

	char sep;
	str >> sep;

	if ('(' != sep)
		str.setstate(std::istream::failbit);

	size_t i = 0;

	for (; i < dimension - 1 && str.good(); ++i)
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

	for (size_t i = 0; i < dimension; ++i)
		v.set(i, temp[i]);

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

	simd::f32x4 temp[2];
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

	using simd::f32x4;
	using simd::flag_zero;
	const f32x4 bbox_min = bbox.get_min();
	const f32x4 bbox_max = bbox.get_max();
	const f32x4 bbox_mid = (bbox_min + bbox_max) * f32x4(.5f);

	const BBox child_bbox[8] __attribute__ ((aligned(64))) = {
		BBox(       bbox_min,                                                    bbox_mid,                                             BBox::flag_direct()),
		BBox(f32x4( bbox_mid[0], bbox_min[1], bbox_min[2], flag_zero() ), f32x4( bbox_max[0], bbox_mid[1], bbox_mid[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_min[0], bbox_mid[1], bbox_min[2], flag_zero() ), f32x4( bbox_mid[0], bbox_max[1], bbox_mid[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_mid[0], bbox_mid[1], bbox_min[2], flag_zero() ), f32x4( bbox_max[0], bbox_max[1], bbox_mid[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_min[0], bbox_min[1], bbox_mid[2], flag_zero() ), f32x4( bbox_mid[0], bbox_mid[1], bbox_max[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_mid[0], bbox_min[1], bbox_mid[2], flag_zero() ), f32x4( bbox_max[0], bbox_mid[1], bbox_max[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_min[0], bbox_mid[1], bbox_mid[2], flag_zero() ), f32x4( bbox_mid[0], bbox_max[1], bbox_max[2], flag_zero() ), BBox::flag_direct()),
		BBox(       bbox_mid,                                                    bbox_max,                                             BBox::flag_direct())
	};

	for (size_t i = 0; i < 8; ++i) {

		if (!child_bbox[i].has_overlap_open(payload.get_bbox()))
			continue;

		OctetId child_id = octet.get(i);

		if (OctetId(octet_empty) == child_id) {

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

	using simd::f32x4;
	using simd::flag_zero;
	const f32x4 bbox_min = bbox.get_min();
	const f32x4 bbox_max = bbox.get_max();
	const f32x4 bbox_mid = (bbox_min + bbox_max) * f32x4(.5f);

	const BBox child_bbox[8] __attribute__ ((aligned(64))) = {
		BBox(       bbox_min,                                                    bbox_mid,                                             BBox::flag_direct()),
		BBox(f32x4( bbox_mid[0], bbox_min[1], bbox_min[2], flag_zero() ), f32x4( bbox_max[0], bbox_mid[1], bbox_mid[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_min[0], bbox_mid[1], bbox_min[2], flag_zero() ), f32x4( bbox_mid[0], bbox_max[1], bbox_mid[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_mid[0], bbox_mid[1], bbox_min[2], flag_zero() ), f32x4( bbox_max[0], bbox_max[1], bbox_mid[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_min[0], bbox_min[1], bbox_mid[2], flag_zero() ), f32x4( bbox_mid[0], bbox_mid[1], bbox_max[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_mid[0], bbox_min[1], bbox_mid[2], flag_zero() ), f32x4( bbox_max[0], bbox_mid[1], bbox_max[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_min[0], bbox_mid[1], bbox_mid[2], flag_zero() ), f32x4( bbox_mid[0], bbox_max[1], bbox_max[2], flag_zero() ), BBox::flag_direct()),
		BBox(       bbox_mid,                                                    bbox_max,                                             BBox::flag_direct())
	};

	for (size_t i = 0; i < 8; ++i) {

		if (!child_bbox[i].has_overlap_open(payload.get_bbox()))
			continue;

		OctetId child_id = octet.get(i);

		if (OctetId(octet_empty) == child_id) {

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

	using simd::f32x4;
	using simd::flag_zero;
	const f32x4 bbox_min = bbox.get_min();
	const f32x4 bbox_max = bbox.get_max();
	const f32x4 bbox_mid = (bbox_min + bbox_max) * f32x4(.5f);

	const BBox child_bbox[8] __attribute__ ((aligned(64))) = {
		BBox(       bbox_min,                                                    bbox_mid,                                             BBox::flag_direct()),
		BBox(f32x4( bbox_mid[0], bbox_min[1], bbox_min[2], flag_zero() ), f32x4( bbox_max[0], bbox_mid[1], bbox_mid[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_min[0], bbox_mid[1], bbox_min[2], flag_zero() ), f32x4( bbox_mid[0], bbox_max[1], bbox_mid[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_mid[0], bbox_mid[1], bbox_min[2], flag_zero() ), f32x4( bbox_max[0], bbox_max[1], bbox_mid[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_min[0], bbox_min[1], bbox_mid[2], flag_zero() ), f32x4( bbox_mid[0], bbox_mid[1], bbox_max[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_mid[0], bbox_min[1], bbox_mid[2], flag_zero() ), f32x4( bbox_max[0], bbox_mid[1], bbox_max[2], flag_zero() ), BBox::flag_direct()),
		BBox(f32x4( bbox_min[0], bbox_mid[1], bbox_mid[2], flag_zero() ), f32x4( bbox_mid[0], bbox_max[1], bbox_max[2], flag_zero() ), BBox::flag_direct()),
		BBox(       bbox_mid,                                                    bbox_max,                                             BBox::flag_direct())
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
	const Array< Voxel >& payload,
	const BBox& root_bbox) {

	m_root_bbox = BBox();
	m_interior.resetCount();
	m_leaf.resetCount();
	m_payload.resetCount();

	const size_t item_count = payload.getCount();

	if (item_count == 0)
		return true;

	if (item_count > PayloadId(-1))
		return false;

	if (!root_bbox.is_valid())
		return false;

	m_root_bbox = root_bbox;
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
