#include <istream>
#include <ostream>
#include <limits>
#include "vectsimd_sse.hpp"
#include "array.hpp"
#include "isfinite.hpp"
#include "stream.hpp"
#include "problem_4.hpp"

// verify iostream-free status
#if _GLIBCXX_IOSTREAM
#error rogue iostream acquired
#endif

__thread const Ray* Timeslice::m_ray;
__thread HitInfo* Timeslice::m_hit;


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


bool
Ray::flatten(
	std::ostream& str) const
{
	if (!str.good())
		return false;

	str << m_origin << ' ' << m_direction;

	return str.good();
}


bool
Ray::deflatten(
	std::istream& str)
{
	if (!str.good())
		return false;

	simd::vect3 temp[2];

	str >> temp[0];

	if (!str.good())
		return false;

	str >> temp[1];

	if (!str.good())
		return false;

	for (size_t i = 0; i < sizeof(temp) / sizeof(temp[0]); ++i)
		for (size_t j = 0; j < temp[i].dimension; ++j)
			if (!isfinite(temp[i].get(j)))
				return false;

	m_origin = temp[0];
	m_direction = temp[1].normalise();

#if RAY_HIGH_PRECISION_RCP_DIR == 1
	const __m128 rcp = _mm_div_ps(_mm_set1_ps(1.f), temp[1].getn());

#else
	const __m128 rcp = _mm_rcp_ps(temp[1].getn());

#endif
	// substitute infinities for max floats in the reciprocal direction
	m_rcpdir.setn(0, _mm_max_ps(_mm_min_ps(rcp,
		_mm_set1_ps( std::numeric_limits< float >::max())),
		_mm_set1_ps(-std::numeric_limits< float >::max())));

#if __AVX__ == 0
	m_origin_x = _mm_shuffle_ps(temp[0].getn(), temp[0].getn(), 0);
	m_origin_y = _mm_shuffle_ps(temp[0].getn(), temp[0].getn(), 0x55);
	m_origin_z = _mm_shuffle_ps(temp[0].getn(), temp[0].getn(), 0xaa);

	m_rcpdir_x = _mm_shuffle_ps(m_rcpdir.getn(), m_rcpdir.getn(), 0);
	m_rcpdir_y = _mm_shuffle_ps(m_rcpdir.getn(), m_rcpdir.getn(), 0x55);
	m_rcpdir_z = _mm_shuffle_ps(m_rcpdir.getn(), m_rcpdir.getn(), 0xaa);

#endif
	return true;
}


template < unsigned OCTREE_LEVEL_T >
bool
Timeslice::add_payload(
	Octet& octet,
	const BBox& bbox,
	const Voxel& payload)
{
	const __m128 bbox_min = bbox.get_min();
	const __m128 bbox_max = bbox.get_max();
	const __m128 bbox_mid = _mm_mul_ps(
		_mm_add_ps(bbox_min, bbox_max),
		_mm_set1_ps(.5f));

	const BBox child_bbox[8] __attribute__ ((aligned(64))) =
	{
		BBox(          bbox_min,                                          bbox_mid,                                BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_min[1], bbox_min[2] }, (__m128){ bbox_max[0], bbox_mid[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_mid[1], bbox_min[2] }, (__m128){ bbox_mid[0], bbox_max[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_mid[1], bbox_min[2] }, (__m128){ bbox_max[0], bbox_max[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_min[1], bbox_mid[2] }, (__m128){ bbox_mid[0], bbox_mid[1], bbox_max[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_min[1], bbox_mid[2] }, (__m128){ bbox_max[0], bbox_mid[1], bbox_max[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_mid[1], bbox_mid[2] }, (__m128){ bbox_mid[0], bbox_max[1], bbox_max[2] }, BBox::flag_direct()),
		BBox(          bbox_mid,                                          bbox_max,                                BBox::flag_direct())
	};

	for (size_t i = 0; i < 8; ++i)
	{
		if (!child_bbox[i].has_overlap_open(payload.get_bbox()))
			continue;

		OctetId child_id = octet.get(i);

		if (OctetId(-1) == child_id)
		{
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
	const Voxel& payload)
{
	const __m128 bbox_min = bbox.get_min();
	const __m128 bbox_max = bbox.get_max();
	const __m128 bbox_mid = _mm_mul_ps(
		_mm_add_ps(bbox_min, bbox_max),
		_mm_set1_ps(.5f));

	const BBox child_bbox[8] __attribute__ ((aligned(64))) =
	{
		BBox(          bbox_min,                                          bbox_mid,                                BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_min[1], bbox_min[2] }, (__m128){ bbox_max[0], bbox_mid[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_mid[1], bbox_min[2] }, (__m128){ bbox_mid[0], bbox_max[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_mid[1], bbox_min[2] }, (__m128){ bbox_max[0], bbox_max[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_min[1], bbox_mid[2] }, (__m128){ bbox_mid[0], bbox_mid[1], bbox_max[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_min[1], bbox_mid[2] }, (__m128){ bbox_max[0], bbox_mid[1], bbox_max[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_mid[1], bbox_mid[2] }, (__m128){ bbox_mid[0], bbox_max[1], bbox_max[2] }, BBox::flag_direct()),
		BBox(          bbox_mid,                                          bbox_max,                                BBox::flag_direct())
	};

	for (size_t i = 0; i < 8; ++i)
	{
		if (!child_bbox[i].has_overlap_open(payload.get_bbox()))
			continue;

		OctetId child_id = octet.get(i);

		if (OctetId(-1) == child_id)
		{
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
	const Voxel& payload)
{
	const __m128 bbox_min = bbox.get_min();
	const __m128 bbox_max = bbox.get_max();
	const __m128 bbox_mid = _mm_mul_ps(
		_mm_add_ps(bbox_min, bbox_max),
		_mm_set1_ps(.5f));

	const BBox child_bbox[8] __attribute__ ((aligned(64))) =
	{
		BBox(          bbox_min,                                          bbox_mid,                                BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_min[1], bbox_min[2] }, (__m128){ bbox_max[0], bbox_mid[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_mid[1], bbox_min[2] }, (__m128){ bbox_mid[0], bbox_max[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_mid[1], bbox_min[2] }, (__m128){ bbox_max[0], bbox_max[1], bbox_mid[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_min[1], bbox_mid[2] }, (__m128){ bbox_mid[0], bbox_mid[1], bbox_max[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_mid[0], bbox_min[1], bbox_mid[2] }, (__m128){ bbox_max[0], bbox_mid[1], bbox_max[2] }, BBox::flag_direct()),
		BBox((__m128){ bbox_min[0], bbox_mid[1], bbox_mid[2] }, (__m128){ bbox_mid[0], bbox_max[1], bbox_max[2] }, BBox::flag_direct()),
		BBox(          bbox_mid,                                          bbox_max,                                BBox::flag_direct())
	};

	for (size_t i = 0; i < 8; ++i)
	{
		if (!child_bbox[i].has_overlap_open(payload.get_bbox()))
			continue;

		if (!leaf.add(i, m_payload, payload))
		{
			stream::cerr << "failure adding content to leaf: insufficient cell capacity?\n";
			return false;
		}
	}

	return true;
}


bool
Timeslice::set_payload_array(
	const Array< Voxel >& payload)
{
	m_root_bbox = BBox();
	m_root = Octet();

	const size_t item_count = payload.getCount();

	if (item_count == 0)
	{
		m_payload.resetCount();
		return true;
	}

	if (item_count > PayloadId(-1))
		return false;

	for (size_t i = 0; i < item_count; ++i)
	{
		const Voxel& item = payload.getElement(i);

		m_root_bbox.grow(item.get_bbox());
	}

	if (!m_root_bbox.is_valid())
		return false;

	if (!m_interior.setCapacity(octree_interior_count))
		return false;

	if (!m_leaf.setCapacity(octree_leaf_count))
		return false;

	if (!m_payload.setCapacity(octree_cell_count * cell_capacity))
		return false;

	// feed payload item by item to the tree, building up the tree in the process
	for (size_t i = 0; i < item_count; ++i)
	{
		const Voxel item = Voxel(payload.getElement(i).get_bbox(), i);

		if (!add_payload< 0 >(m_root, m_root_bbox, item))
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


#if CLANG_QUIRK_0001 != 0
bool
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

bool
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

bool
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
