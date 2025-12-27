// source_buffer
inline struct Octet get_octet(
	__global const ushort4* const octet,
	const uint idx)
{
	return (struct Octet){
		(ushort8)(
			octet[idx * 2 + 0],
			octet[idx * 2 + 1]
		)
	};
}
inline struct Leaf get_leaf(
	__global const ushort4* const leaf,
	const uint idx)
{
	return (struct Leaf){
		(ushort8)(
			leaf[idx * 4 + 0],
			leaf[idx * 4 + 1]
		),
		(ushort8)(
			leaf[idx * 4 + 2],
			leaf[idx * 4 + 3]
		)
	};
}
inline struct Voxel get_voxel(
	__global const float4* const voxel,
	const uint idx)
{
	return (struct Voxel){
		voxel[idx * 2 + 0],
		voxel[idx * 2 + 1]
	};
}

uint traverself(
	const struct Leaf leaf,
	__global const float4* const voxel,
	const struct BBox* const bbox,
	struct Ray* const ray,
	struct Hit* const hit)
{
	struct ChildIndex child_index;

	const uint hitCount = octlf_intersect_wide(
		leaf,
		bbox,
		ray,
		&child_index);

	float8 distance = child_index.distance;
#if OCL_QUIRK_0005 != 0
	ushort8 index = child_index.index;
#else
	ushort8 leaf_start = shuffle(leaf.start, child_index.index);
	ushort8 leaf_count = shuffle(leaf.count, child_index.index);
#endif
	const uint prior_id = as_uint(ray->origin.w);

	for (uint i = 0; i < hitCount; ++i) {
#if OCL_QUIRK_0005 != 0
		const uint payload_start = shuffle(leaf.start, (ushort8)(index.s0)).s0;
		const uint payload_count = shuffle(leaf.count, (ushort8)(index.s0)).s0;
#else
		const uint payload_start = leaf_start.s0;
		const uint payload_count = leaf_count.s0;
#endif
		float nearest_dist = distance.s0;

		distance = distance.s12345677;
#if OCL_QUIRK_0005 != 0
		index = index.s12345677;
#else
		leaf_start = leaf_start.s12345677;
		leaf_count = leaf_count.s12345677;
#endif
		uint voxel_id = -1U;
		struct Hit maybe_hit;

		for (uint j = payload_start; j < payload_start + payload_count; ++j) {
			const struct Voxel payload = get_voxel(voxel, j);
			const struct BBox payload_bbox = { payload.min.xyz, payload.max.xyz };
			const uint id = as_uint(payload.min.w);
			const float dist = intersect(&payload_bbox, ray, &maybe_hit);

			if (id != prior_id & dist < nearest_dist) {
				nearest_dist = dist;
				voxel_id = id;
				*hit = maybe_hit;
			}
		}

		if (-1U != voxel_id) {
			ray->rcpdir.w = nearest_dist;
			return voxel_id;
		}
	}
	return -1U;
}

bool occludelf(
	const struct Leaf leaf,
	__global const float4* const voxel,
	const struct BBox* const bbox,
	const struct Ray* const ray)
{
	struct ChildIndex child_index;

	const uint hitCount = octlf_intersect_wide(
		leaf,
		bbox,
		ray,
		&child_index);

#if OCL_QUIRK_0005 != 0
	ushort8 index = child_index.index;
#else
	ushort8 leaf_start = shuffle(leaf.start, child_index.index);
	ushort8 leaf_count = shuffle(leaf.count, child_index.index);
#endif
	const uint prior_id = as_uint(ray->origin.w);

	for (uint i = 0; i < hitCount; ++i) {
#if OCL_QUIRK_0005 != 0
		const uint payload_start = shuffle(leaf.start, (ushort8)(index.s0)).s0;
		const uint payload_count = shuffle(leaf.count, (ushort8)(index.s0)).s0;
#else
		const uint payload_start = leaf_start.s0;
		const uint payload_count = leaf_count.s0;
#endif
#if OCL_QUIRK_0005 != 0
		index = index.s12345677;
#else
		leaf_start = leaf_start.s12345677;
		leaf_count = leaf_count.s12345677;
#endif
		for (uint j = payload_start; j < payload_start + payload_count; ++j) {
			const struct Voxel payload = get_voxel(voxel, j);
			const struct BBox payload_bbox = { payload.min.xyz, payload.max.xyz };
			const uint id = as_uint(payload.min.w);

			if (id != prior_id & occluded(&payload_bbox, ray))
				return true;
		}
	}
	return false;
}

uint traverse(
	const struct Octet octet,
	__global const ushort4* const leaf,
	__global const float4* const voxel,
	const struct BBox* const bbox,
	struct Ray* const ray,
	struct Hit* const hit)
{
	struct ChildIndex child_index;
	struct BBox child_bbox[8];

	const uint hitCount = octet_intersect_wide(
		octet,
		bbox,
		ray,
		&child_index,
		child_bbox);

	ushort8 index = child_index.index;
#if OCL_QUIRK_0005 == 0
	ushort8 octet_child = shuffle(octet.child, index);
#endif
	for (uint i = 0; i < hitCount; ++i) {
#if OCL_QUIRK_0005 != 0
		const uint child = shuffle(octet.child, (ushort8)(index.s0)).s0;
#else
		const uint child = octet_child.s0;
#endif
		const uint hitId = traverself(get_leaf(leaf, child), voxel, child_bbox + index.s0, ray, hit);

		index = index.s12345677;
#if OCL_QUIRK_0005 == 0
		octet_child = octet_child.s12345677;
#endif
		if (-1U != hitId)
			return hitId;
	}
	return -1U;
}

bool occlude(
	const struct Octet octet,
	__global const ushort4* const leaf,
	__global const float4* const voxel,
	const struct BBox* const bbox,
	const struct Ray* const ray)
{
	struct ChildIndex child_index;
	struct BBox child_bbox[8];

	const uint hitCount = octet_intersect_wide(
		octet,
		bbox,
		ray,
		&child_index,
		child_bbox);

	ushort8 index = child_index.index;
#if OCL_QUIRK_0005 == 0
	ushort8 octet_child = shuffle(octet.child, index);
#endif
	for (uint i = 0; i < hitCount; ++i) {
#if OCL_QUIRK_0005 != 0
		const uint child = shuffle(octet.child, (ushort8)(index.s0)).s0;
#else
		const uint child = octet_child.s0;
#endif
		const struct BBox* const bbox = child_bbox + index.s0;

		index = index.s12345677;
#if OCL_QUIRK_0005 == 0
		octet_child = octet_child.s12345677;
#endif
		if (occludelf(get_leaf(leaf, child), voxel, bbox, ray))
			return true;
	}
	return false;
}

__kernel __attribute__ ((vec_type_hint(float4)))
void monokernel(
	__global const ushort4* const src_a,
	__global const ushort4* const src_b,
	__global const float4* const src_c,
#if OCL_QUIRK_0001
	__constant float* const src_d,
#else
	__constant float4* const src_d,
#endif
#if OCL_OGL_INTEROP
	__write_only image2d_t dst,
#else
	__global uchar* const dst,
#endif
	const uint frame)
{

