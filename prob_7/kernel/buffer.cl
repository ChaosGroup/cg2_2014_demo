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
	uint8 index = child_index.index;
	const uint8 leaf_start = convert_uint8(leaf.start);
	const uint8 leaf_count = convert_uint8(leaf.count);
	const uint prior_id = as_uint(ray->origin.w);

	for (uint i = 0; i < hitCount; ++i) {
		const uint payload_start = shuffle(leaf_start, (uint8)(index.s0)).s0;
		const uint payload_count = shuffle(leaf_count, (uint8)(index.s0)).s0;
		float nearest_dist = distance.s0;

		distance = shuffle(distance, (uint8)(1, 2, 3, 4, 5, 6, 7, 7));
		index    = shuffle(index,    (uint8)(1, 2, 3, 4, 5, 6, 7, 7));

		uint voxel_id = -1U;
		struct Hit maybe_hit;

		for (uint j = payload_start; j < payload_start + payload_count; ++j) {
			const struct Voxel payload = get_voxel(voxel, j);
			const struct BBox payload_bbox = (struct BBox){ payload.min.xyz, payload.max.xyz };
			const uint id = as_uint(payload.min.w);
			const float dist = intersect(&payload_bbox, ray, &maybe_hit);

			if (id != prior_id & dist < nearest_dist) {
				nearest_dist = dist;
				voxel_id = id;
				ray->rcpdir.w = dist;
				*hit = maybe_hit;
			}
		}

		if (-1U != voxel_id)
			return voxel_id;
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

	uint8 index = child_index.index;
	const uint8 leaf_start = convert_uint8(leaf.start);
	const uint8 leaf_count = convert_uint8(leaf.count);
	const uint prior_id = as_uint(ray->origin.w);

	for (uint i = 0; i < hitCount; ++i) {
		const uint payload_start = shuffle(leaf_start, (uint8)(index.s0)).s0;
		const uint payload_count = shuffle(leaf_count, (uint8)(index.s0)).s0;

		index = shuffle(index, (uint8)(1, 2, 3, 4, 5, 6, 7, 7));

		for (uint j = payload_start; j < payload_start + payload_count; ++j) {
			const struct Voxel payload = get_voxel(voxel, j);
			const struct BBox payload_bbox = (struct BBox){ payload.min.xyz, payload.max.xyz };
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

	uint8 index = child_index.index;
	const uint8 octet_child = convert_uint8(octet.child);

	for (uint i = 0; i < hitCount; ++i) {
		const uint child = shuffle(octet_child, (uint8)(index.s0)).s0;
		const uint hitId = traverself(get_leaf(leaf, child), voxel, child_bbox + index.s0, ray, hit);

		index = shuffle(index, (uint8)(1, 2, 3, 4, 5, 6, 7, 7));

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

	uint8 index = child_index.index;
	const uint8 octet_child = convert_uint8(octet.child);

	for (uint i = 0; i < hitCount; ++i) {
		const uint child = shuffle(octet_child, (uint8)(index.s0)).s0;
		const struct BBox* const bbox = child_bbox + index.s0;

		index = shuffle(index, (uint8)(1, 2, 3, 4, 5, 6, 7, 7));

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
	__write_only image2d_t dst)
#else
	__global uchar* const dst)
#endif
{

