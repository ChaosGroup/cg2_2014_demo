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

	const uint8 leaf_start = convert_uint8(leaf.start);
	const uint leafStart[8] = {
		leaf_start.s0,
		leaf_start.s1,
		leaf_start.s2,
		leaf_start.s3,
		leaf_start.s4,
		leaf_start.s5,
		leaf_start.s6,
		leaf_start.s7
	};
	const uint8 leaf_count = convert_uint8(leaf.count);
	const uint leafCount[8] = {
		leaf_count.s0,
		leaf_count.s1,
		leaf_count.s2,
		leaf_count.s3,
		leaf_count.s4,
		leaf_count.s5,
		leaf_count.s6,
		leaf_count.s7
	};
	const float distance[8] = {
		child_index.distance.s0,
		child_index.distance.s1,
		child_index.distance.s2,
		child_index.distance.s3,
		child_index.distance.s4,
		child_index.distance.s5,
		child_index.distance.s6,
		child_index.distance.s7
	};
	const uint index[8] = {
		child_index.index.s0,
		child_index.index.s1,
		child_index.index.s2,
		child_index.index.s3,
		child_index.index.s4,
		child_index.index.s5,
		child_index.index.s6,
		child_index.index.s7
	};
	const uint prior_id = as_uint(ray->origin.w);

	for (uint i = 0; i < hitCount; ++i) {
		const uint payload_start = leafStart[index[i]];
		const uint payload_count = leafCount[index[i]];
		float nearest_dist = distance[i];
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

	const uint8 octet_child = convert_uint8(octet.child);
	const uint octetChild[8] = {
		octet_child.s0,
		octet_child.s1,
		octet_child.s2,
		octet_child.s3,
		octet_child.s4,
		octet_child.s5,
		octet_child.s6,
		octet_child.s7
	};
	const uint index[8] = {
		child_index.index.s0,
		child_index.index.s1,
		child_index.index.s2,
		child_index.index.s3,
		child_index.index.s4,
		child_index.index.s5,
		child_index.index.s6,
		child_index.index.s7
	};
	for (uint i = 0; i < hitCount; ++i) {
		const uint hitId = traverself(get_leaf(leaf, octetChild[index[i]]), voxel, child_bbox + index[i], ray, hit);
		if (-1U != hitId)
			return hitId;
	}
	return -1U;
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

