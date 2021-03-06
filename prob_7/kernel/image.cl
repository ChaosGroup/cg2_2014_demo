// source_image
__constant sampler_t sampler_a = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t sampler_b = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t sampler_c = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

inline struct Octet get_octet(
	__read_only image2d_t octet,
	const uint idx)
{
	return (struct Octet){
		(ushort8)(
			convert_ushort4(read_imageui(octet, sampler_a, (int2)(0, idx))),
			convert_ushort4(read_imageui(octet, sampler_a, (int2)(1, idx)))
		)
	};
}
inline struct Leaf get_leaf(
	__read_only image2d_t leaf,
	const uint idx)
{
	return (struct Leaf){
		(ushort8)(
			convert_ushort4(read_imageui(leaf, sampler_b, (int2)(0, idx))),
			convert_ushort4(read_imageui(leaf, sampler_b, (int2)(1, idx)))
		),
		(ushort8)(
			convert_ushort4(read_imageui(leaf, sampler_b, (int2)(2, idx))),
			convert_ushort4(read_imageui(leaf, sampler_b, (int2)(3, idx)))
		)
	};
}
inline struct Voxel get_voxel(
	__read_only image2d_t voxel,
	const uint idx)
{
	return (struct Voxel){
		read_imagef(voxel, sampler_c, (int2)(0, idx)),
		read_imagef(voxel, sampler_c, (int2)(1, idx))
	};
}

uint traverself(
	const struct Leaf leaf,
	__read_only image2d_t voxel,
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

		distance = distance.s12345677;
		index = index.s12345677;

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
	__read_only image2d_t voxel,
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

		index = index.s12345677;

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
	__read_only image2d_t leaf,
	__read_only image2d_t voxel,
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

		index = index.s12345677;

		if (-1U != hitId)
			return hitId;
	}
	return -1U;
}

bool occlude(
	const struct Octet octet,
	__read_only image2d_t leaf,
	__read_only image2d_t voxel,
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

		index = index.s12345677;

		if (occludelf(get_leaf(leaf, child), voxel, bbox, ray))
			return true;
	}
	return false;
}

__kernel __attribute__ ((vec_type_hint(float4)))
void monokernel(
	__read_only image2d_t src_a,
	__read_only image2d_t src_b,
	__read_only image2d_t src_c,
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

