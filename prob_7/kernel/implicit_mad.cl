// source_implicit_mad
	const int idx = (int)get_global_id(0);
	const int idy = (int)get_global_id(1);
	const int dimx = (int)get_global_size(0);
	const int dimy = (int)get_global_size(1);
#if OCL_QUIRK_0001
	const float3 cam0 = (float3)(src_d[0], src_d[1], src_d[ 2]);
	const float3 cam1 = (float3)(src_d[4], src_d[5], src_d[ 6]);
	const float3 cam2 = (float3)(src_d[8], src_d[9], src_d[10]);
	const float3 ray_origin = (float3)(src_d[12], src_d[13], src_d[14]);
	const float3 bbox_min   = (float3)(src_d[16], src_d[17], src_d[18]);
	const float3 bbox_max   = (float3)(src_d[20], src_d[21], src_d[22]);
#else
	const float3 cam0 = src_d[0].xyz;
	const float3 cam1 = src_d[1].xyz;
	const float3 cam2 = src_d[2].xyz;
	const float3 ray_origin = src_d[3].xyz;
	const float3 bbox_min   = src_d[4].xyz;
	const float3 bbox_max   = src_d[5].xyz;
#endif
	const struct BBox root_bbox = (struct BBox){ bbox_min, bbox_max };
	const float3 ray_direction =
		cam0 * ((idx * 2 - dimx) * (1.0f / dimx)) +
		cam1 * ((idy * 2 - dimy) * (1.0f / dimy)) +
		cam2;
	const float3 ray_rcpdir = clamp(1.f / ray_direction, -MAXFLOAT, MAXFLOAT);
	struct RayHit ray = (struct RayHit){ (struct Ray){ (float4)(ray_origin, as_float(-1U)), (float4)(ray_rcpdir, MAXFLOAT) } };
	uint result = traverse(get_octet(src_a, 0), src_b, src_c, &root_bbox, &ray.ray, &ray.hit);
	const uint axis_x = 0x020100;
	const uint axis_y = 0x010002;
	const uint axis_z = 0x000201;
	const int a_mask = -ray.hit.a_mask;
	const int b_mask = -ray.hit.b_mask;
	const uint axis = (axis_x & a_mask | axis_y & ~a_mask) & b_mask | axis_z & ~b_mask;
#if 1
	const int3 axis_sign = (int3)(0x80000000) & ray.hit.min_mask;
	const float3 normal = shuffle((float4)(1.f, 0.f, 0.f, 0.f), convert_uint4((uchar4)(axis >> 0, axis >> 8, axis >> 16, axis >> 24))).xyz;
	const uint luma = convert_int(max(1.f / 16.f, dot(as_float3(as_int3(normal) ^ axis_sign), sun)) * 255.f);

	if (-1U != result) {
		const float dist = ray.ray.rcpdir.w;
		const float3 ray_rcpdir = clamp(1.f / sun, -MAXFLOAT, MAXFLOAT);
		struct RayHit occl = (struct RayHit){ (struct Ray){ (float4)(ray_origin + ray_direction * dist, as_float(result)), (float4)(ray_rcpdir, MAXFLOAT) } };
		result = select(convert_uint(1.f / 16.f * 255.f), luma, -1U == traverse(get_octet(src_a, 0), src_b, src_c, &root_bbox, &occl.ray, &occl.hit));
	}
	else
		result = 0;
#else
	result = -1U != result ? (axis & 0x3) * 64 + 64 : 0;
#endif

