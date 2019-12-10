// source_prologue
#if 0
constant float3 sun = (float3)(0.57735026919f, -0.57735026919f, 0.57735026919f);
#else
constant float3 sun = (float3)(0.87287156094f, -0.21821789023f, 0.43643578047f);
#endif
struct BBox {
	float3 min;
	float3 max;
};

struct Ray {
	float4 origin; // .xyz = origin, .w = as_float(prior_id)
	float4 rcpdir; // .xyz = rcpdir, .w = dist
};

struct Hit {
	int3 min_mask;
	int a_mask;
	int b_mask;
};

struct RayHit {
	struct Ray ray;
	struct Hit hit;
};

struct Octet {
	ushort8 child;
};

struct Leaf {
	ushort8 start;
	ushort8 count;
};

struct Voxel {
	float4 min;
	float4 max;
};

struct ChildIndex {
	float8 distance;
	uint8 index;
};

inline float intersect(
	const struct BBox* const bbox,
	const struct Ray* const ray,
	struct Hit* const hit)
{
	const float3 t0 = (bbox->min - ray->origin.xyz) * ray->rcpdir.xyz;
	const float3 t1 = (bbox->max - ray->origin.xyz) * ray->rcpdir.xyz;
	const float ray_len = ray->rcpdir.w;

	const float3 axial_min = fmin(t0, t1);
	const float3 axial_max = fmax(t0, t1);

	hit->min_mask = islessequal(t0, t1);
	hit->a_mask = isgreaterequal(axial_min.x, axial_min.y);
	hit->b_mask = isgreaterequal(fmax(axial_min.x, axial_min.y), axial_min.z);

	const float min = fmax(fmax(axial_min.x, axial_min.y), axial_min.z);
	const float max = fmin(fmin(axial_max.x, axial_max.y), axial_max.z);

#if INFINITE_RAY
	return select(INFINITY, min, isless(0.f, min) & isless(min, max));
#else
	return select(INFINITY, min, isless(0.f, min) & isless(min, max) & isless(min, ray_len));
#endif
}

inline bool occluded(
	const struct BBox* const bbox,
	const struct Ray* const ray)
{
	const float3 t0 = (bbox->min - ray->origin.xyz) * ray->rcpdir.xyz;
	const float3 t1 = (bbox->max - ray->origin.xyz) * ray->rcpdir.xyz;
	const float ray_len = ray->rcpdir.w;

	const float3 axial_min = fmin(t0, t1);
	const float3 axial_max = fmax(t0, t1);

	const float min = fmax(fmax(axial_min.x, axial_min.y), axial_min.z);
	const float max = fmin(fmin(axial_max.x, axial_max.y), axial_max.z);

#if INFINITE_RAY
	return isless(0.f, min) & isless(min, max);
#else
	return isless(0.f, min) & isless(min, max) & isless(min, ray_len);
#endif
}

inline void intersect8(
	const float8 bbox_min_x,
	const float8 bbox_min_y,
	const float8 bbox_min_z,
	const float8 bbox_max_x,
	const float8 bbox_max_y,
	const float8 bbox_max_z,
	const struct Ray* const ray,
	float8* const t,
	int8* const r)
{
	const float3 ray_origin = ray->origin.xyz;
	const float3 ray_rcpdir = ray->rcpdir.xyz;
	const float ray_len = ray->rcpdir.w;

	const float8 tmin_x = (bbox_min_x - ray_origin.xxxxxxxx) * ray_rcpdir.xxxxxxxx;
	const float8 tmax_x = (bbox_max_x - ray_origin.xxxxxxxx) * ray_rcpdir.xxxxxxxx;
	const float8 tmin_y = (bbox_min_y - ray_origin.yyyyyyyy) * ray_rcpdir.yyyyyyyy;
	const float8 tmax_y = (bbox_max_y - ray_origin.yyyyyyyy) * ray_rcpdir.yyyyyyyy;
	const float8 tmin_z = (bbox_min_z - ray_origin.zzzzzzzz) * ray_rcpdir.zzzzzzzz;
	const float8 tmax_z = (bbox_max_z - ray_origin.zzzzzzzz) * ray_rcpdir.zzzzzzzz;

	const float8 x_min = fmin(tmin_x, tmax_x);
	const float8 x_max = fmax(tmin_x, tmax_x);
	const float8 y_min = fmin(tmin_y, tmax_y);
	const float8 y_max = fmax(tmin_y, tmax_y);
	const float8 z_min = fmin(tmin_z, tmax_z);
	const float8 z_max = fmax(tmin_z, tmax_z);

	const float8 min = fmax(fmax(x_min, y_min), z_min);
	const float8 max = fmin(fmin(x_max, y_max), z_max);
	*t = max;

#if INFINITE_RAY
	const int8 msk = isless(min, max) & isless((float8)(0.f), max);
#else
	const int8 msk = isless(min, max) & isless((float8)(0.f), max) & isless(min, (float8)(ray_len));
#endif
	*r = msk;
}

uint octlf_intersect_wide(
	const struct Leaf octet,
	const struct BBox* const bbox,
	const struct Ray* const ray,
	struct ChildIndex* const child_index)
{
	const float3 par_min = bbox->min;
	const float3 par_max = bbox->max;
	const float3 par_mid = (par_min + par_max) * 0.5f;

	const float8 bbox_min_x = (float8)( par_min.x, par_mid.x, par_min.x, par_mid.x, par_min.x, par_mid.x, par_min.x, par_mid.x );
	const float8 bbox_min_y = (float8)( par_min.yy, par_mid.yy, par_min.yy, par_mid.yy );
	const float8 bbox_min_z = (float8)( par_min.zzzz, par_mid.zzzz );
	const float8 bbox_max_x = (float8)( par_mid.x, par_max.x, par_mid.x, par_max.x, par_mid.x, par_max.x, par_mid.x, par_max.x );
	const float8 bbox_max_y = (float8)( par_mid.yy, par_max.yy, par_mid.yy, par_max.yy );
	const float8 bbox_max_z = (float8)( par_mid.zzzz, par_max.zzzz );

	float8 t;
	int8 r;
	intersect8(bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z, ray, &t, &r);
#if OCL_QUIRK_0003
	const int8 occupancy = as_int8(((ushort8)(0) != octet.count).s0011223344556677);
#else
	const int8 occupancy = convert_int8((ushort8)(0) != octet.count);
#endif
	r &= occupancy;

#if OCL_QUIRK_0004
	const int8 cnt0 = -r;
	const int4 cnt1 = cnt0.s0123 + cnt0.s4567;
	const int2 cnt2 = cnt1.s01   + cnt1.s23;
	const int count = cnt2.s0    + cnt2.s1;

#else
	int count = 0;
	count -= r.s0;
	count -= r.s1;
	count -= r.s2;
	count -= r.s3;
	count -= r.s4;
	count -= r.s5;
	count -= r.s6;
	count -= r.s7;

#endif
	t = select((float8)(INFINITY), t, r);

	const float4 r0_A = (float4)(t.s0, t.s3, t.s4, t.s7);
	const float4 r0_B = (float4)(t.s1, t.s2, t.s5, t.s6);
	const int4 r0x_A = (int4)(0, 3, 4, 7);
	const int4 r0x_B = (int4)(1, 2, 5, 6);
	const int4 m0 = islessequal(r0_A, r0_B);
	const float4 r0_min = fmin(r0_A, r0_B);
	const float4 r0_max = fmax(r0_A, r0_B);
	const int4 r0x_min = select(r0x_B, r0x_A, m0);
	const int4 r0x_max = select(r0x_A, r0x_B, m0);

	const float4 r1_A = (float4)(r0_min.s0, r0_max.s0, r0_max.s3, r0_min.s3);
	const float4 r1_B = (float4)(r0_max.s1, r0_min.s1, r0_min.s2, r0_max.s2);
	const int4 r1x_A = (int4)(r0x_min.s0, r0x_max.s0, r0x_max.s3, r0x_min.s3);
	const int4 r1x_B = (int4)(r0x_max.s1, r0x_min.s1, r0x_min.s2, r0x_max.s2);
	const int4 m1 = islessequal(r1_A, r1_B);
	const float4 r1_min = fmin(r1_A, r1_B);
	const float4 r1_max = fmax(r1_A, r1_B);
	const int4 r1x_min = select(r1x_B, r1x_A, m1);
	const int4 r1x_max = select(r1x_A, r1x_B, m1);

	const float4 r2_A = (float4)(r1_min.s0, r1_max.s0, r1_max.s3, r1_min.s3);
	const float4 r2_B = (float4)(r1_min.s1, r1_max.s1, r1_max.s2, r1_min.s2);
	const int4 r2x_A = (int4)(r1x_min.s0, r1x_max.s0, r1x_max.s3, r1x_min.s3);
	const int4 r2x_B = (int4)(r1x_min.s1, r1x_max.s1, r1x_max.s2, r1x_min.s2);
	const int4 m2 = islessequal(r2_A, r2_B);
	const float4 r2_min = fmin(r2_A, r2_B);
	const float4 r2_max = fmax(r2_A, r2_B);
	const int4 r2x_min = select(r2x_B, r2x_A, m2);
	const int4 r2x_max = select(r2x_A, r2x_B, m2);

	const float4 r3_A = (float4)(r2_min.s0, r2_max.s0, r2_min.s1, r2_max.s1);
	const float4 r3_B = (float4)(r2_max.s2, r2_min.s2, r2_max.s3, r2_min.s3);
	const int4 r3x_A = (int4)(r2x_min.s0, r2x_max.s0, r2x_min.s1, r2x_max.s1);
	const int4 r3x_B = (int4)(r2x_max.s2, r2x_min.s2, r2x_max.s3, r2x_min.s3);
	const int4 m3 = islessequal(r3_A, r3_B);
	const float4 r3_min = fmin(r3_A, r3_B);
	const float4 r3_max = fmax(r3_A, r3_B);
	const int4 r3x_min = select(r3x_B, r3x_A, m3);
	const int4 r3x_max = select(r3x_A, r3x_B, m3);

	const float4 r4_A = (float4)(r3_min.s0, r3_min.s1, r3_max.s0, r3_max.s1);
	const float4 r4_B = (float4)(r3_min.s2, r3_min.s3, r3_max.s2, r3_max.s3);
	const int4 r4x_A = (int4)(r3x_min.s0, r3x_min.s1, r3x_max.s0, r3x_max.s1);
	const int4 r4x_B = (int4)(r3x_min.s2, r3x_min.s3, r3x_max.s2, r3x_max.s3);
	const int4 m4 = islessequal(r4_A, r4_B);
	const float4 r4_min = fmin(r4_A, r4_B);
	const float4 r4_max = fmax(r4_A, r4_B);
	const int4 r4x_min = select(r4x_B, r4x_A, m4);
	const int4 r4x_max = select(r4x_A, r4x_B, m4);

	const float4 r5_A = (float4)(r4_min.s0, r4_max.s0, r4_min.s2, r4_max.s2);
	const float4 r5_B = (float4)(r4_min.s1, r4_max.s1, r4_min.s3, r4_max.s3);
	const int4 r5x_A = (int4)(r4x_min.s0, r4x_max.s0, r4x_min.s2, r4x_max.s2);
	const int4 r5x_B = (int4)(r4x_min.s1, r4x_max.s1, r4x_min.s3, r4x_max.s3);
	const int4 m5 = islessequal(r5_A, r5_B);
	const float4 r5_min = fmin(r5_A, r5_B);
	const float4 r5_max = fmax(r5_A, r5_B);
	const int4 r5x_min = select(r5x_B, r5x_A, m5);
	const int4 r5x_max = select(r5x_A, r5x_B, m5);

	child_index->distance = (float8)(r5_min.s0, r5_max.s0, r5_min.s1, r5_max.s1, r5_min.s2, r5_max.s2, r5_min.s3, r5_max.s3);
	child_index->index = (uint8)(r5x_min.s0, r5x_max.s0, r5x_min.s1, r5x_max.s1, r5x_min.s2, r5x_max.s2, r5x_min.s3, r5x_max.s3);
	return as_uint(count);
}

uint octet_intersect_wide(
	const struct Octet octet,
	const struct BBox* const bbox,
	const struct Ray* const ray,
	struct ChildIndex* const child_index,
	struct BBox child_bbox[8])
{
	const float3 par_min = bbox->min;
	const float3 par_max = bbox->max;
	const float3 par_mid = (par_min + par_max) * 0.5f;

	const float8 bbox_min_x = (float8)( par_min.x, par_mid.x, par_min.x, par_mid.x, par_min.x, par_mid.x, par_min.x, par_mid.x );
	const float8 bbox_min_y = (float8)( par_min.yy, par_mid.yy, par_min.yy, par_mid.yy );
	const float8 bbox_min_z = (float8)( par_min.zzzz, par_mid.zzzz );
	const float8 bbox_max_x = (float8)( par_mid.x, par_max.x, par_mid.x, par_max.x, par_mid.x, par_max.x, par_mid.x, par_max.x );
	const float8 bbox_max_y = (float8)( par_mid.yy, par_max.yy, par_mid.yy, par_max.yy );
	const float8 bbox_max_z = (float8)( par_mid.zzzz, par_max.zzzz );

	child_bbox[0] = (struct BBox){ (float3)( bbox_min_x.s0, bbox_min_y.s0, bbox_min_z.s0 ), (float3)( bbox_max_x.s0, bbox_max_y.s0, bbox_max_z.s0 ) };
	child_bbox[1] = (struct BBox){ (float3)( bbox_min_x.s1, bbox_min_y.s1, bbox_min_z.s1 ), (float3)( bbox_max_x.s1, bbox_max_y.s1, bbox_max_z.s1 ) };
	child_bbox[2] = (struct BBox){ (float3)( bbox_min_x.s2, bbox_min_y.s2, bbox_min_z.s2 ), (float3)( bbox_max_x.s2, bbox_max_y.s2, bbox_max_z.s2 ) };
	child_bbox[3] = (struct BBox){ (float3)( bbox_min_x.s3, bbox_min_y.s3, bbox_min_z.s3 ), (float3)( bbox_max_x.s3, bbox_max_y.s3, bbox_max_z.s3 ) };
	child_bbox[4] = (struct BBox){ (float3)( bbox_min_x.s4, bbox_min_y.s4, bbox_min_z.s4 ), (float3)( bbox_max_x.s4, bbox_max_y.s4, bbox_max_z.s4 ) };
	child_bbox[5] = (struct BBox){ (float3)( bbox_min_x.s5, bbox_min_y.s5, bbox_min_z.s5 ), (float3)( bbox_max_x.s5, bbox_max_y.s5, bbox_max_z.s5 ) };
	child_bbox[6] = (struct BBox){ (float3)( bbox_min_x.s6, bbox_min_y.s6, bbox_min_z.s6 ), (float3)( bbox_max_x.s6, bbox_max_y.s6, bbox_max_z.s6 ) };
	child_bbox[7] = (struct BBox){ (float3)( bbox_min_x.s7, bbox_min_y.s7, bbox_min_z.s7 ), (float3)( bbox_max_x.s7, bbox_max_y.s7, bbox_max_z.s7 ) };

	float8 t;
	int8 r;
	intersect8(bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z, ray, &t, &r);
#if OCL_QUIRK_0003
	const int8 occupancy = as_int8(((ushort8)(-1) != octet.child).s0011223344556677);
#else
	const int8 occupancy = convert_int8((ushort8)(-1) != octet.child);
#endif
	r &= occupancy;

#if OCL_QUIRK_0004
	const int8 cnt0 = -r;
	const int4 cnt1 = cnt0.s0123 + cnt0.s4567;
	const int2 cnt2 = cnt1.s01   + cnt1.s23;
	const int count = cnt2.s0    + cnt2.s1;

#else
	int count = 0;
	count -= r.s0;
	count -= r.s1;
	count -= r.s2;
	count -= r.s3;
	count -= r.s4;
	count -= r.s5;
	count -= r.s6;
	count -= r.s7;

#endif
	t = select((float8)(INFINITY), t, r);

	const float4 r0_A = (float4)(t.s0, t.s3, t.s4, t.s7);
	const float4 r0_B = (float4)(t.s1, t.s2, t.s5, t.s6);
	const int4 r0x_A = (int4)(0, 3, 4, 7);
	const int4 r0x_B = (int4)(1, 2, 5, 6);
	const int4 m0 = islessequal(r0_A, r0_B);
	const float4 r0_min = fmin(r0_A, r0_B);
	const float4 r0_max = fmax(r0_A, r0_B);
	const int4 r0x_min = select(r0x_B, r0x_A, m0);
	const int4 r0x_max = select(r0x_A, r0x_B, m0);

	const float4 r1_A = (float4)(r0_min.s0, r0_max.s0, r0_max.s3, r0_min.s3);
	const float4 r1_B = (float4)(r0_max.s1, r0_min.s1, r0_min.s2, r0_max.s2);
	const int4 r1x_A = (int4)(r0x_min.s0, r0x_max.s0, r0x_max.s3, r0x_min.s3);
	const int4 r1x_B = (int4)(r0x_max.s1, r0x_min.s1, r0x_min.s2, r0x_max.s2);
	const int4 m1 = islessequal(r1_A, r1_B);
	const float4 r1_min = fmin(r1_A, r1_B);
	const float4 r1_max = fmax(r1_A, r1_B);
	const int4 r1x_min = select(r1x_B, r1x_A, m1);
	const int4 r1x_max = select(r1x_A, r1x_B, m1);

	const float4 r2_A = (float4)(r1_min.s0, r1_max.s0, r1_max.s3, r1_min.s3);
	const float4 r2_B = (float4)(r1_min.s1, r1_max.s1, r1_max.s2, r1_min.s2);
	const int4 r2x_A = (int4)(r1x_min.s0, r1x_max.s0, r1x_max.s3, r1x_min.s3);
	const int4 r2x_B = (int4)(r1x_min.s1, r1x_max.s1, r1x_max.s2, r1x_min.s2);
	const int4 m2 = islessequal(r2_A, r2_B);
	const float4 r2_min = fmin(r2_A, r2_B);
	const float4 r2_max = fmax(r2_A, r2_B);
	const int4 r2x_min = select(r2x_B, r2x_A, m2);
	const int4 r2x_max = select(r2x_A, r2x_B, m2);

	const float4 r3_A = (float4)(r2_min.s0, r2_max.s0, r2_min.s1, r2_max.s1);
	const float4 r3_B = (float4)(r2_max.s2, r2_min.s2, r2_max.s3, r2_min.s3);
	const int4 r3x_A = (int4)(r2x_min.s0, r2x_max.s0, r2x_min.s1, r2x_max.s1);
	const int4 r3x_B = (int4)(r2x_max.s2, r2x_min.s2, r2x_max.s3, r2x_min.s3);
	const int4 m3 = islessequal(r3_A, r3_B);
	const float4 r3_min = fmin(r3_A, r3_B);
	const float4 r3_max = fmax(r3_A, r3_B);
	const int4 r3x_min = select(r3x_B, r3x_A, m3);
	const int4 r3x_max = select(r3x_A, r3x_B, m3);

	const float4 r4_A = (float4)(r3_min.s0, r3_min.s1, r3_max.s0, r3_max.s1);
	const float4 r4_B = (float4)(r3_min.s2, r3_min.s3, r3_max.s2, r3_max.s3);
	const int4 r4x_A = (int4)(r3x_min.s0, r3x_min.s1, r3x_max.s0, r3x_max.s1);
	const int4 r4x_B = (int4)(r3x_min.s2, r3x_min.s3, r3x_max.s2, r3x_max.s3);
	const int4 m4 = islessequal(r4_A, r4_B);
	const float4 r4_min = fmin(r4_A, r4_B);
	const float4 r4_max = fmax(r4_A, r4_B);
	const int4 r4x_min = select(r4x_B, r4x_A, m4);
	const int4 r4x_max = select(r4x_A, r4x_B, m4);

	const float4 r5_A = (float4)(r4_min.s0, r4_max.s0, r4_min.s2, r4_max.s2);
	const float4 r5_B = (float4)(r4_min.s1, r4_max.s1, r4_min.s3, r4_max.s3);
	const int4 r5x_A = (int4)(r4x_min.s0, r4x_max.s0, r4x_min.s2, r4x_max.s2);
	const int4 r5x_B = (int4)(r4x_min.s1, r4x_max.s1, r4x_min.s3, r4x_max.s3);
	const int4 m5 = islessequal(r5_A, r5_B);
	const float4 r5_min = fmin(r5_A, r5_B);
	const float4 r5_max = fmax(r5_A, r5_B);
	const int4 r5x_min = select(r5x_B, r5x_A, m5);
	const int4 r5x_max = select(r5x_A, r5x_B, m5);

	child_index->distance = (float8)(r5_min.s0, r5_max.s0, r5_min.s1, r5_max.s1, r5_min.s2, r5_max.s2, r5_min.s3, r5_max.s3);
	child_index->index = (uint8)(r5x_min.s0, r5x_max.s0, r5x_min.s1, r5x_max.s1, r5x_min.s2, r5x_max.s2, r5x_min.s3, r5x_max.s3);
	return as_uint(count);
}

