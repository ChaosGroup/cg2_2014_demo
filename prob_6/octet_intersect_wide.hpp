#ifndef octet_intersect_wide_H__
#define octet_intersect_wide_H__

#if !defined(prob_4_H__) && !defined(prob_7_H__)
	#error prob_4_H__ or prob_7_H__ required
#endif

inline size_t
octet_intersect_wide(
	const Octet& octet,
	const BBox& bbox,
	const Ray& ray,
	ChildIndex& child_index,
	BBox (& child_bbox)[8])
{
	assert(bbox.is_valid());

	const __m128 par_min = bbox.get_min();
	const __m128 par_max = bbox.get_max();
	const __m128 par_mid = _mm_mul_ps(
		_mm_add_ps(par_min, par_max),
		_mm_set1_ps(.5f));

#if __AVX__ != 0
	const __m256 par_min_x = _mm256_set1_ps(bbox.get_min()[0]);
	const __m256 par_min_y = _mm256_set1_ps(bbox.get_min()[1]);
	const __m256 par_min_z = _mm256_set1_ps(bbox.get_min()[2]);
	const __m256 par_max_x = _mm256_set1_ps(bbox.get_max()[0]);
	const __m256 par_max_y = _mm256_set1_ps(bbox.get_max()[1]);
	const __m256 par_max_z = _mm256_set1_ps(bbox.get_max()[2]);
	const __m256 par_mid_x = _mm256_set1_ps(par_mid[0]);
	const __m256 par_mid_y = _mm256_set1_ps(par_mid[1]);
	const __m256 par_mid_z = _mm256_set1_ps(par_mid[2]);

#else
	const __m128 par_min_x = _mm_shuffle_ps(par_min, par_min, 0);
	const __m128 par_min_y = _mm_shuffle_ps(par_min, par_min, 0x55);
	const __m128 par_min_z = _mm_shuffle_ps(par_min, par_min, 0xaa);
	const __m128 par_max_x = _mm_shuffle_ps(par_max, par_max, 0);
	const __m128 par_max_y = _mm_shuffle_ps(par_max, par_max, 0x55);
	const __m128 par_max_z = _mm_shuffle_ps(par_max, par_max, 0xaa);
	const __m128 par_mid_x = _mm_shuffle_ps(par_mid, par_mid, 0);
	const __m128 par_mid_y = _mm_shuffle_ps(par_mid, par_mid, 0x55);
	const __m128 par_mid_z = _mm_shuffle_ps(par_mid, par_mid, 0xaa);

#endif

#if __AVX__ != 0
	const __m256 bbox_min_x = _mm256_unpacklo_ps(par_min_x, par_mid_x);
#else
	const __m128 bbox_min_x[] = {
		_mm_unpacklo_ps(par_min_x, par_mid_x),
		_mm_unpacklo_ps(par_min_x, par_mid_x)
	};
#endif

#if __AVX__ != 0
	const __m256 bbox_min_y = _mm256_shuffle_ps(par_min_y, par_mid_y, 0);
#else
	const __m128 bbox_min_y[] = {
		_mm_shuffle_ps(par_min_y, par_mid_y, 0),
		_mm_shuffle_ps(par_min_y, par_mid_y, 0)
	};
#endif

#if __AVX__ != 0
	const __m256 bbox_min_z = _mm256_insertf128_ps(par_min_z, _mm256_castps256_ps128(par_mid_z), 1);
#else
	const __m128 bbox_min_z[] = {
		par_min_z,
		par_mid_z
	};
#endif

#if __AVX__ != 0
	const __m256 bbox_max_x = _mm256_unpacklo_ps(par_mid_x, par_max_x);
#else
	const __m128 bbox_max_x[] = {
		_mm_unpacklo_ps(par_mid_x, par_max_x),
		_mm_unpacklo_ps(par_mid_x, par_max_x)
	};
#endif

#if __AVX__ != 0
	const __m256 bbox_max_y = _mm256_shuffle_ps(par_mid_y, par_max_y, 0);
#else
	const __m128 bbox_max_y[] = {
		_mm_shuffle_ps(par_mid_y, par_max_y, 0),
		_mm_shuffle_ps(par_mid_y, par_max_y, 0)
	};
#endif

#if __AVX__ != 0
	const __m256 bbox_max_z = _mm256_insertf128_ps(par_mid_z, _mm256_castps256_ps128(par_max_z), 1);
#else
	const __m128 bbox_max_z[] = {
		par_mid_z,
		par_max_z
	};
#endif

	// to this day Intel don't have an adequate non-const permute op; split loop for wide-access shuffles
#if __AVX__ != 0
	__m128 running_min_x = _mm256_castps256_ps128(bbox_min_x);
	__m128 running_min_y = _mm256_castps256_ps128(bbox_min_y);
	__m128 running_min_z = _mm256_castps256_ps128(bbox_min_z);
	__m128 running_max_x = _mm256_castps256_ps128(bbox_max_x);
	__m128 running_max_y = _mm256_castps256_ps128(bbox_max_y);
	__m128 running_max_z = _mm256_castps256_ps128(bbox_max_z);

#else
	__m128 running_min_x = bbox_min_x[0];
	__m128 running_min_y = bbox_min_y[0];
	__m128 running_min_z = bbox_min_z[0];
	__m128 running_max_x = bbox_max_x[0];
	__m128 running_max_y = bbox_max_y[0];
	__m128 running_max_z = bbox_max_z[0];

#endif
	for (size_t i = 0; i < 4; ++i)
	{
		const __m128 min = _mm_movelh_ps(_mm_unpacklo_ps(running_min_x, running_min_y), running_min_z);
		const __m128 max = _mm_movelh_ps(_mm_unpacklo_ps(running_max_x, running_max_y), running_max_z);

		child_bbox[i] = BBox(min, max, BBox::flag_direct());

		running_min_x = _mm_shuffle_ps(running_min_x, running_min_x, 0x39);
		running_min_y = _mm_shuffle_ps(running_min_y, running_min_y, 0x39);
		running_min_z = _mm_shuffle_ps(running_min_z, running_min_z, 0x39);
		running_max_x = _mm_shuffle_ps(running_max_x, running_max_x, 0x39);
		running_max_y = _mm_shuffle_ps(running_max_y, running_max_y, 0x39);
		running_max_z = _mm_shuffle_ps(running_max_z, running_max_z, 0x39);
	}

#if __AVX__ != 0
	running_min_x = _mm256_castps256_ps128(_mm256_permute2f128_ps(bbox_min_x, bbox_min_x, 0x11));
	running_min_y = _mm256_castps256_ps128(_mm256_permute2f128_ps(bbox_min_y, bbox_min_y, 0x11));
	running_min_z = _mm256_castps256_ps128(_mm256_permute2f128_ps(bbox_min_z, bbox_min_z, 0x11));
	running_max_x = _mm256_castps256_ps128(_mm256_permute2f128_ps(bbox_max_x, bbox_max_x, 0x11));
	running_max_y = _mm256_castps256_ps128(_mm256_permute2f128_ps(bbox_max_y, bbox_max_y, 0x11));
	running_max_z = _mm256_castps256_ps128(_mm256_permute2f128_ps(bbox_max_z, bbox_max_z, 0x11));

#else
	running_min_x = bbox_min_x[1];
	running_min_y = bbox_min_y[1];
	running_min_z = bbox_min_z[1];
	running_max_x = bbox_max_x[1];
	running_max_y = bbox_max_y[1];
	running_max_z = bbox_max_z[1];

#endif
	for (size_t i = 4; i < 8; ++i)
	{
		const __m128 min = _mm_movelh_ps(_mm_unpacklo_ps(running_min_x, running_min_y), running_min_z);
		const __m128 max = _mm_movelh_ps(_mm_unpacklo_ps(running_max_x, running_max_y), running_max_z);

		child_bbox[i] = BBox(min, max, BBox::flag_direct());

		running_min_x = _mm_shuffle_ps(running_min_x, running_min_x, 0x39);
		running_min_y = _mm_shuffle_ps(running_min_y, running_min_y, 0x39);
		running_min_z = _mm_shuffle_ps(running_min_z, running_min_z, 0x39);
		running_max_x = _mm_shuffle_ps(running_max_x, running_max_x, 0x39);
		running_max_y = _mm_shuffle_ps(running_max_y, running_max_y, 0x39);
		running_max_z = _mm_shuffle_ps(running_max_z, running_max_z, 0x39);
	}

	// compute intersection distances (use distance-to-exit)

#if __AVX__ != 0
	float t[8] __attribute__ ((aligned(sizeof(__m256))));
	uint32_t r[8] __attribute__ ((aligned(sizeof(__m256))));

	intersect8(
		bbox_min_x,
		bbox_min_y,
		bbox_min_z,
		bbox_max_x,
		bbox_max_y,
		bbox_max_z,
		ray, t, r);

#else
	float t[8] __attribute__ ((aligned(sizeof(__m128))));
	uint32_t r[8] __attribute__ ((aligned(sizeof(__m128))));

	intersect8(
		bbox_min_x,
		bbox_min_y,
		bbox_min_z,
		bbox_max_x,
		bbox_max_y,
		bbox_max_z,
		ray, t, r);

#endif
	// filter out empty nodes
	const __m128i empty = octet.get_occupancy();
	const __m128i empty0 = _mm_unpacklo_epi16(empty, empty);
	const __m128i empty1 = _mm_unpackhi_epi16(empty, empty);

#if __AVX__ != 0
	*(__m256*) r = _mm256_andnot_ps(
		_mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(empty0), empty1, 1)),
		*(__m256*) r);

#else
	*(__m128i*) (r + 0) = _mm_andnot_si128(empty0, *(__m128i*) (r + 0));
	*(__m128i*) (r + 4) = _mm_andnot_si128(empty1, *(__m128i*) (r + 4));

#endif
	// count the non-empty, intersected nodes
	uint32_t count = 0;

	count -= r[0];
	count -= r[1];
	count -= r[2];
	count -= r[3];
	count -= r[4];
	count -= r[5];
	count -= r[6];
	count -= r[7];

#if __AVX__ != 0
	*(__m256*) t = _mm256_blendv_ps(
		_mm256_set1_ps(std::numeric_limits< float >::infinity()),
		*(__m256*) t,
		*(__m256*) r);

#else
	*(__m128*) (t + 0) = _nn_blendv_ps(
		_mm_set1_ps(std::numeric_limits< float >::infinity()),
		*(__m128*) (t + 0),
		*(__m128*) (r + 0));

	*(__m128*) (t + 4) = _nn_blendv_ps(
		_mm_set1_ps(std::numeric_limits< float >::infinity()),
		*(__m128*) (t + 4),
		*(__m128*) (r + 4));

#endif
	// init child indices
	child_index.index[0] = 0;
	child_index.index[1] = 1;
	child_index.index[2] = 2;
	child_index.index[3] = 3;
	child_index.index[4] = 4;
	child_index.index[5] = 5;
	child_index.index[6] = 6;
	child_index.index[7] = 7;

	// sort intersected nodes in ascending order; for the purpose
	// use Ken Batcher's bitonic sorting network for 8 elements

	// Ken Batcher's bitonic sorting network
	// (0, 1), (3, 2), (4, 5), (7, 6)    : stage0
	// (0, 2), (1, 3), (6, 4), (7, 5)    : stage1
	// (0, 1), (2, 3), (5, 4), (7, 6)    : stage2
	// (0, 4), (1, 5), (2, 6), (3, 7)    : stage3
	// (0, 2), (1, 3), (4, 6), (5, 7)    : stage4
	// (0, 1), (2, 3), (4, 5), (6, 7)    : stage5

	const __m128 r0_in0 = *(__m128*)(t + 0); // 0, 1, 2, 3
	const __m128 r0_in1 = *(__m128*)(t + 4); // 4, 5, 6, 7

	const __m128 r0x_in0 = *(__m128*)(child_index.index + 0);
	const __m128 r0x_in1 = *(__m128*)(child_index.index + 4);

	// stage 0
	const __m128 r0_A = _mm_shuffle_ps(r0_in0, r0_in1, 0xcc); // 0, 3, 4, 7
	const __m128 r0_B = _mm_shuffle_ps(r0_in0, r0_in1, 0x99); // 1, 2, 5, 6

	const __m128 r0x_A = _mm_shuffle_ps(r0x_in0, r0x_in1, 0xcc);
	const __m128 r0x_B = _mm_shuffle_ps(r0x_in0, r0x_in1, 0x99);

	const __m128 m0 = _mm_cmple_ps(r0_A, r0_B);

	const __m128 r0_min = _mm_min_ps(r0_A, r0_B); // 0, 3, 4, 7
	const __m128 r0_max = _mm_max_ps(r0_A, r0_B); // 1, 2, 5, 6

	const __m128 r0x_min = _nn_blendv_ps(r0x_B, r0x_A, m0);
	const __m128 r0x_max = _nn_blendv_ps(r0x_A, r0x_B, m0);

	// stage 1
	__m128 r1_A = _mm_shuffle_ps(r0_max, r0_max, 0xf0); // 1, 1, 6, 6
	__m128 r1_B = _mm_shuffle_ps(r0_max, r0_max, 0xa5); // 2, 2, 5, 5
	       r1_A = _nn_blend_ps(r1_A, r0_min, 0x9);      // 0, 1, 6, 7
	       r1_B = _nn_blend_ps(r1_B, r0_min, 0x6);      // 2, 3, 4, 5

	__m128 r1x_A = _mm_shuffle_ps(r0x_max, r0x_max, 0xf0);
	__m128 r1x_B = _mm_shuffle_ps(r0x_max, r0x_max, 0xa5);
	       r1x_A = _nn_blend_ps(r1x_A, r0x_min, 0x9);
	       r1x_B = _nn_blend_ps(r1x_B, r0x_min, 0x6);

	const __m128 m1 = _mm_cmple_ps(r1_A, r1_B);

	const __m128 r1_min = _mm_min_ps(r1_A, r1_B); // 0, 1, 6, 7
	const __m128 r1_max = _mm_max_ps(r1_A, r1_B); // 2, 3, 4, 5

	const __m128 r1x_min = _nn_blendv_ps(r1x_B, r1x_A, m1);
	const __m128 r1x_max = _nn_blendv_ps(r1x_A, r1x_B, m1);

	// stage 2
	__m128 r2_A = _mm_shuffle_ps(r1_max, r1_max, 0xf0); // 2, 2, 5, 5
	__m128 r2_B = _mm_shuffle_ps(r1_min, r1_min, 0xa5); // 1, 1, 6, 6
	       r2_A = _nn_blend_ps(r2_A, r1_min, 0x9);      // 0, 2, 5, 7
	       r2_B = _nn_blend_ps(r2_B, r1_max, 0x6);      // 1, 3, 4, 6

	__m128 r2x_A = _mm_shuffle_ps(r1x_max, r1x_max, 0xf0);
	__m128 r2x_B = _mm_shuffle_ps(r1x_min, r1x_min, 0xa5);
	       r2x_A = _nn_blend_ps(r2x_A, r1x_min, 0x9);
	       r2x_B = _nn_blend_ps(r2x_B, r1x_max, 0x6);

	const __m128 m2 = _mm_cmple_ps(r2_A, r2_B);

	const __m128 r2_min = _mm_min_ps(r2_A, r2_B); // 0, 2, 5, 7
	const __m128 r2_max = _mm_max_ps(r2_A, r2_B); // 1, 3, 4, 6

	const __m128 r2x_min = _nn_blendv_ps(r2x_B, r2x_A, m2);
	const __m128 r2x_max = _nn_blendv_ps(r2x_A, r2x_B, m2);

	// stage 3
	const __m128 r3_A = _mm_unpacklo_ps(r2_min, r2_max); // 0, 1, 2, 3
	const __m128 r3_B = _mm_unpackhi_ps(r2_max, r2_min); // 4, 5, 6, 7

	const __m128 r3x_A = _mm_unpacklo_ps(r2x_min, r2x_max);
	const __m128 r3x_B = _mm_unpackhi_ps(r2x_max, r2x_min);

	const __m128 m3 = _mm_cmple_ps(r3_A, r3_B);

	const __m128 r3_min = _mm_min_ps(r3_A, r3_B); // 0, 1, 2, 3
	const __m128 r3_max = _mm_max_ps(r3_A, r3_B); // 4, 5, 6, 7

	const __m128 r3x_min = _nn_blendv_ps(r3x_B, r3x_A, m3);
	const __m128 r3x_max = _nn_blendv_ps(r3x_A, r3x_B, m3);

	// stage 4
	const __m128 r4_A = _mm_movelh_ps(r3_min, r3_max); // 0, 1, 4, 5
	const __m128 r4_B = _mm_movehl_ps(r3_max, r3_min); // 2, 3, 6, 7

	const __m128 r4x_A = _mm_movelh_ps(r3x_min, r3x_max);
	const __m128 r4x_B = _mm_movehl_ps(r3x_max, r3x_min);

	const __m128 m4 = _mm_cmple_ps(r4_A, r4_B);

	const __m128 r4_min = _mm_min_ps(r4_A, r4_B); // 0, 1, 4, 5
	const __m128 r4_max = _mm_max_ps(r4_A, r4_B); // 2, 3, 6, 7

	const __m128 r4x_min = _nn_blendv_ps(r4x_B, r4x_A, m4);
	const __m128 r4x_max = _nn_blendv_ps(r4x_A, r4x_B, m4);

	// stage 5
	const __m128 r5_a = _mm_unpacklo_ps(r4_min, r4_max); // 0, 2, 1, 3
	const __m128 r5_b = _mm_unpackhi_ps(r4_min, r4_max); // 4, 6, 5, 7
	const __m128 r5_A = _mm_movelh_ps(r5_a, r5_b); // 0, 2, 4, 6
	const __m128 r5_B = _mm_movehl_ps(r5_b, r5_a); // 1, 3, 5, 7

	const __m128 r5x_a = _mm_unpacklo_ps(r4x_min, r4x_max);
	const __m128 r5x_b = _mm_unpackhi_ps(r4x_min, r4x_max);
	const __m128 r5x_A = _mm_movelh_ps(r5x_a, r5x_b);
	const __m128 r5x_B = _mm_movehl_ps(r5x_b, r5x_a);

	const __m128 m5 = _mm_cmple_ps(r5_A, r5_B);

	const __m128 r5_min = _mm_min_ps(r5_A, r5_B); // 0, 2, 4, 6
	const __m128 r5_max = _mm_max_ps(r5_A, r5_B); // 1, 3, 5, 7

	const __m128 r5x_min = _nn_blendv_ps(r5x_B, r5x_A, m5);
	const __m128 r5x_max = _nn_blendv_ps(r5x_A, r5x_B, m5);

	// output
	*(__m128*)(child_index.distance + 0) = _mm_unpacklo_ps(r5_min, r5_max); // 0, 1, 2, 3
	*(__m128*)(child_index.distance + 4) = _mm_unpackhi_ps(r5_min, r5_max); // 4, 5, 6, 7

	*(__m128*)(child_index.index + 0) = _mm_unpacklo_ps(r5x_min, r5x_max);
	*(__m128*)(child_index.index + 4) = _mm_unpackhi_ps(r5x_min, r5x_max);

	return count;
}

#endif // octet_intersect_wide_H__
