#ifndef prim_map_H__
#define prim_map_H__

namespace testbed
{

namespace rgbv
{

bool
deinit_resources();

bool
init_resources(
	const size_t w,
	const size_t h);

bool
render(
	const void* pixmap);

bool
render(
	const void* pixmap,
	const float contrast_middle,
	const float contrast_k,
	const float blur_split);

} // namespace rgbv
} // namespace testbed

#endif // prim_map_H__
