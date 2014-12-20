#ifndef prim_map_H__
#define prim_map_H__

namespace testbed
{

namespace rayv
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

} // namespace rayv
} // namespace testbed

#endif // prim_map_H__
