#ifndef prim_mono_view_H__
#define prim_mono_view_H__

namespace testbed {
namespace monv {

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
	const GLuint tex_name);

} // namespace monv
} // namespace testbed

#endif // prim_mono_view_H__
