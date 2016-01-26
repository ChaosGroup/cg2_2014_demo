#ifndef prim_mesh_H__
#define prim_mesh_H__

#include "vectsimd_sse.hpp"

namespace testbed {
namespace mesh {

bool
deinit_resources();

bool
init_resources(
	const size_t countof_vert,
	const void* const vert,
	const size_t countof_face,
	const void* const face);

bool
render(
	const simd::matx4& mv,
	const simd::matx4& proj,
	const simd::vect3 (& clp)[2],
	const simd::vect3 (& wld)[2],
	const simd::vect4& color);

} // namespace mesh
} // namespace testbed

#endif // prim_mesh_H__

