#ifndef prim_sphere_H__
#define prim_sphere_H__

#include "vectsimd.hpp"

namespace testbed {
namespace sphe {

bool
deinit_resources();

bool
init_resources();

bool
render(
	const simd::matx4& mvp,
	const simd::vect4 (& clp)[2],
	const simd::vect4& color);

} // namespace sphe
} // namespace testbed

#endif // prim_sphere_H__
