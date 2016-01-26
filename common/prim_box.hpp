#ifndef prim_box_H__
#define prim_box_H__

#include "vectsimd_sse.hpp"

namespace testbed {
namespace box {

bool
deinit_resources();

bool
init_resources();

bool
render(
	const simd::matx4& mvp,
	const simd::vect3 (& extremum)[2],
	const simd::vect4& color);

} // namespace box
} // namespace testbed

#endif // prim_box_H__
