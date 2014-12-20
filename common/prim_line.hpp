#ifndef prim_line_H__
#define prim_line_H__

#include "vectsimd.hpp"

namespace testbed {
namespace line {

bool
deinit_resources();

bool
init_resources();

bool
render(
	const simd::matx4& mv,
	const simd::vect3 (& end)[2],
	const simd::vect4& color);

} // namespace line
} // namespace testbed

#endif // prim_line_H__
