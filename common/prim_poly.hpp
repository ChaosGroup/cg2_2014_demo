#ifndef prim_poly_H__
#define prim_poly_H__

#include "vectsimd.hpp"

namespace testbed
{
namespace poly
{

bool
deinit_resources();

bool
init_resources(
	const unsigned count);

bool
init_resources(
	const unsigned start,
	const unsigned count,
	const void* const vert);

bool
render(
	const simd::matx4& mvp,
	const simd::vect4& color,
	const unsigned start,
	const unsigned count,
	const bool striped);

} // namespace poly
} // namespace testbed

#endif // prim_poly_H__
