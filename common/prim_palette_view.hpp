#ifndef prim_palette_view_H__
#define prim_palette_view_H__

#include "vectsimd.hpp"

namespace testbed
{

namespace pltv
{

bool
deinit_resources();

bool
init_resources(
	const size_t w,
	const size_t h);

bool
render(
	const simd::vect4 (& palette)[4],
	const void* pixmap);

} // namespace pltv
} // namespace testbed

#endif // prim_palette_view_H__
