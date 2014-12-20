#ifndef isfinite_H__
#define isfinite_H__

#include "cmath_fix"
#include <limits>

#if __FINITE_MATH_ONLY__ == 1 && __clang__ == 0
inline bool
isfinite(
	const float f)
{
	return fabs(f) != std::numeric_limits< float >::infinity();
	// one comparison to infinity covers both infinity and NaN cases. under
	// normal conditions a NaN equals nothing, but under 'finite math only'
}	// a NaN equals everything, including infinity (g++ 4.4.x - 4.6.x)

#else
using std::isfinite;

#endif

#endif // isfinite_H__
