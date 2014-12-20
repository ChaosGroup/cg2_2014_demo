#ifndef xrandr_util_H__
#define xrandr_util_H__

#include <X11/Xlib.h>

namespace util
{

bool
set_screen_config(
	Display* display,
	const Window root,
	const int width,
	const int height);

} // namespace util

#endif // xrandr_util_H__
