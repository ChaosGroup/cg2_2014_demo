#include <iostream>
#include <X11/Xlib.h>
#include <X11/extensions/Xrandr.h>
#include "xrandr_util.hpp"

namespace util
{

bool
set_screen_config(
	Display* display,
	const Window root,
	const int width,
	const int height)
{
	int dummy;

	if (!XQueryExtension(display, "RANDR", &dummy, &dummy, &dummy))
	{
		std::cerr << "XRANDR not found" << std::endl;
		return false;
	}

	XRRScreenConfiguration *config = 0;
	XRRScreenSize *sizes = 0;
	Rotation current_rotation;
	int nsizes = 0;

	config = XRRGetScreenInfo(display, root);
	sizes = XRRConfigSizes(config, &nsizes);

	if (0 == config || 0 == nsizes)
	{
		std::cerr << "failed to obtain screen configs through XRANDR" << std::endl;
		return false;
	}

	XRRConfigCurrentConfiguration(config, &current_rotation);

	std::cout << "XRR screen config 0: " <<
		sizes[0].width << " x " << sizes[0].height << std::endl;

	int mode = 0;

	for (int i = 1; i < nsizes; ++i)
	{
		std::cout << "XRR screen config " << i << ": " <<
			sizes[i].width << " x " << sizes[i].height << std::endl;

		if (sizes[i].width >= width &&
			sizes[i].height >= height &&
			(sizes[mode].width < width || sizes[mode].width > sizes[i].width ||
			 sizes[mode].height < height || sizes[mode].height > sizes[i].height))
		{
			mode = i;
		}
	}

	if (sizes[mode].width < width || sizes[mode].height < height)
	{
		std::cerr << "failed to find a suitable screen config through XRANDR" << std::endl;
		return false;
	}

	std::cout << "chosen XRR screen config: " << mode << std::endl;

	XRRSetScreenConfig(display, config, root, mode, current_rotation, CurrentTime);
	XRRFreeScreenConfigInfo(config);

	return true;
}

} // namespace util
