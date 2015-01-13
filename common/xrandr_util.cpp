#include <X11/Xlib.h>
#include <X11/extensions/Xrandr.h>
#include "xrandr_util.hpp"
#include "stream.hpp"

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
		stream::cerr << "XRANDR not found\n";
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
		stream::cerr << "failed to obtain screen configs through XRANDR\n";
		return false;
	}

	XRRConfigCurrentConfiguration(config, &current_rotation);

	stream::cout << "XRR screen config 0: " <<
		sizes[0].width << " x " << sizes[0].height << '\n';

	int mode = 0;

	for (int i = 1; i < nsizes; ++i)
	{
		stream::cout << "XRR screen config " << i << ": " <<
			sizes[i].width << " x " << sizes[i].height << '\n';

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
		stream::cerr << "failed to find a suitable screen config through XRANDR\n";
		return false;
	}

	stream::cout << "chosen XRR screen config: " << mode << "\n";

	XRRSetScreenConfig(display, config, root, mode, current_rotation, CurrentTime);
	XRRFreeScreenConfigInfo(config);

	return true;
}

} // namespace util
