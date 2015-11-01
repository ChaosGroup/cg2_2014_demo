#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#if GLX_ARB_create_context == 0
#error Missing required extension GLX_ARB_create_context.
#endif 

#include <assert.h>
#include <string.h>

#ifdef USE_XRANDR
#include "xrandr_util.hpp"
#endif

#include "testbed.hpp"
#include "stream.hpp"


static Display* display;
static Window window;
static GLXContext context;

static Atom wm_protocols;
static Atom wm_delete_window;


static size_t bitcount(
	size_t n)
{
	size_t count = 0;

	for (; 0 != n; ++count)
		n &= n - 1;

	return count;
}


namespace testbed
{

int
initGL(
	const unsigned w,
	const unsigned h,
	unsigned (&bitness)[4],
	const unsigned zbitness,
	const unsigned fsaa)
{
	assert(0 == display);

	display = XOpenDisplay(0);

	if (0 == display)
	{
		stream::cerr << "failed to open display\n";
		return 1;
	}

	if (0 == bitness[0] + bitness[1] + bitness[2] + bitness[3])
	{
		const int screen = XDefaultScreen(display);

		XVisualInfo vinfo_template;
		memset(&vinfo_template, 0, sizeof(vinfo_template));

		vinfo_template.visualid = XVisualIDFromVisual(XDefaultVisual(display, screen));
		vinfo_template.screen = screen;

		const long vinfo_mask = VisualIDMask | VisualScreenMask;

		int vinfo_count = 0;
		XVisualInfo* const vinfo = XGetVisualInfo(display, vinfo_mask, &vinfo_template, &vinfo_count);

		if (0 == vinfo || 0 == vinfo_count)
		{
			stream::cerr << "failed to retrieve visual info\n";
			return 1;
		}

		bitness[0] = bitcount(vinfo->red_mask);
		bitness[1] = bitcount(vinfo->green_mask);
		bitness[2] = bitcount(vinfo->blue_mask);
		bitness[3] = 0;

		XFree(vinfo);

		stream::cout << "using screen RGB bitness: " <<
			bitness[0] << ' ' <<
			bitness[1] << ' ' <<
			bitness[2] << '\n';
	}

	const Window root = XDefaultRootWindow(display);

#ifdef USE_XRANDR
	util::set_screen_config(display, root, w, h);
#endif

	const char* const extensions = glXQueryExtensionsString(display, DefaultScreen(display));
	stream::cout << extensions << '\n';

	int visual_attr[64];
	unsigned na = 0;

	visual_attr[na++] = GLX_X_RENDERABLE;
	visual_attr[na++] = True;
	visual_attr[na++] = GLX_DRAWABLE_TYPE;
	visual_attr[na++] = GLX_WINDOW_BIT;
	visual_attr[na++] = GLX_RENDER_TYPE;
	visual_attr[na++] = GLX_RGBA_BIT;
	visual_attr[na++] = GLX_X_VISUAL_TYPE;
	visual_attr[na++] = GLX_TRUE_COLOR;
	visual_attr[na++] = GLX_RED_SIZE;
	visual_attr[na++] = bitness[0];
	visual_attr[na++] = GLX_GREEN_SIZE;
	visual_attr[na++] = bitness[1];
	visual_attr[na++] = GLX_BLUE_SIZE;
	visual_attr[na++] = bitness[2];
	visual_attr[na++] = GLX_ALPHA_SIZE;
	visual_attr[na++] = bitness[3];
	visual_attr[na++] = GLX_DOUBLEBUFFER;
	visual_attr[na++] = True;

	if (-1U != zbitness)
	{
		visual_attr[na++] = GLX_DEPTH_SIZE;
		visual_attr[na++] = 0 == zbitness ? bitness[0] + bitness[1] + bitness[2] : zbitness;
	}

	if (fsaa)
	{
		visual_attr[na++] = GLX_SAMPLE_BUFFERS;
		visual_attr[na++] = 1;
		visual_attr[na++] = GLX_SAMPLES;
		visual_attr[na++] = fsaa;
	}

	assert(na < sizeof(visual_attr) / sizeof(visual_attr[0]));

	visual_attr[na] = None;

	int fbc_count = 0;
	const GLXFBConfig* const fbc = glXChooseFBConfig(display, DefaultScreen(display), visual_attr, &fbc_count);

	if (0 == fbc || 0 == fbc_count)
	{
		stream::cerr << "failed to retrieve framebuffer config\n";
		return 1;
	}

	const XVisualInfo* const vi = glXGetVisualFromFBConfig(display, fbc[0]);

	XSetWindowAttributes swa;
	swa.colormap = XCreateColormap(display, root, vi->visual, AllocNone);
	swa.border_pixel = 0;
	swa.event_mask = StructureNotifyMask;

	const unsigned border_width = 0;
	const int x = 0, y = 0;

	window = XCreateWindow(display, root, x, y, w, h, border_width, vi->depth,
		InputOutput, vi->visual, CWBorderPixel | CWColormap | CWEventMask, &swa);

	if (0 == window)
	{
		stream::cerr << "failed to create window\n";
		return 1;
	}

	wm_protocols = XInternAtom(display, "WM_PROTOCOLS", False);
	wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);

	XSetWMProtocols(display, window, &wm_delete_window, 1);

	XMapWindow(display, window);
	XFlush(display);

	XSelectInput(display, window, KeyPressMask | KeyReleaseMask);

	static const int context_attribs[] =
	{
		GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
#if OUTDATED_MESA == 1
		GLX_CONTEXT_MINOR_VERSION_ARB, 1,
#else
		GLX_CONTEXT_MINOR_VERSION_ARB, 2,
#endif
		GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
		None
	};

#if OUTDATED_MESA == 1
	// Mesa does not export the following GLX symbol - procure it manually
	PFNGLXCREATECONTEXTATTRIBSARBPROC const pfn_glXCreateContextAttribsARB =
		reinterpret_cast< PFNGLXCREATECONTEXTATTRIBSARBPROC >(glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB"));

	assert(0 != pfn_glXCreateContextAttribsARB);
	context = pfn_glXCreateContextAttribsARB(display, fbc[0], 0, True, context_attribs);

#else
	context = glXCreateContextAttribsARB(display, fbc[0], 0, True, context_attribs);

#endif

	if (0 == context)
	{
		stream::cerr << "failed to create GL context\n";
		return 1;
	}

	glXMakeCurrent(display, window, context);

	if (util::reportGLError())
	{
		stream::cerr << "warning: gl context started with errors; wiping the slate clean\n";
		while (GL_NO_ERROR != glGetError());
	}

	if (-1U != zbitness)
		glEnable(GL_DEPTH_TEST);

	return 0;
}


bool
deinitGL()
{
	if (0 != context)
	{
		glXMakeCurrent(display, None, 0);
		glXDestroyContext(display, context);
	}

	if (0 != window)
		XDestroyWindow(display, window);

	if (0 != display)
		XCloseDisplay(display);

	context = 0;
	window = 0;
	display = 0;

	return true;
}


void
swapBuffers()
{
	assert(0 != display);
	assert(0 != window);

	glXSwapBuffers(display, window);
}


bool
processEvents()
{
	assert(0 != display);
	assert(0 != window);

	XEvent event;

	if (!XCheckTypedWindowEvent(display, window, ClientMessage, &event))
		return true;

	if (event.xclient.message_type == wm_protocols &&
		event.xclient.data.l[0] == long(wm_delete_window))
	{
		return false;
	}

	XPutBackEvent(display, &event);

	return true;
}


bool
processEvents(
	GLuint& input)
{
	assert(0 != display);
	assert(0 != window);

	XEvent event;
	char buffer[16];

	while (XPending(display) > 0)
	{
		XNextEvent(display, &event);
		GLuint input_mask = 0;

		switch (event.type)
		{
		case ClientMessage:
			if (event.xclient.message_type == wm_protocols &&
				event.xclient.data.l[0] == long(wm_delete_window))
			{
				return false;
			}
			break;

		case KeyPress:
		case KeyRelease:

			XLookupString((XKeyPressedEvent *) &event, buffer, sizeof(buffer) - 1, NULL, NULL);

			switch (buffer[0])
			{
			case 27:
				input_mask = testbed::INPUT_MASK_ESC;
				break;
			case ' ':
				input_mask = testbed::INPUT_MASK_ACTION;
				break;
			case '1':
				input_mask = testbed::INPUT_MASK_OPTION_1;
				break;
			case '2':
				input_mask = testbed::INPUT_MASK_OPTION_2;
				break;
			case '3':
				input_mask = testbed::INPUT_MASK_OPTION_3;
				break;
			case '4':
				input_mask = testbed::INPUT_MASK_OPTION_4;
				break;
			case 'a':
			case 'A':
				input_mask = testbed::INPUT_MASK_ALT_LEFT;
				break;
			case 'd':
			case 'D':
				input_mask = testbed::INPUT_MASK_ALT_RIGHT;
				break;
			case 'i':
			case 'I':
				input_mask = testbed::INPUT_MASK_UP;
				break;
			case 'j':
			case 'J':
				input_mask = testbed::INPUT_MASK_LEFT;
				break;
			case 'l':
			case 'L':
				input_mask = testbed::INPUT_MASK_RIGHT;
				break;
			case 'm':
			case 'M':
				input_mask = testbed::INPUT_MASK_DOWN;
				break;
			case 'w':
			case 'W':
				input_mask = testbed::INPUT_MASK_ALT_UP;
				break;
			case 'z':
			case 'Z':
				input_mask = testbed::INPUT_MASK_ALT_DOWN;
				break;
			}

			if (KeyPress == event.type)
				input |= input_mask;
			else
				input &= ~input_mask;

			break;

		default:
			// drop this event
			break;
		}
	}

	return true;
}

} // namespace testbed

