#ifndef testbed_H__
#define testbed_H__

#include <GL/gl.h>
#include <string>
#include "pure_macro.hpp"
#include "scoped.hpp"
#include "stream.hpp"

#define DEBUG_GL_ERR()												\
	if (DEBUG_LITERAL && testbed::util::reportGLError())			\
	{																\
		stream::cerr << __FILE__ ":" XQUOTE(__LINE__) "\n";			\
		return false;												\
	}

namespace testbed
{

int
initGL(
	const unsigned w,
	const unsigned h,
	unsigned (&bitness)[4],
	const unsigned zbitness,
	const unsigned fsaa);

bool
deinitGL();

void
swapBuffers();

bool
processEvents();

enum
{
	INPUT_MASK_ESC			= 1 <<  0,
	INPUT_MASK_LEFT			= 1 <<  1,
	INPUT_MASK_RIGHT		= 1 <<  2,
	INPUT_MASK_DOWN			= 1 <<  3,
	INPUT_MASK_UP			= 1 <<  4,
	INPUT_MASK_ALT_LEFT		= 1 <<  5,
	INPUT_MASK_ALT_RIGHT	= 1 <<  6,
	INPUT_MASK_ALT_DOWN		= 1 <<  7,
	INPUT_MASK_ALT_UP		= 1 <<  8,
	INPUT_MASK_ACTION		= 1 <<  9,
	INPUT_MASK_OPTION_1		= 1 << 10,
	INPUT_MASK_OPTION_2		= 1 << 11,
	INPUT_MASK_OPTION_3		= 1 << 12,

	INPUT_MASK_FORCE_TYPE	= GLuint(-1)
};

bool
processEvents(
	GLuint& input);

namespace util
{

bool
reportGLError();

bool
reportGLCaps();

bool
setupShader(
	const GLuint shader_name,
	const char* const filename);

bool
setupShaderWithPatch(
	const GLuint shader_name,
	const char* const filename,
	const std::string& patch_out,
	const std::string& patch_in);

bool
setupProgram(
	const GLuint prog_name,
	const GLuint shader_vname,
	const GLuint shader_fname);

} // namespace util
} // namespace testbed

#endif // testbed_H__
