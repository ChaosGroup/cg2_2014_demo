#ifndef testbed_H__
#define testbed_H__

#include "native_gl.h"
#include "input.h"
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

bool
processEvents(
	unsigned& input);

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
