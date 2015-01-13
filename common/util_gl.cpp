#include <GL/gl.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sstream>
#include <iomanip>

#include "testbed.hpp"
#include "get_file_size.hpp"
#include "scoped.hpp"
#include "stream.hpp"

static std::string
string_from_GL_error(
	const GLenum error)
{
	switch (error) {

	case GL_NO_ERROR:
		return "GL_NO_ERROR";

	case GL_INVALID_ENUM:
		return "GL_INVALID_ENUM";

	case GL_INVALID_FRAMEBUFFER_OPERATION:
		return "GL_INVALID_FRAMEBUFFER_OPERATION";

	case GL_INVALID_VALUE:
		return "GL_INVALID_VALUE";

	case GL_INVALID_OPERATION:
		return "GL_INVALID_OPERATION";

	case GL_OUT_OF_MEMORY:
		return "GL_OUT_OF_MEMORY";
	}

	std::ostringstream s;
	s << "unknown GL error (0x" << std::hex << std::setw(8) << std::setfill('0') << error << ")";

	return s.str();
}


static bool
setupShaderFromString(
	const GLuint shader_name,
	const char* const source,
	const size_t length)
{
	assert(0 != source);
	assert(0 != length);

	if (GL_FALSE == glIsShader(shader_name))
	{
		stream::cerr << __FUNCTION__ << " argument is not a valid shader object\n";
		return false;
	}

	const char* src[] =
	{
#if OUTDATED_MESA == 1
		"#version 140\n",
#else
		"#version 150\n",
#endif
		source
	};

	const GLint src_len[] =
	{
		(GLint) strlen(src[0]),
		(GLint) length
	};

	glShaderSource(shader_name, sizeof(src) / sizeof(src[0]), src, src_len);
	glCompileShader(shader_name);

	const GLsizei log_max_len = 2048;
	static char log[log_max_len + 1];
	GLsizei log_len;

	glGetShaderInfoLog(shader_name, log_max_len, &log_len, log);

	log[log_len] = '\0';
	stream::cerr << "shader compile log: " << log << '\n';

	GLint success = GL_FALSE;
	glGetShaderiv(shader_name, GL_COMPILE_STATUS, &success);

	return GL_TRUE == success;
}

namespace testbed
{

template < typename T >
class generic_free
{
public:

	void operator()(T* arg)
	{
		assert(0 != arg);
		free(arg);
	}
};

namespace util
{

bool
reportGLError()
{
	const GLenum error = glGetError();

	if (GL_NO_ERROR == error)
		return false;

	stream::cerr << "GL error: " << string_from_GL_error(error) << '\n';
	return true;
}


bool
reportGLCaps()
{
	const GLubyte* str_version	= glGetString(GL_VERSION);
	const GLubyte* str_vendor	= glGetString(GL_VENDOR);
	const GLubyte* str_renderer	= glGetString(GL_RENDERER);
	const GLubyte* str_glsl_ver	= glGetString(GL_SHADING_LANGUAGE_VERSION);

	stream::cout << "gl version, vendor, renderer, glsl version, extensions:"
		"\n\t" << (const char*) str_version <<
		"\n\t" << (const char*) str_vendor <<
		"\n\t" << (const char*) str_renderer <<
		"\n\t" << (const char*) str_glsl_ver <<
		"\n\t";

	GLint num_extensions;
	glGetIntegerv(GL_NUM_EXTENSIONS, &num_extensions);

	if (0 != num_extensions)
	{
		stream::cout << (const char*) glGetStringi(GL_EXTENSIONS, 0);

		for (GLuint i = 1; i != (GLuint) num_extensions; ++i)
			stream::cout << ' ' << (const char*) glGetStringi(GL_EXTENSIONS, i);

		stream::cout << "\n\n";
	}
	else
		stream::cout << "nil\n\n";

	GLint params[2]; // we won't need more than 2

	glGetIntegerv(GL_MAX_TEXTURE_SIZE, params);
	stream::cout << "GL_MAX_TEXTURE_SIZE: " << params[0] << '\n';

	glGetIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE, params);
	stream::cout << "GL_MAX_CUBE_MAP_TEXTURE_SIZE: " << params[0] << '\n';

	glGetIntegerv(GL_MAX_VIEWPORT_DIMS, params);
	stream::cout << "GL_MAX_VIEWPORT_DIMS: " << params[0] << ", " << params[1] << '\n';

	glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, params);
	stream::cout << "GL_MAX_RENDERBUFFER_SIZE: " << params[0] << '\n';

	glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, params);
	stream::cout << "GL_MAX_VERTEX_ATTRIBS: " << params[0] << '\n';

	glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, params);
	stream::cout << "GL_MAX_VERTEX_UNIFORM_COMPONENTS: " << params[0] << '\n';

	// for forward-compatible contexts we have to give up the next pname
#if 0
	glGetIntegerv(GL_MAX_VARYING_COMPONENTS, params);
	stream::cout << "GL_MAX_VARYING_COMPONENTS: " << params[0] << '\n';
#endif

	glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, params);
	stream::cout << "GL_MAX_FRAGMENT_UNIFORM_COMPONENTS: " << params[0] << '\n';

	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, params);
	stream::cout << "GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS: " << params[0] << '\n';

	glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, params);
	stream::cout << "GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS: " << params[0] << '\n';

	glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, params);
	stream::cout << "GL_MAX_TEXTURE_IMAGE_UNITS: " << params[0] << "\n\n";

	return true;
}


bool
setupShader(
	const GLuint shader_name,
	const char* const filename)
{
	assert(0 != filename);

	size_t length;
	const scoped_ptr< char, generic_free > source(
		get_buffer_from_file(filename, length));

	if (0 == source())
	{
		stream::cerr << __FUNCTION__ << " failed to read shader file '" << filename << "'\n";
		return false;
	}

	return setupShaderFromString(shader_name, source(), length);
}


bool
setupShaderWithPatch(
	const GLuint shader_name,
	const char* const filename,
	const std::string& patch_out,
	const std::string& patch_in)
{
	assert(0 != filename);

	size_t length;
	const scoped_ptr< char, generic_free > source(
		get_buffer_from_file(filename, length));

	if (0 == source())
	{
		stream::cerr << __FUNCTION__ << " failed to read shader file '" << filename << "'\n";
		return false;
	}

	if (patch_out.empty() || patch_out == patch_in)
		return setupShaderFromString(shader_name, source(), length);

	std::string src_final(source(), length);
	const size_t len_out = patch_out.length();
	const size_t len_in = patch_in.length();

	stream::cout << "turn: " << patch_out << "\ninto: " << patch_in;

	bool patched = false;

	for (size_t pos = src_final.find(patch_out);
		std::string::npos != pos;
		pos = src_final.find(patch_out, pos + len_in))
	{
		src_final.replace(pos, len_out, patch_in);
		patched = true;
	}

	if (patched)
		stream::cout << "\npatched\n";
	else
		stream::cout << "\nnot patched\n";

	return setupShaderFromString(shader_name, src_final.c_str(), src_final.length());
}


bool
setupProgram(
	const GLuint prog,
	const GLuint shader_vert,
	const GLuint shader_frag)
{
	if (GL_FALSE == glIsProgram(prog))
	{
		stream::cerr << __FUNCTION__ << " argument is not a valid program object\n";
		return false;
	}

	if (GL_FALSE == glIsShader(shader_vert) ||
		GL_FALSE == glIsShader(shader_frag))
	{
		stream::cerr << __FUNCTION__ << " argument is not a valid shader object\n";
		return false;
	}

	glAttachShader(prog, shader_vert);
	glAttachShader(prog, shader_frag);

	glLinkProgram(prog);

	const GLsizei log_max_len = 1024;
	static char log[log_max_len + 1];
	GLsizei log_len;

	glGetProgramInfoLog(prog, log_max_len, &log_len, log);

	log[log_len] = '\0';
	stream::cerr << "shader link log: " << log << '\n';

	GLint success = GL_FALSE;
	glGetProgramiv(prog, GL_LINK_STATUS, &success);

	return GL_TRUE == success;
}

} // namespace util
} // namespace testbed

