#include "native_gl.h"
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "cmath_fix"
#include <sstream>

#include "platform.hpp"
#include "scoped.hpp"
#include "stream.hpp"
#include "prim_rgb_view.hpp"

#include "rendVertAttr.hpp"

namespace
{

#define SETUP_VERTEX_ATTR_POINTERS_MASK	(			\
		SETUP_VERTEX_ATTR_POINTERS_MASK_vert2d)

#include "rendVertAttr_setupVertAttrPointers.hpp"
#undef SETUP_VERTEX_ATTR_POINTERS_MASK

struct Vertex
{
	GLfloat pos[2];
};

} // namespace

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

namespace rgbv
{

enum {
	PROG_RGB_VIEW,
	PROG_RGB_VIEW_VERTICAL,
	PROG_RGB_VIEW_HORIZONTAL,

	PROG_COUNT,
	PROG_FORCE_UINT = -1U
};

enum {
	UNI_SAMPLER_BILLBOARD,
	UNI_SAMPLER_FBO,
	UNI_CONTRAST_MIDDLE_AND_K,

	UNI_COUNT,
	UNI_FORCE_UINT = -1U
};

enum {
	TEX_BILLBOARD,
	TEX_FBO,

	TEX_COUNT,
	TEX_FORCE_UINT = -1U
};

enum {
	MESH_BILLBOARD,

	MESH_COUNT,
	MESH_FORCE_UINT = -1U
};

enum {
	VBO_BILLBOARD_VTX,

	VBO_COUNT,
	VBO_FORCE_UINT = -1U
};

static GLint g_uni[PROG_COUNT][UNI_COUNT];

static GLuint g_tex[TEX_COUNT];
static GLuint g_vao[PROG_COUNT];
static GLuint g_vbo[VBO_COUNT];
static GLuint g_shader_vert[PROG_COUNT];
static GLuint g_shader_frag[PROG_COUNT];
static GLuint g_shader_prog[PROG_COUNT];
static GLuint g_fbo;

static rend::ActiveAttrSemantics g_active_attr_semantics[PROG_COUNT];

static GLsizei g_tex_w;
static GLsizei g_tex_h;

bool
deinit_resources()
{
	for (size_t i = 0; i < sizeof(g_shader_prog) / sizeof(g_shader_prog[0]); ++i)
	{
		glDeleteProgram(g_shader_prog[i]);
		g_shader_prog[i] = 0;
	}

	for (size_t i = 0; i < sizeof(g_shader_vert) / sizeof(g_shader_vert[0]); ++i)
	{
		glDeleteShader(g_shader_vert[i]);
		g_shader_vert[i] = 0;
	}

	for (size_t i = 0; i < sizeof(g_shader_frag) / sizeof(g_shader_frag[0]); ++i)
	{
		glDeleteShader(g_shader_frag[i]);
		g_shader_frag[i] = 0;
	}

	glDeleteFramebuffers(1, &g_fbo);
	g_fbo = 0;

	glDeleteTextures(sizeof(g_tex) / sizeof(g_tex[0]), g_tex);
	memset(g_tex, 0, sizeof(g_tex));

	glDeleteVertexArrays(sizeof(g_vao) / sizeof(g_vao[0]), g_vao);
	memset(g_vao, 0, sizeof(g_vao));

	glDeleteBuffers(sizeof(g_vbo) / sizeof(g_vbo[0]), g_vbo);
	memset(g_vbo, 0, sizeof(g_vbo));

	return true;
}


bool
init_resources(
	const size_t w,
	const size_t h)
{
	scoped_ptr< deinit_resources_t, scoped_functor > on_error(deinit_resources);

	g_tex_w = w;
	g_tex_h = h;

	glGenTextures(sizeof(g_tex) / sizeof(g_tex[0]), g_tex);

	for (size_t i = 0; i < sizeof(g_tex) / sizeof(g_tex[0]); ++i)
		if (0 == g_tex[i])
		{
			stream::cerr << __FUNCTION__ << " failed at glGenTextures\n";
			return false;
		}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE, g_tex[TEX_BILLBOARD]);

	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA8UI, g_tex_w, g_tex_h, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, 0);

	glBindTexture(GL_TEXTURE_RECTANGLE, 0);

	if (util::reportGLError())
	{
		stream::cerr << __FUNCTION__ << " failed at texture setup\n";
		return false;
	}

	// reset uniforms
	for (size_t i = 0; i < PROG_COUNT; ++i)
		for (unsigned j = 0; j < UNI_COUNT; ++j)
			g_uni[i][j] = -1;

	// load shaders "basic" and "texture", and resolve args
	g_shader_vert[PROG_RGB_VIEW] = glCreateShader(GL_VERTEX_SHADER);
	assert(g_shader_vert[PROG_RGB_VIEW]);

	if (!util::setupShader(g_shader_vert[PROG_RGB_VIEW], "basic.glslv"))
	{
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_frag[PROG_RGB_VIEW] = glCreateShader(GL_FRAGMENT_SHADER);
	assert(g_shader_frag[PROG_RGB_VIEW]);

	if (!util::setupShader(g_shader_frag[PROG_RGB_VIEW], "texture_rgba.glslf"))
	{
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_prog[PROG_RGB_VIEW] = glCreateProgram();
	assert(g_shader_prog[PROG_RGB_VIEW]);

	if (!util::setupProgram(
			g_shader_prog[PROG_RGB_VIEW],
			g_shader_vert[PROG_RGB_VIEW],
			g_shader_frag[PROG_RGB_VIEW]))
	{
		stream::cerr << __FUNCTION__ << " failed at setupProgram\n";
		return false;
	}

	g_uni[PROG_RGB_VIEW][UNI_SAMPLER_BILLBOARD] =
		glGetUniformLocation(g_shader_prog[PROG_RGB_VIEW], "albedo_map");

	g_active_attr_semantics[PROG_RGB_VIEW].registerVertexAttr(
		glGetAttribLocation(g_shader_prog[PROG_RGB_VIEW], "at_Vertex"));

	// load shaders "basic" and "texture_vertical", and resolve args
	g_shader_vert[PROG_RGB_VIEW_VERTICAL] = glCreateShader(GL_VERTEX_SHADER);
	assert(g_shader_vert[PROG_RGB_VIEW_VERTICAL]);

	if (!util::setupShader(g_shader_vert[PROG_RGB_VIEW_VERTICAL], "basic.glslv"))
	{
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_frag[PROG_RGB_VIEW_VERTICAL] = glCreateShader(GL_FRAGMENT_SHADER);
	assert(g_shader_frag[PROG_RGB_VIEW_VERTICAL]);

	if (!util::setupShader(g_shader_frag[PROG_RGB_VIEW_VERTICAL], "texture_rgba_v.glslf"))
	{
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_prog[PROG_RGB_VIEW_VERTICAL] = glCreateProgram();
	assert(g_shader_prog[PROG_RGB_VIEW_VERTICAL]);

	if (!util::setupProgram(
			g_shader_prog[PROG_RGB_VIEW_VERTICAL],
			g_shader_vert[PROG_RGB_VIEW_VERTICAL],
			g_shader_frag[PROG_RGB_VIEW_VERTICAL]))
	{
		stream::cerr << __FUNCTION__ << " failed at setupProgram\n";
		return false;
	}

	g_uni[PROG_RGB_VIEW_VERTICAL][UNI_SAMPLER_BILLBOARD] =
		glGetUniformLocation(g_shader_prog[PROG_RGB_VIEW_VERTICAL], "albedo_map");

	g_active_attr_semantics[PROG_RGB_VIEW_VERTICAL].registerVertexAttr(
		glGetAttribLocation(g_shader_prog[PROG_RGB_VIEW_VERTICAL], "at_Vertex"));

	// load shaders "basic" and "texture_horizontal", and resolve args
	g_shader_vert[PROG_RGB_VIEW_HORIZONTAL] = glCreateShader(GL_VERTEX_SHADER);
	assert(g_shader_vert[PROG_RGB_VIEW_HORIZONTAL]);

	if (!util::setupShader(g_shader_vert[PROG_RGB_VIEW_HORIZONTAL], "basic.glslv"))
	{
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_frag[PROG_RGB_VIEW_HORIZONTAL] = glCreateShader(GL_FRAGMENT_SHADER);
	assert(g_shader_frag[PROG_RGB_VIEW_HORIZONTAL]);

	std::ostringstream patch;
	patch << " " << w / 2 << ".0 ";

	if (!util::setupShaderWithPatch(g_shader_frag[PROG_RGB_VIEW_HORIZONTAL], "texture_rgba_h.glslf",
		" 256.0 ", patch.str()))
	{
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_prog[PROG_RGB_VIEW_HORIZONTAL] = glCreateProgram();
	assert(g_shader_prog[PROG_RGB_VIEW_HORIZONTAL]);

	if (!util::setupProgram(
			g_shader_prog[PROG_RGB_VIEW_HORIZONTAL],
			g_shader_vert[PROG_RGB_VIEW_HORIZONTAL],
			g_shader_frag[PROG_RGB_VIEW_HORIZONTAL]))
	{
		stream::cerr << __FUNCTION__ << " failed at setupProgram\n";
		return false;
	}

	g_uni[PROG_RGB_VIEW_HORIZONTAL][UNI_SAMPLER_BILLBOARD] =
		glGetUniformLocation(g_shader_prog[PROG_RGB_VIEW_HORIZONTAL], "albedo_map");

	g_uni[PROG_RGB_VIEW_HORIZONTAL][UNI_SAMPLER_FBO] =
		glGetUniformLocation(g_shader_prog[PROG_RGB_VIEW_HORIZONTAL], "fbo_map");

	g_uni[PROG_RGB_VIEW_HORIZONTAL][UNI_CONTRAST_MIDDLE_AND_K] =
		glGetUniformLocation(g_shader_prog[PROG_RGB_VIEW_HORIZONTAL], "contrast_mid_k_cookie");

	g_active_attr_semantics[PROG_RGB_VIEW_HORIZONTAL].registerVertexAttr(
		glGetAttribLocation(g_shader_prog[PROG_RGB_VIEW_HORIZONTAL], "at_Vertex"));

	// prepare meshes
	glGenVertexArrays(sizeof(g_vao) / sizeof(g_vao[0]), g_vao);

	for (size_t i = 0; i < sizeof(g_vao) / sizeof(g_vao[0]); ++i)
		if (0 == g_vao[i])
		{
			stream::cerr << __FUNCTION__ << " failed at glGenVertexArrays\n";
			return false;
		}

	glGenBuffers(sizeof(g_vbo) / sizeof(g_vbo[0]), g_vbo);

	for (size_t i = 0; i < sizeof(g_vbo) / sizeof(g_vbo[0]); ++i)
		if (0 == g_vbo[i])
		{
			stream::cerr << __FUNCTION__ << " failed at glGenBuffers\n";
			return false;
		}

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo[VBO_BILLBOARD_VTX]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	const Vertex vtx[] = // tri-strip
	{
		{ { -1.f, -1.f } },
		{ {  1.f, -1.f } },
		{ { -1.f,  1.f } },
		{ {  1.f,  1.f } }
	};

	glBufferData(GL_ARRAY_BUFFER, sizeof(vtx), vtx, GL_STATIC_DRAW);

	if (util::reportGLError())
	{
		stream::cerr << __FUNCTION__ << " failed at glBufferData\n";
		return false;
	}

	for (size_t i = 0; i < PROG_COUNT; ++i)
	{
		glBindVertexArray(g_vao[i]);

		if (!setupVertexAttrPointers< Vertex >(g_active_attr_semantics[i]) ||
			0 == DEBUG_LITERAL && util::reportGLError())
		{
			stream::cerr << __FUNCTION__ << " failed at setupVertexAttrPointers\n";
			return false;
		}
	}

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	// create fbo
	glGenFramebuffers(1, &g_fbo);
	assert(g_fbo);

	glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);

	glBindTexture(GL_TEXTURE_RECTANGLE, g_tex[TEX_FBO]);

	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA8UI, g_tex_w, g_tex_h, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, 0);

	if (util::reportGLError())
	{
		stream::cerr << __FUNCTION__ << " failed at TEX_FBO creation\n";
		return false;
	}

	glBindTexture(GL_TEXTURE_RECTANGLE, 0);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, g_tex[TEX_FBO], 0);

	const GLenum fbo_success = glCheckFramebufferStatus(GL_FRAMEBUFFER);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	if (GL_FRAMEBUFFER_COMPLETE != fbo_success)
	{
		stream::cerr << __FUNCTION__ << " failed at glCheckFramebufferStatus\n";
		return false;
	}

	on_error.reset();
	return true;
}


bool
render(
	const void* pixmap,
	const float contrast_middle,
	const float contrast_k,
	const float blur_split)
{
	assert(0 != pixmap);

	const GLfloat contrast[] =
	{
		contrast_middle,
		contrast_k,
		blur_split
	};

	glDisable(GL_DEPTH_TEST);

	// prepare and issue the first drawcall
	size_t prog = PROG_RGB_VIEW_VERTICAL;

	glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);

	DEBUG_GL_ERR()

	glUseProgram(g_shader_prog[prog]);

	DEBUG_GL_ERR()

	if (-1 != g_uni[prog][UNI_SAMPLER_BILLBOARD])
	{
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_RECTANGLE, g_tex[TEX_BILLBOARD]);

		glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, g_tex_w, g_tex_h, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, pixmap);

		glUniform1i(g_uni[prog][UNI_SAMPLER_BILLBOARD], 0);
	}

	glBindVertexArray(g_vao[prog]);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[prog].num_active_attr; ++i)
		glEnableVertexAttribArray(g_active_attr_semantics[prog].active_attr[i]);

	DEBUG_GL_ERR()

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[prog].num_active_attr; ++i)
		glDisableVertexAttribArray(g_active_attr_semantics[prog].active_attr[i]);

	DEBUG_GL_ERR()

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// prepare and issue the concluding drawcall
	prog = PROG_RGB_VIEW_HORIZONTAL;

	glUseProgram(g_shader_prog[prog]);

	DEBUG_GL_ERR()

	if (-1 != g_uni[prog][UNI_SAMPLER_BILLBOARD])
	{
		// billoard texture already set up at tex0
		glUniform1i(g_uni[prog][UNI_SAMPLER_BILLBOARD], 0);
	}

	if (-1 != g_uni[prog][UNI_SAMPLER_FBO])
	{
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_RECTANGLE, g_tex[TEX_FBO]);

		glUniform1i(g_uni[prog][UNI_SAMPLER_FBO], 1);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[prog][UNI_CONTRAST_MIDDLE_AND_K])
	{
		glUniform3fv(g_uni[prog][UNI_CONTRAST_MIDDLE_AND_K], 1, contrast);
	}

	DEBUG_GL_ERR()

	glBindVertexArray(g_vao[prog]);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[prog].num_active_attr; ++i)
		glEnableVertexAttribArray(g_active_attr_semantics[prog].active_attr[i]);

	DEBUG_GL_ERR()

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[prog].num_active_attr; ++i)
		glDisableVertexAttribArray(g_active_attr_semantics[prog].active_attr[i]);

	DEBUG_GL_ERR()

	glEnable(GL_DEPTH_TEST);

	return true;
}


bool
render(
	const void* pixmap)
{
	assert(0 != pixmap);

	glDisable(GL_DEPTH_TEST);

	// prepare and issue the drawcall
	const size_t prog = PROG_RGB_VIEW;

	glUseProgram(g_shader_prog[prog]);

	DEBUG_GL_ERR()

	if (-1 != g_uni[prog][UNI_SAMPLER_BILLBOARD])
	{
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_RECTANGLE, g_tex[TEX_BILLBOARD]);

		glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, g_tex_w, g_tex_h, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, pixmap);

		glUniform1i(g_uni[prog][UNI_SAMPLER_BILLBOARD], 0);
	}

	DEBUG_GL_ERR()

	glBindVertexArray(g_vao[prog]);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[prog].num_active_attr; ++i)
		glEnableVertexAttribArray(g_active_attr_semantics[prog].active_attr[i]);

	DEBUG_GL_ERR()

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[prog].num_active_attr; ++i)
		glDisableVertexAttribArray(g_active_attr_semantics[prog].active_attr[i]);

	DEBUG_GL_ERR()

	glEnable(GL_DEPTH_TEST);

	return true;
}

} // namespace rgbv
} // namespace testbed
