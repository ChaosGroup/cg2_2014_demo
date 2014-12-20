#include <GL/gl.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "cmath_fix"
#include <iostream>

#include "vectsimd.hpp"
#include "testbed.hpp"
#include "scoped.hpp"
#include "prim_palette_view.hpp"

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

namespace pltv
{

enum {
	PROG_PALETTE_VIEW,

	PROG_COUNT,
	PROG_FORCE_UINT = -1U
};

enum {
	UNI_COLOR_PALETTE,
	UNI_SAMPLER_BILLBOARD,

	UNI_COUNT,
	UNI_FORCE_UINT = -1U
};

enum {
	TEX_BILLBOARD,

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
			std::cerr << __FUNCTION__ << " failed at glGenTextures" << std::endl;
			return false;
		}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE, g_tex[TEX_BILLBOARD]);

	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_R8UI, g_tex_w, g_tex_h, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, 0);

	glBindTexture(GL_TEXTURE_RECTANGLE, 0);

	if (util::reportGLError())
	{
		std::cerr << __FUNCTION__ << " failed at texture setup" << std::endl;
		return false;
	}

	// reset uniforms
	for (size_t i = 0; i < PROG_COUNT; ++i)
		for (unsigned j = 0; j < UNI_COUNT; ++j)
			g_uni[i][j] = -1;

	// load shaders "basic" and "texture", and resolve args
	g_shader_vert[PROG_PALETTE_VIEW] = glCreateShader(GL_VERTEX_SHADER);
	assert(g_shader_vert[PROG_PALETTE_VIEW]);

	if (!util::setupShader(g_shader_vert[PROG_PALETTE_VIEW], "basic.glslv"))
	{
		std::cerr << __FUNCTION__ << " failed at setupShader" << std::endl;
		return false;
	}

	g_shader_frag[PROG_PALETTE_VIEW] = glCreateShader(GL_FRAGMENT_SHADER);
	assert(g_shader_frag[PROG_PALETTE_VIEW]);

	if (!util::setupShader(g_shader_frag[PROG_PALETTE_VIEW], "texture.glslf"))
	{
		std::cerr << __FUNCTION__ << " failed at setupShader" << std::endl;
		return false;
	}

	g_shader_prog[PROG_PALETTE_VIEW] = glCreateProgram();
	assert(g_shader_prog[PROG_PALETTE_VIEW]);

	if (!util::setupProgram(
			g_shader_prog[PROG_PALETTE_VIEW],
			g_shader_vert[PROG_PALETTE_VIEW],
			g_shader_frag[PROG_PALETTE_VIEW]))
	{
		std::cerr << __FUNCTION__ << " failed at setupProgram" << std::endl;
		return false;
	}

	g_uni[PROG_PALETTE_VIEW][UNI_COLOR_PALETTE] =
		glGetUniformLocation(g_shader_prog[PROG_PALETTE_VIEW], "color_palette");
	g_uni[PROG_PALETTE_VIEW][UNI_SAMPLER_BILLBOARD] =
		glGetUniformLocation(g_shader_prog[PROG_PALETTE_VIEW], "albedo_map");

	g_active_attr_semantics[PROG_PALETTE_VIEW].registerVertexAttr(
		glGetAttribLocation(g_shader_prog[PROG_PALETTE_VIEW], "at_Vertex"));

	// prepare meshes
	glGenVertexArrays(sizeof(g_vao) / sizeof(g_vao[0]), g_vao);

	for (size_t i = 0; i < sizeof(g_vao) / sizeof(g_vao[0]); ++i)
		if (0 == g_vao[i])
		{
			std::cerr << __FUNCTION__ << " failed at glGenVertexArrays" << std::endl;
			return false;
		}

	glGenBuffers(sizeof(g_vbo) / sizeof(g_vbo[0]), g_vbo);

	for (size_t i = 0; i < sizeof(g_vbo) / sizeof(g_vbo[0]); ++i)
		if (0 == g_vbo[i])
		{
			std::cerr << __FUNCTION__ << " failed at glGenBuffers" << std::endl;
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
		std::cerr << __FUNCTION__ << " failed at glBufferData" << std::endl;
		return false;
	}

	for (size_t i = 0; i < PROG_COUNT; ++i)
	{
		glBindVertexArray(g_vao[i]);

		if (!setupVertexAttrPointers< Vertex >(g_active_attr_semantics[i]) ||
			0 == DEBUG_LITERAL && util::reportGLError())
		{
			std::cerr << __FUNCTION__ << " failed at setupVertexAttrPointers" << std::endl;
			return false;
		}
	}

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	on_error.reset();
	return true;
}


bool
render(
	const simd::vect4 (& palette)[4],
	const void* pixmap)
{
	glDisable(GL_DEPTH_TEST);

	const GLfloat color_palette[][4] =
	{
		{ palette[0][0], palette[0][1], palette[0][2], palette[0][3] },
		{ palette[1][0], palette[1][1], palette[1][2], palette[1][3] },
		{ palette[2][0], palette[2][1], palette[2][2], palette[2][3] },
		{ palette[3][0], palette[3][1], palette[3][2], palette[3][3] }
	};

	// prepare and issue the drawcall
	const unsigned prog = PROG_PALETTE_VIEW;

	glUseProgram(g_shader_prog[prog]);

	DEBUG_GL_ERR()

	if (-1 != g_uni[prog][UNI_SAMPLER_BILLBOARD] && 0 != pixmap)
	{
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_RECTANGLE, g_tex[TEX_BILLBOARD]);

		glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, g_tex_w, g_tex_h, GL_RED_INTEGER, GL_UNSIGNED_BYTE, pixmap);

		glUniform1i(g_uni[prog][UNI_SAMPLER_BILLBOARD], 0);
	}

	DEBUG_GL_ERR()

	glBindVertexArray(g_vao[prog]);

	DEBUG_GL_ERR()

	if (-1 != g_uni[prog][UNI_COLOR_PALETTE])
	{
		glUniform4fv(g_uni[prog][UNI_COLOR_PALETTE], 4, color_palette[0]);
	}

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

} // namespace pltv
} // namespace testbed
