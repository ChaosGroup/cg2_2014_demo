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
#include "prim_poly.hpp"

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

namespace poly
{

enum {
	PROG_BASIC,
	PROG_STRIPED,

	PROG_COUNT,
	PROG_FORCE_UINT = -1U
};

enum {
	UNI_MVP,
	UNI_COLOR,

	UNI_COUNT,
	UNI_FORCE_UINT = -1U
};

enum {
	MESH_POLY,

	MESH_COUNT,
	MESH_FORCE_UINT = -1U
};

enum {
	VBO_POLY_VTX,

	VBO_COUNT,
	VBO_FORCE_UINT = -1U
};

static GLint g_uni[PROG_COUNT][UNI_COUNT];

static GLuint g_vao[PROG_COUNT];
static GLuint g_vbo[VBO_COUNT];
static GLuint g_shader_vert[PROG_COUNT];
static GLuint g_shader_frag[PROG_COUNT];
static GLuint g_shader_prog[PROG_COUNT];

static rend::ActiveAttrSemantics g_active_attr_semantics[PROG_COUNT];

bool
deinit_resources()
{
	for (unsigned i = 0; i < sizeof(g_shader_prog) / sizeof(g_shader_prog[0]); ++i)
	{
		glDeleteProgram(g_shader_prog[i]);
		g_shader_prog[i] = 0;
	}

	for (unsigned i = 0; i < sizeof(g_shader_vert) / sizeof(g_shader_vert[0]); ++i)
	{
		glDeleteShader(g_shader_vert[i]);
		g_shader_vert[i] = 0;
	}

	for (unsigned i = 0; i < sizeof(g_shader_frag) / sizeof(g_shader_frag[0]); ++i)
	{
		glDeleteShader(g_shader_frag[i]);
		g_shader_frag[i] = 0;
	}

	glDeleteVertexArrays(sizeof(g_vao) / sizeof(g_vao[0]), g_vao);
	memset(g_vao, 0, sizeof(g_vao));

	glDeleteBuffers(sizeof(g_vbo) / sizeof(g_vbo[0]), g_vbo);
	memset(g_vbo, 0, sizeof(g_vbo));

	return true;
}


bool
init_resources(
	const unsigned count)
{
	scoped_ptr< deinit_resources_t, scoped_functor > on_error(deinit_resources);

	// reset uniforms
	for (unsigned i = 0; i < PROG_COUNT; ++i)
		for (unsigned j = 0; j < UNI_COUNT; ++j)
			g_uni[i][j] = -1;

	// load shader "basic" and resolve args
	g_shader_vert[PROG_BASIC] = glCreateShader(GL_VERTEX_SHADER);
	assert(g_shader_vert[PROG_BASIC]);

	if (!util::setupShader(g_shader_vert[PROG_BASIC], "basic.glslv"))
	{
		std::cerr << __FUNCTION__ << " failed at setupShader" << std::endl;
		return false;
	}

	g_shader_frag[PROG_BASIC] = glCreateShader(GL_FRAGMENT_SHADER);
	assert(g_shader_frag[PROG_BASIC]);

	if (!util::setupShader(g_shader_frag[PROG_BASIC], "basic.glslf"))
	{
		std::cerr << __FUNCTION__ << " failed at setupShader" << std::endl;
		return false;
	}

	g_shader_prog[PROG_BASIC] = glCreateProgram();
	assert(g_shader_prog[PROG_BASIC]);

	if (!util::setupProgram(
			g_shader_prog[PROG_BASIC],
			g_shader_vert[PROG_BASIC],
			g_shader_frag[PROG_BASIC]))
	{
		std::cerr << __FUNCTION__ << " failed at setupProgram" << std::endl;
		return false;
	}

	g_uni[PROG_BASIC][UNI_MVP] =
		glGetUniformLocation(g_shader_prog[PROG_BASIC], "mvp");
	g_uni[PROG_BASIC][UNI_COLOR] =
		glGetUniformLocation(g_shader_prog[PROG_BASIC], "color");

	g_active_attr_semantics[PROG_BASIC].registerVertexAttr(
		glGetAttribLocation(g_shader_prog[PROG_BASIC], "at_Vertex"));

	// load shader "striped" and resolve args
	g_shader_vert[PROG_STRIPED] = glCreateShader(GL_VERTEX_SHADER);
	assert(g_shader_vert[PROG_STRIPED]);

	if (!util::setupShader(g_shader_vert[PROG_STRIPED], "basic.glslv"))
	{
		std::cerr << __FUNCTION__ << " failed at setupShader" << std::endl;
		return false;
	}

	g_shader_frag[PROG_STRIPED] = glCreateShader(GL_FRAGMENT_SHADER);
	assert(g_shader_frag[PROG_STRIPED]);

	if (!util::setupShader(g_shader_frag[PROG_STRIPED], "striped.glslf"))
	{
		std::cerr << __FUNCTION__ << " failed at setupShader" << std::endl;
		return false;
	}

	g_shader_prog[PROG_STRIPED] = glCreateProgram();
	assert(g_shader_prog[PROG_STRIPED]);

	if (!util::setupProgram(
			g_shader_prog[PROG_STRIPED],
			g_shader_vert[PROG_STRIPED],
			g_shader_frag[PROG_STRIPED]))
	{
		std::cerr << __FUNCTION__ << " failed at setupProgram" << std::endl;
		return false;
	}

	g_uni[PROG_STRIPED][UNI_MVP] =
		glGetUniformLocation(g_shader_prog[PROG_STRIPED], "mvp");
	g_uni[PROG_STRIPED][UNI_COLOR] =
		glGetUniformLocation(g_shader_prog[PROG_STRIPED], "color");

	g_active_attr_semantics[PROG_STRIPED].registerVertexAttr(
		glGetAttribLocation(g_shader_prog[PROG_STRIPED], "at_Vertex"));

	// prepare meshes
	glGenVertexArrays(sizeof(g_vao) / sizeof(g_vao[0]), g_vao);

	for (unsigned i = 0; i < sizeof(g_vao) / sizeof(g_vao[0]); ++i)
		assert(g_vao[i]);

	glGenBuffers(sizeof(g_vbo) / sizeof(g_vbo[0]), g_vbo);

	for (unsigned i = 0; i < sizeof(g_vbo) / sizeof(g_vbo[0]); ++i)
		assert(g_vbo[i]);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo[VBO_POLY_VTX]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBufferData(GL_ARRAY_BUFFER, count * sizeof(Vertex), 0, GL_STATIC_DRAW);

	if (util::reportGLError())
	{
		std::cerr << __FUNCTION__ <<
			" failed at glBufferData" << std::endl;
		return false;
	}

	for (unsigned i = 0; i < PROG_COUNT; ++i)
	{
		glBindVertexArray(g_vao[i]);

		if (!setupVertexAttrPointers< Vertex >(g_active_attr_semantics[i]) ||
			0 == DEBUG_LITERAL && util::reportGLError())
		{
			std::cerr << __FUNCTION__ <<
				" failed at setupVertexAttrPointers" << std::endl;
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
init_resources(
	const unsigned start,
	const unsigned count,
	const void* const vert)
{
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo[VBO_POLY_VTX]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	if (util::reportGLError())
	{
		std::cerr << __FUNCTION__ <<
			" failed at glBindBuffer" << std::endl;
		return false;
	}

	glBufferSubData(GL_ARRAY_BUFFER,
		start * sizeof(Vertex),
		count * sizeof(Vertex),
		vert);

	if (util::reportGLError())
	{
		std::cerr << __FUNCTION__ <<
			" failed at glBufferSubData" << std::endl;
		return false;
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	return true;
}


bool
render(
	const simd::matx4& mv,
	const simd::vect4& color,
	const unsigned start,
	const unsigned count,
	const bool striped)
{
	// sign-inver z in all rows, for GL screen space
	const GLfloat mvp[4][4] =
	{
		{ mv[0][0], mv[0][1], -mv[0][2], 0.f },
		{ mv[1][0], mv[1][1], -mv[1][2], 0.f },
		{ mv[2][0], mv[2][1], -mv[2][2], 0.f },
		{ mv[3][0], mv[3][1], -mv[3][2], 1.f }
	};

	const GLfloat shade[] =
	{
		color[0],
		color[1],
		color[2],
		color[3]
	};

	// prepare and issue the drawcall
	unsigned prog = PROG_BASIC;

	if (striped)
		prog = PROG_STRIPED;

	glUseProgram(g_shader_prog[prog]);

	DEBUG_GL_ERR()

	glBindVertexArray(g_vao[prog]);

	DEBUG_GL_ERR()

	if (-1 != g_uni[prog][UNI_MVP])
	{
		glUniformMatrix4fv(g_uni[prog][UNI_MVP], 1, GL_FALSE, mvp[0]);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[prog][UNI_COLOR])
	{
		glUniform4fv(g_uni[prog][UNI_COLOR], 1, shade);
	}

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[prog].num_active_attr; ++i)
		glEnableVertexAttribArray(g_active_attr_semantics[prog].active_attr[i]);

	DEBUG_GL_ERR()

	glDrawArrays(GL_TRIANGLE_FAN, start, count);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[prog].num_active_attr; ++i)
		glDisableVertexAttribArray(g_active_attr_semantics[prog].active_attr[i]);

	DEBUG_GL_ERR()

	return true;
}

} // namespace poly
} // namespace testbed
