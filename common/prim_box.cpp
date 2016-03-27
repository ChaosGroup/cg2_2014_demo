#include <GL/gl.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "cmath_fix"

#include "vectsimd_sse.hpp"
#include "platform.hpp"
#include "scoped.hpp"
#include "stream.hpp"
#include "prim_box.hpp"

#include "rendVertAttr.hpp"

namespace {

#define SETUP_VERTEX_ATTR_POINTERS_MASK	(			\
		SETUP_VERTEX_ATTR_POINTERS_MASK_index)

#include "rendVertAttr_setupVertAttrPointers.hpp"
#undef SETUP_VERTEX_ATTR_POINTERS_MASK

struct Vertex {
	GLuint idx[1];
};

} // namespace

namespace testbed {

template < typename T >
class generic_free {
public:
	void operator()(T* arg) {
		assert(0 != arg);
		free(arg);
	}
};

namespace box {

enum {
	PROG_LINE,

	PROG_COUNT,
	PROG_FORCE_UINT = -1U
};

enum {
	UNI_MVP,
	UNI_COLOR,
	UNI_END_POINTS,

	UNI_COUNT,
	UNI_FORCE_UINT = -1U
};

enum {
	MESH_LINE,

	MESH_COUNT,
	MESH_FORCE_UINT = -1U
};

enum {
	VBO_LINE_VTX,

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
deinit_resources() {

	for (unsigned i = 0; i < sizeof(g_shader_prog) / sizeof(g_shader_prog[0]); ++i) {
		glDeleteProgram(g_shader_prog[i]);
		g_shader_prog[i] = 0;
	}

	for (unsigned i = 0; i < sizeof(g_shader_vert) / sizeof(g_shader_vert[0]); ++i) {
		glDeleteShader(g_shader_vert[i]);
		g_shader_vert[i] = 0;
	}

	for (unsigned i = 0; i < sizeof(g_shader_frag) / sizeof(g_shader_frag[0]); ++i) {
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
init_resources() {

	scoped_ptr< deinit_resources_t, scoped_functor > on_error(deinit_resources);

	// reset uniforms
	for (unsigned i = 0; i < PROG_COUNT; ++i)
		for (unsigned j = 0; j < UNI_COUNT; ++j)
			g_uni[i][j] = -1;

	// load shaders and resolve args
	g_shader_vert[PROG_LINE] = glCreateShader(GL_VERTEX_SHADER);
	assert(g_shader_vert[PROG_LINE]);

	if (!util::setupShader(g_shader_vert[PROG_LINE], "box.glslv")) {
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_frag[PROG_LINE] = glCreateShader(GL_FRAGMENT_SHADER);
	assert(g_shader_frag[PROG_LINE]);

	if (!util::setupShader(g_shader_frag[PROG_LINE], "basic.glslf")) {
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_prog[PROG_LINE] = glCreateProgram();
	assert(g_shader_prog[PROG_LINE]);

	if (!util::setupProgram(
			g_shader_prog[PROG_LINE],
			g_shader_vert[PROG_LINE],
			g_shader_frag[PROG_LINE])) {

		stream::cerr << __FUNCTION__ << " failed at setupProgram\n";
		return false;
	}

	g_uni[PROG_LINE][UNI_MVP] =
		glGetUniformLocation(g_shader_prog[PROG_LINE], "mvp");
	g_uni[PROG_LINE][UNI_COLOR] =
		glGetUniformLocation(g_shader_prog[PROG_LINE], "color");
	g_uni[PROG_LINE][UNI_END_POINTS]	=
		glGetUniformLocation(g_shader_prog[PROG_LINE], "end");

	g_active_attr_semantics[PROG_LINE].registerIndexAttr(
		glGetAttribLocation(g_shader_prog[PROG_LINE], "at_Index"));

	// prepare meshes
	glGenVertexArrays(sizeof(g_vao) / sizeof(g_vao[0]), g_vao);

	for (unsigned i = 0; i < sizeof(g_vao) / sizeof(g_vao[0]); ++i)
		assert(g_vao[i]);

	glGenBuffers(sizeof(g_vbo) / sizeof(g_vbo[0]), g_vbo);

	for (unsigned i = 0; i < sizeof(g_vbo) / sizeof(g_vbo[0]); ++i)
		assert(g_vbo[i]);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo[VBO_LINE_VTX]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	struct Triad {
		GLuint m;

		Triad(
			const unsigned x,
			const unsigned y,
			const unsigned z)
		: m(((x & 1) << 0) | ((y & 1) << 1) | ((z & 1) << 2)) {
		}
		operator GLuint () const {
			return m;
		}
	};

	// crate a mesh of twelve vertices each of a sole index attribute, encoding the vertex components: 0 - min extremum, 1 - max extremum
	const Vertex v[] = {
		{ { Triad(0, 0, 0) } },
		{ { Triad(1, 0, 0) } },
		{ { Triad(0, 0, 0) } },
		{ { Triad(0, 1, 0) } },
		{ { Triad(0, 0, 0) } },
		{ { Triad(0, 0, 1) } },
		{ { Triad(1, 1, 1) } },
		{ { Triad(0, 1, 1) } },
		{ { Triad(1, 1, 1) } },
		{ { Triad(1, 0, 1) } },
		{ { Triad(1, 1, 1) } },
		{ { Triad(1, 1, 0) } },
		{ { Triad(0, 0, 1) } },
		{ { Triad(0, 1, 1) } },
		{ { Triad(1, 0, 0) } },
		{ { Triad(1, 1, 0) } },
		{ { Triad(0, 1, 0) } },
		{ { Triad(0, 1, 1) } },
		{ { Triad(0, 1, 0) } },
		{ { Triad(1, 1, 0) } },
		{ { Triad(1, 0, 1) } },
		{ { Triad(1, 0, 0) } },
		{ { Triad(1, 0, 1) } },
		{ { Triad(0, 0, 1) } }
	};

	glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (util::reportGLError()) {
		stream::cerr << __FUNCTION__ << " failed at glBufferData\n";
		return false;
	}

	glBindVertexArray(g_vao[PROG_LINE]);
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo[VBO_LINE_VTX]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	if (!setupVertexAttrPointers< Vertex >(g_active_attr_semantics[PROG_LINE]) ||
		0 == DEBUG_LITERAL && util::reportGLError()) {

		stream::cerr << __FUNCTION__ << " failed at setupVertexAttrPointers\n";
		return false;
	}

	glBindVertexArray(0);

	on_error.reset();
	return true;
}


bool
render(
	const simd::matx4& mvp,
	const simd::vect3 (& extremum)[2],
	const simd::vect4& color) {

	const GLfloat mat[4][4] = {
		{ mvp[0][0], mvp[0][1], mvp[0][2], mvp[0][3] },
		{ mvp[1][0], mvp[1][1], mvp[1][2], mvp[1][3] },
		{ mvp[2][0], mvp[2][1], mvp[2][2], mvp[2][3] },
		{ mvp[3][0], mvp[3][1], mvp[3][2], mvp[3][3] }
	};

	const GLfloat point[][3] = {
		{ extremum[0][0], extremum[0][1], extremum[0][2] },
		{ extremum[1][0], extremum[1][1], extremum[1][2] }
	};

	const GLfloat shade[] = {
		color[0],
		color[1],
		color[2],
		color[3]
	};

	// prepare and issue the drawcall
	glUseProgram(g_shader_prog[PROG_LINE]);

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_LINE][UNI_MVP]) {
		glUniformMatrix4fv(g_uni[PROG_LINE][UNI_MVP], 1, GL_FALSE, mat[0]);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_LINE][UNI_COLOR]) {
		glUniform4fv(g_uni[PROG_LINE][UNI_COLOR], 1, shade);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_LINE][UNI_END_POINTS]) {
		glUniform3fv(g_uni[PROG_LINE][UNI_END_POINTS], 2, point[0]);
	}

	DEBUG_GL_ERR()

	glBindVertexArray(g_vao[PROG_LINE]);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[PROG_LINE].num_active_attr; ++i)
		glEnableVertexAttribArray(g_active_attr_semantics[PROG_LINE].active_attr[i]);

	DEBUG_GL_ERR()

	glDrawArrays(GL_LINES, 0, 24);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[PROG_LINE].num_active_attr; ++i)
		glDisableVertexAttribArray(g_active_attr_semantics[PROG_LINE].active_attr[i]);

	DEBUG_GL_ERR()

	return true;
}

} // namespace box
} // namespace testbed

