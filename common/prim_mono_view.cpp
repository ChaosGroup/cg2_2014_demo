#include "native_gl.h"
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "cmath_fix"

#include "platform.hpp"
#include "scoped.hpp"
#include "stream.hpp"
#include "prim_mono_view.hpp"

#include "rendVertAttr.hpp"

namespace {

#define SETUP_VERTEX_ATTR_POINTERS_MASK	(			\
		SETUP_VERTEX_ATTR_POINTERS_MASK_vert2d)

#include "rendVertAttr_setupVertAttrPointers.hpp"
#undef SETUP_VERTEX_ATTR_POINTERS_MASK

struct Vertex {
	GLfloat pos[2];
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

namespace monv {

enum {
	PROG_PALETTE_VIEW,

	PROG_COUNT,
	PROG_FORCE_UINT = -1U
};

enum {
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
deinit_resources() {

	for (size_t i = 0; i < sizeof(g_shader_prog) / sizeof(g_shader_prog[0]); ++i) {
		glDeleteProgram(g_shader_prog[i]);
		g_shader_prog[i] = 0;
	}

	for (size_t i = 0; i < sizeof(g_shader_vert) / sizeof(g_shader_vert[0]); ++i) {
		glDeleteShader(g_shader_vert[i]);
		g_shader_vert[i] = 0;
	}

	for (size_t i = 0; i < sizeof(g_shader_frag) / sizeof(g_shader_frag[0]); ++i) {
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
	const size_t h) {

	scoped_ptr< deinit_resources_t, scoped_functor > on_error(deinit_resources);

	g_tex_w = w;
	g_tex_h = h;

	glGenTextures(sizeof(g_tex) / sizeof(g_tex[0]), g_tex);

	for (size_t i = 0; i < sizeof(g_tex) / sizeof(g_tex[0]); ++i)
		if (0 == g_tex[i]) {
			stream::cerr << __FUNCTION__ << " failed at glGenTextures\n";
			return false;
		}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE, g_tex[TEX_BILLBOARD]);

	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_R8UI, g_tex_w, g_tex_h, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, 0);

	glBindTexture(GL_TEXTURE_RECTANGLE, 0);

	if (util::reportGLError()) {
		stream::cerr << __FUNCTION__ << " failed at texture setup\n";
		return false;
	}

	// reset uniforms
	for (size_t i = 0; i < PROG_COUNT; ++i)
		for (unsigned j = 0; j < UNI_COUNT; ++j)
			g_uni[i][j] = -1;

	// load shaders "basic" and "texture", and resolve args
	g_shader_vert[PROG_PALETTE_VIEW] = glCreateShader(GL_VERTEX_SHADER);
	assert(g_shader_vert[PROG_PALETTE_VIEW]);

	if (!util::setupShader(g_shader_vert[PROG_PALETTE_VIEW], "basic.glslv")) {
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_frag[PROG_PALETTE_VIEW] = glCreateShader(GL_FRAGMENT_SHADER);
	assert(g_shader_frag[PROG_PALETTE_VIEW]);

	if (!util::setupShader(g_shader_frag[PROG_PALETTE_VIEW], "texture.glslf")) {
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_prog[PROG_PALETTE_VIEW] = glCreateProgram();
	assert(g_shader_prog[PROG_PALETTE_VIEW]);

	if (!util::setupProgram(
			g_shader_prog[PROG_PALETTE_VIEW],
			g_shader_vert[PROG_PALETTE_VIEW],
			g_shader_frag[PROG_PALETTE_VIEW])) {
		stream::cerr << __FUNCTION__ << " failed at setupProgram\n";
		return false;
	}

	g_uni[PROG_PALETTE_VIEW][UNI_SAMPLER_BILLBOARD] =
		glGetUniformLocation(g_shader_prog[PROG_PALETTE_VIEW], "albedo_map");

	g_active_attr_semantics[PROG_PALETTE_VIEW].registerVertexAttr(
		glGetAttribLocation(g_shader_prog[PROG_PALETTE_VIEW], "at_Vertex"));

	// prepare meshes
	glGenVertexArrays(sizeof(g_vao) / sizeof(g_vao[0]), g_vao);

	for (size_t i = 0; i < sizeof(g_vao) / sizeof(g_vao[0]); ++i)
		if (0 == g_vao[i]) {
			stream::cerr << __FUNCTION__ << " failed at glGenVertexArrays\n";
			return false;
		}

	glGenBuffers(sizeof(g_vbo) / sizeof(g_vbo[0]), g_vbo);

	for (size_t i = 0; i < sizeof(g_vbo) / sizeof(g_vbo[0]); ++i)
		if (0 == g_vbo[i]) {
			stream::cerr << __FUNCTION__ << " failed at glGenBuffers\n";
			return false;
		}

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo[VBO_BILLBOARD_VTX]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	const Vertex vtx[] = { // tri-strip
		{ { -1.f, -1.f } },
		{ {  1.f, -1.f } },
		{ { -1.f,  1.f } },
		{ {  1.f,  1.f } }
	};

	glBufferData(GL_ARRAY_BUFFER, sizeof(vtx), vtx, GL_STATIC_DRAW);

	if (util::reportGLError()) {
		stream::cerr << __FUNCTION__ << " failed at glBufferData\n";
		return false;
	}

	for (size_t i = 0; i < PROG_COUNT; ++i) {
		glBindVertexArray(g_vao[i]);

		if (!setupVertexAttrPointers< Vertex >(g_active_attr_semantics[i])) {
			stream::cerr << __FUNCTION__ << " failed at setupVertexAttrPointers\n";
			return false;
		}

		for (unsigned j = 0; j < g_active_attr_semantics[i].num_active_attr; ++j)
			glEnableVertexAttribArray(g_active_attr_semantics[i].active_attr[j]);

		DEBUG_GL_ERR()
	}

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	on_error.reset();
	return true;
}


bool
render(
	const void* pixmap)
{
	glDisable(GL_DEPTH_TEST);

	// prepare and issue the drawcall
	const unsigned prog = PROG_PALETTE_VIEW;

	glUseProgram(g_shader_prog[prog]);

	DEBUG_GL_ERR()

	if (-1 != g_uni[prog][UNI_SAMPLER_BILLBOARD] && 0 != pixmap) {

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_RECTANGLE, g_tex[TEX_BILLBOARD]);

		glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, g_tex_w, g_tex_h, GL_RED_INTEGER, GL_UNSIGNED_BYTE, pixmap);

		glUniform1i(g_uni[prog][UNI_SAMPLER_BILLBOARD], 0);
	}

	DEBUG_GL_ERR()

	glBindVertexArray(g_vao[prog]);

	DEBUG_GL_ERR()

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	DEBUG_GL_ERR()

	glEnable(GL_DEPTH_TEST);

	return true;
}


bool
render(
	const GLuint tex_name)
{
	glDisable(GL_DEPTH_TEST);

	// prepare and issue the drawcall
	const unsigned prog = PROG_PALETTE_VIEW;

	glUseProgram(g_shader_prog[prog]);

	DEBUG_GL_ERR()

	if (-1 != g_uni[prog][UNI_SAMPLER_BILLBOARD]) {

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_RECTANGLE, tex_name);

		glUniform1i(g_uni[prog][UNI_SAMPLER_BILLBOARD], 0);
	}

	DEBUG_GL_ERR()

	glBindVertexArray(g_vao[prog]);

	DEBUG_GL_ERR()

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	DEBUG_GL_ERR()

	glEnable(GL_DEPTH_TEST);

	return true;
}

} // namespace monv
} // namespace testbed
