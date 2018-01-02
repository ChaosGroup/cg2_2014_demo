#include "native_gl.h"
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
#include "prim_mesh.hpp"

#include "rendVertAttr.hpp"

namespace {

#define SETUP_VERTEX_ATTR_POINTERS_MASK	(			\
		SETUP_VERTEX_ATTR_POINTERS_MASK_vertex)

#include "rendVertAttr_setupVertAttrPointers.hpp"
#undef SETUP_VERTEX_ATTR_POINTERS_MASK

struct Vertex {
	GLfloat pos[3];
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

namespace mesh
{

enum {
	PROG_STATICMESH,

	PROG_COUNT,
	PROG_FORCE_UINT = -1U
};

enum {
	UNI_LP_OBJ,
	UNI_VP_OBJ,
	UNI_MVP,
	UNI_LPROD_SPECULAR,
	UNI_CLIP_EXTREMES,
	UNI_WORLD_EXTREMES,

	UNI_COUNT,
	UNI_FORCE_UINT = -1U
};

enum {
	MESH_STATICMESH,

	MESH_COUNT,
	MESH_FORCE_UINT = -1U
};

enum {
	VBO_STATICMESH_VTX,
	VBO_STATICMESH_IDX,

	VBO_COUNT,
	VBO_FORCE_UINT = -1U
};

static GLint g_uni[PROG_COUNT][UNI_COUNT];

static GLuint g_vao[PROG_COUNT];
static GLuint g_vbo[VBO_COUNT];
static GLuint g_shader_vert[PROG_COUNT];
static GLuint g_shader_frag[PROG_COUNT];
static GLuint g_shader_prog[PROG_COUNT];

static unsigned g_num_faces[MESH_COUNT];

static rend::ActiveAttrSemantics g_active_attr_semantics[PROG_COUNT];


class matx3_rotate : public simd::matx3 {
	matx3_rotate();

public:
	matx3_rotate(
		const float a,
		const float x,
		const float y,
		const float z) {

		const float sin_a = std::sin(a);
		const float cos_a = std::cos(a);

		static_cast< simd::matx3& >(*this) = simd::matx3(
			x * x + cos_a * (1 - x * x),         x * y - cos_a * (x * y) + sin_a * z, x * z - cos_a * (x * z) - sin_a * y,
			y * x - cos_a * (y * x) - sin_a * z, y * y + cos_a * (1 - y * y),         y * z - cos_a * (y * z) + sin_a * x,
			z * x - cos_a * (z * x) + sin_a * y, z * y - cos_a * (z * y) - sin_a * x, z * z + cos_a * (1 - z * z));
	}
};


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
init_resources(
	const size_t countof_vert,
	const void* const vert,
	const size_t countof_face,
	const void* const face) {

	if (0 == countof_face) {
		stream::cerr << __FUNCTION__ << " received a countof_face argument which is not positive\n";
		return false;
	}

	scoped_ptr< deinit_resources_t, scoped_functor > on_error(deinit_resources);

	// reset uniforms
	for (unsigned i = 0; i < PROG_COUNT; ++i)
		for (unsigned j = 0; j < UNI_COUNT; ++j)
			g_uni[i][j] = -1;

	// load shaders and resolve args
	g_shader_vert[PROG_STATICMESH] = glCreateShader(GL_VERTEX_SHADER);
	assert(g_shader_vert[PROG_STATICMESH]);

	if (!util::setupShader(g_shader_vert[PROG_STATICMESH], "phong.glslv")) {
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_frag[PROG_STATICMESH] = glCreateShader(GL_FRAGMENT_SHADER);
	assert(g_shader_frag[PROG_STATICMESH]);

	if (!util::setupShader(g_shader_frag[PROG_STATICMESH], "phong.glslf")) {
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_prog[PROG_STATICMESH] = glCreateProgram();
	assert(g_shader_prog[PROG_STATICMESH]);

	if (!util::setupProgram(
			g_shader_prog[PROG_STATICMESH],
			g_shader_vert[PROG_STATICMESH],
			g_shader_frag[PROG_STATICMESH])) {

		stream::cerr << __FUNCTION__ << " failed at setupProgram\n";
		return false;
	}

	g_uni[PROG_STATICMESH][UNI_MVP] =
		glGetUniformLocation(g_shader_prog[PROG_STATICMESH], "mvp");
	g_uni[PROG_STATICMESH][UNI_LP_OBJ] =
		glGetUniformLocation(g_shader_prog[PROG_STATICMESH], "lp_obj");
	g_uni[PROG_STATICMESH][UNI_VP_OBJ] =
		glGetUniformLocation(g_shader_prog[PROG_STATICMESH], "vp_obj");
	g_uni[PROG_STATICMESH][UNI_LPROD_SPECULAR] =
		glGetUniformLocation(g_shader_prog[PROG_STATICMESH], "lprod_specular");
	g_uni[PROG_STATICMESH][UNI_CLIP_EXTREMES] =
		glGetUniformLocation(g_shader_prog[PROG_STATICMESH], "clp");
	g_uni[PROG_STATICMESH][UNI_WORLD_EXTREMES] =
		glGetUniformLocation(g_shader_prog[PROG_STATICMESH], "wld");

	g_active_attr_semantics[PROG_STATICMESH].registerVertexAttr(
		glGetAttribLocation(g_shader_prog[PROG_STATICMESH], "at_Vertex"));

	// prepare meshes
	glGenVertexArrays(sizeof(g_vao) / sizeof(g_vao[0]), g_vao);

	for (unsigned i = 0; i < sizeof(g_vao) / sizeof(g_vao[0]); ++i)
		assert(g_vao[i]);

	glGenBuffers(sizeof(g_vbo) / sizeof(g_vbo[0]), g_vbo);

	for (unsigned i = 0; i < sizeof(g_vbo) / sizeof(g_vbo[0]); ++i)
		assert(g_vbo[i]);

	// upload mesh data to vertex and index buffers
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo[VBO_STATICMESH_VTX]);
	glBufferData(GL_ARRAY_BUFFER, countof_vert * sizeof(Vertex), vert, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (util::reportGLError()) {
		stream::cerr << __FUNCTION__ << " failed at glBindBuffer/glBufferData for ARRAY_BUFFER\n";
		return false;
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_vbo[VBO_STATICMESH_IDX]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, countof_face * sizeof(GLuint[3]), face, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	if (util::reportGLError()) {
		stream::cerr << __FUNCTION__ << " failed at glBindBuffer/glBufferData for ELEMENT_ARRAY_BUFFER\n";
		return false;
	}

	glBindVertexArray(g_vao[PROG_STATICMESH]);
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo[VBO_STATICMESH_VTX]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_vbo[VBO_STATICMESH_IDX]);

	if (!setupVertexAttrPointers< Vertex >(g_active_attr_semantics[PROG_STATICMESH]) ||
		0 == DEBUG_LITERAL && util::reportGLError()) {

		stream::cerr << __FUNCTION__ << " failed at setupVertexAttrPointers\n";
		return false;
	}

	glBindVertexArray(0);

	g_num_faces[MESH_STATICMESH] = countof_face;

	on_error.reset();
	return true;
}


bool
render(
	const simd::matx4& mv,
	const simd::matx4& mvp,
	const simd::vect3 (& clp)[2],
	const simd::vect3 (& wld)[2],
	const simd::vect4& color) {

	const GLfloat mat[][4] = {
		{ mvp[0][0], mvp[0][1], mvp[0][2], mvp[0][3] },
		{ mvp[1][0], mvp[1][1], mvp[1][2], mvp[1][3] },
		{ mvp[2][0], mvp[2][1], mvp[2][2], mvp[2][3] },
		{ mvp[3][0], mvp[3][1], mvp[3][2], mvp[3][3] }
	};

	const GLfloat clip[][3] = {
		{ clp[0][0], clp[0][1], clp[0][2] },
		{ clp[1][0], clp[1][1], clp[1][2] }
	};

	const GLfloat world[][3] = {
		{ wld[0][0], wld[0][1], wld[0][2] },
		{ wld[1][0], wld[1][1], wld[1][2] }
	};

	const GLfloat specular[] = {
		color[0],
		color[1],
		color[2],
		color[3]
	};

	// prepare and issue the drawcall
	glUseProgram(g_shader_prog[PROG_STATICMESH]);

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_STATICMESH][UNI_MVP]) {
		glUniformMatrix4fv(g_uni[PROG_STATICMESH][UNI_MVP], 1, GL_FALSE, mat[0]);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_STATICMESH][UNI_LP_OBJ]) {
		const GLfloat nonlocal_light[4] = {
			mv[0][2],
			mv[1][2],
			mv[2][2],
			0.f
		};

		glUniform4fv(g_uni[PROG_STATICMESH][UNI_LP_OBJ], 1, nonlocal_light);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_STATICMESH][UNI_VP_OBJ]) {
		const GLfloat nonlocal_viewer[4] = {
			mv[0][2],
			mv[1][2],
			mv[2][2],
			0.f
		};

		glUniform4fv(g_uni[PROG_STATICMESH][UNI_VP_OBJ], 1, nonlocal_viewer);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_STATICMESH][UNI_LPROD_SPECULAR]) {
		glUniform4fv(g_uni[PROG_STATICMESH][UNI_LPROD_SPECULAR], 1, specular);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_STATICMESH][UNI_CLIP_EXTREMES]) {
		glUniform3fv(g_uni[PROG_STATICMESH][UNI_CLIP_EXTREMES], 2, clip[0]);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_STATICMESH][UNI_WORLD_EXTREMES]) {
		glUniform3fv(g_uni[PROG_STATICMESH][UNI_WORLD_EXTREMES], 2, world[0]);
	}

	DEBUG_GL_ERR()

	glBindVertexArray(g_vao[PROG_STATICMESH]);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[PROG_STATICMESH].num_active_attr; ++i)
		glEnableVertexAttribArray(g_active_attr_semantics[PROG_STATICMESH].active_attr[i]);

	DEBUG_GL_ERR()

	glDrawElements(GL_TRIANGLES, g_num_faces[MESH_STATICMESH] * 3, GL_UNSIGNED_INT, 0);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[PROG_STATICMESH].num_active_attr; ++i)
		glDisableVertexAttribArray(g_active_attr_semantics[PROG_STATICMESH].active_attr[i]);

	DEBUG_GL_ERR()

	return true;
}

} // namespace mesh
} // namespace testbed

