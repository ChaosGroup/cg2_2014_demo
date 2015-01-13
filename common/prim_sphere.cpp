#include <GL/gl.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "cmath_fix"

#include "vectsimd.hpp"
#include "testbed.hpp"
#include "scoped.hpp"
#include "stream.hpp"
#include "prim_sphere.hpp"

#include "rendVertAttr.hpp"

namespace
{

#define SETUP_VERTEX_ATTR_POINTERS_MASK	(			\
		SETUP_VERTEX_ATTR_POINTERS_MASK_vertex |	\
		SETUP_VERTEX_ATTR_POINTERS_MASK_normal)

#include "rendVertAttr_setupVertAttrPointers.hpp"
#undef SETUP_VERTEX_ATTR_POINTERS_MASK

struct Vertex
{
	GLfloat pos[3];
	GLfloat nrm[3];
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

namespace sphe
{

enum {
	PROG_SPHERE,

	PROG_COUNT,
	PROG_FORCE_UINT = -1U
};

enum {
	UNI_LP_OBJ,
	UNI_VP_OBJ,
	UNI_MVP,
	UNI_LPROD_DIFFUSE,
	UNI_CLIP_PLANES,

	UNI_COUNT,
	UNI_FORCE_UINT = -1U
};

enum {
	MESH_SPHERE,

	MESH_COUNT,
	MESH_FORCE_UINT = -1U
};

enum {
	VBO_SPHERE_VTX,
	VBO_SPHERE_IDX,

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


class matx3_rotate : public simd::matx3
{
	matx3_rotate();

public:
	matx3_rotate(
		const float a,
		const float x,
		const float y,
		const float z)
	{
		const float sin_a = std::sin(a);
		const float cos_a = std::cos(a);

		static_cast< simd::matx3& >(*this) = simd::matx3(
			x * x + cos_a * (1 - x * x),         x * y - cos_a * (x * y) + sin_a * z, x * z - cos_a * (x * z) - sin_a * y,
			y * x - cos_a * (y * x) - sin_a * z, y * y + cos_a * (1 - y * y),         y * z - cos_a * (y * z) + sin_a * x,
			z * x - cos_a * (z * x) + sin_a * y, z * y - cos_a * (z * y) - sin_a * x, z * z + cos_a * (1 - z * z));
	}
};


static bool
createIndexedPolarSphere(
	const unsigned rows,
	const unsigned cols,
	const GLuint vbo_arr,
	const GLuint vbo_idx,
	unsigned& num_faces)
{
	assert(vbo_arr && vbo_idx);
	assert(rows > 2);
	assert(cols > 3);

	const unsigned num_verts = (rows - 2) * cols + 2 * (cols - 1);
	const unsigned num_indes = ((rows - 3) * 2 + 2) * (cols - 1);
	num_faces = ((rows - 3) * 2 + 2) * (cols - 1);

	const size_t sizeof_arr = num_verts * sizeof(Vertex);
	scoped_ptr< Vertex, generic_free > arr((Vertex*) malloc(sizeof_arr));
	unsigned ai = 0;

	const float r = 1.f;

	// north pole
	for (unsigned j = 0; j < cols - 1; ++j)
	{
		assert(ai < sizeof_arr / sizeof(arr()[0]));

		arr()[ai].pos[0] = 0.f;
		arr()[ai].pos[1] = r;
		arr()[ai].pos[2] = 0.f;

		arr()[ai].nrm[0] = 0.f;
		arr()[ai].nrm[1] = 1.f;
		arr()[ai].nrm[2] = 0.f;

		++ai;
	}

	for (unsigned i = 1; i < rows - 1; ++i)
		for (unsigned j = 0; j < cols; ++j)
		{
			assert(ai < sizeof_arr / sizeof(arr()[0]));

			const float azim_angle = (j * 2) * M_PI / (cols - 1);
			const float azim_x = 0.f;
			const float azim_y = 1.f;
			const float azim_z = 0.f;

			const float decl_angle = -M_PI_2 + M_PI * i / (rows - 1);
			const float decl_x = 1.f;
			const float decl_y = 0.f;
			const float decl_z = 0.f;

			const simd::matx3 azim_decl = simd::matx3().mul(
				matx3_rotate(decl_angle, decl_x, decl_y, decl_z),
				matx3_rotate(azim_angle, azim_x, azim_y, azim_z));

			arr()[ai].pos[0] = azim_decl[2][0] * r;
			arr()[ai].pos[1] = azim_decl[2][1] * r;
			arr()[ai].pos[2] = azim_decl[2][2] * r;

			arr()[ai].nrm[0] = azim_decl[2][0];
			arr()[ai].nrm[1] = azim_decl[2][1];
			arr()[ai].nrm[2] = azim_decl[2][2];

			++ai;
		}

	// south pole
	for (unsigned j = 0; j < cols - 1; ++j)
	{
		assert(ai < sizeof_arr / sizeof(arr()[0]));

		arr()[ai].pos[0] = 0.f;
		arr()[ai].pos[1] = -r;
		arr()[ai].pos[2] = 0.f;

		arr()[ai].nrm[0] = 0.f;
		arr()[ai].nrm[1] = -1.f;
		arr()[ai].nrm[2] = 0.f;

		++ai;
	}

	assert(ai == sizeof_arr / sizeof(arr()[0]));

	const size_t sizeof_idx = num_indes * sizeof(uint16_t[3]);
	scoped_ptr< uint16_t[3], generic_free > idx((uint16_t (*)[3]) malloc(sizeof_idx));
	unsigned ii = 0;

	// north pole
	for (unsigned j = 0; j < cols - 1; ++j)
	{
		assert(ii < sizeof_idx / sizeof(idx()[0]));

		idx()[ii][0] = uint16_t(j);
		idx()[ii][1] = uint16_t(j + cols - 1);
		idx()[ii][2] = uint16_t(j + cols);

		++ii;
	}

	for (unsigned i = 1; i < rows - 2; ++i)
		for (unsigned j = 0; j < cols - 1; ++j)
		{
			assert(ii < sizeof_idx / sizeof(idx()[0]));

			idx()[ii][0] = uint16_t(j + i * cols);
			idx()[ii][1] = uint16_t(j + i * cols - 1);
			idx()[ii][2] = uint16_t(j + (i + 1) * cols);

			++ii;

			assert(ii < sizeof_idx / sizeof(idx()[0]));

			idx()[ii][0] = uint16_t(j + (i + 1) * cols - 1);
			idx()[ii][1] = uint16_t(j + (i + 1) * cols);
			idx()[ii][2] = uint16_t(j + i * cols - 1);

			++ii;
		}

	// south pole
	for (unsigned j = 0; j < cols - 1; ++j)
	{
		assert(ii < sizeof_idx / sizeof(idx()[0]));

		idx()[ii][0] = uint16_t(j + (rows - 2) * cols);
		idx()[ii][1] = uint16_t(j + (rows - 2) * cols - 1);
		idx()[ii][2] = uint16_t(j + (rows - 2) * cols + cols - 1);

		++ii;
	}

	assert(ii == sizeof_idx / sizeof(idx()[0]));

	stream::cout << "number of vertices: " << num_verts <<
		"\nnumber of indices: " << num_indes << '\n';

	glBindBuffer(GL_ARRAY_BUFFER, vbo_arr);
	glBufferData(GL_ARRAY_BUFFER, sizeof_arr, arr(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (util::reportGLError())
	{
		stream::cerr << __FUNCTION__ << " failed at glBindBuffer/glBufferData for ARRAY_BUFFER\n";
		return false;
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_idx);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof_idx, idx(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	if (util::reportGLError())
	{
		stream::cerr << __FUNCTION__ << " failed at glBindBuffer/glBufferData for ELEMENT_ARRAY_BUFFER\n";
		return false;
	}

	return true;
}


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
init_resources()
{
	scoped_ptr< deinit_resources_t, scoped_functor > on_error(deinit_resources);

	// reset uniforms
	for (unsigned i = 0; i < PROG_COUNT; ++i)
		for (unsigned j = 0; j < UNI_COUNT; ++j)
			g_uni[i][j] = -1;

	// load shaders and resolve args
	g_shader_vert[PROG_SPHERE] = glCreateShader(GL_VERTEX_SHADER);
	assert(g_shader_vert[PROG_SPHERE]);

	if (!util::setupShader(g_shader_vert[PROG_SPHERE], "phong.glslv"))
	{
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_frag[PROG_SPHERE] = glCreateShader(GL_FRAGMENT_SHADER);
	assert(g_shader_frag[PROG_SPHERE]);

	if (!util::setupShader(g_shader_frag[PROG_SPHERE], "phong.glslf"))
	{
		stream::cerr << __FUNCTION__ << " failed at setupShader\n";
		return false;
	}

	g_shader_prog[PROG_SPHERE] = glCreateProgram();
	assert(g_shader_prog[PROG_SPHERE]);

	if (!util::setupProgram(
			g_shader_prog[PROG_SPHERE],
			g_shader_vert[PROG_SPHERE],
			g_shader_frag[PROG_SPHERE]))
	{
		stream::cerr << __FUNCTION__ << " failed at setupProgram\n";
		return false;
	}

	g_uni[PROG_SPHERE][UNI_MVP] =
		glGetUniformLocation(g_shader_prog[PROG_SPHERE], "mvp");
	g_uni[PROG_SPHERE][UNI_LP_OBJ] =
		glGetUniformLocation(g_shader_prog[PROG_SPHERE], "lp_obj");
	g_uni[PROG_SPHERE][UNI_VP_OBJ] =
		glGetUniformLocation(g_shader_prog[PROG_SPHERE], "vp_obj");
	g_uni[PROG_SPHERE][UNI_LPROD_DIFFUSE] =
		glGetUniformLocation(g_shader_prog[PROG_SPHERE], "lprod_diffuse");
	g_uni[PROG_SPHERE][UNI_CLIP_PLANES] =
		glGetUniformLocation(g_shader_prog[PROG_SPHERE], "clp");

	g_active_attr_semantics[PROG_SPHERE].registerVertexAttr(
		glGetAttribLocation(g_shader_prog[PROG_SPHERE], "at_Vertex"));
	g_active_attr_semantics[PROG_SPHERE].registerNormalAttr(
		glGetAttribLocation(g_shader_prog[PROG_SPHERE], "at_Normal"));

	// prepare meshes
	glGenVertexArrays(sizeof(g_vao) / sizeof(g_vao[0]), g_vao);

	for (unsigned i = 0; i < sizeof(g_vao) / sizeof(g_vao[0]); ++i)
		assert(g_vao[i]);

	glGenBuffers(sizeof(g_vbo) / sizeof(g_vbo[0]), g_vbo);

	for (unsigned i = 0; i < sizeof(g_vbo) / sizeof(g_vbo[0]); ++i)
		assert(g_vbo[i]);

	// bend a polar sphere from a grid of the following dimensions:
	const unsigned rows = 33;
	const unsigned cols = 65;

	if (!createIndexedPolarSphere(
			rows,
			cols,
			g_vbo[VBO_SPHERE_VTX],
			g_vbo[VBO_SPHERE_IDX],
			g_num_faces[MESH_SPHERE]))
	{
		stream::cerr << __FUNCTION__ << " failed at createIndexedPolarSphere\n";
		return false;
	}

	glBindVertexArray(g_vao[PROG_SPHERE]);
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo[VBO_SPHERE_VTX]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_vbo[VBO_SPHERE_IDX]);

	if (!setupVertexAttrPointers< Vertex >(g_active_attr_semantics[PROG_SPHERE]) ||
		0 == DEBUG_LITERAL && util::reportGLError())
	{
		stream::cerr << __FUNCTION__ << " failed at setupVertexAttrPointers\n";
		return false;
	}

	glBindVertexArray(0);

	on_error.reset();
	return true;
}


bool
render(
	const simd::matx4& mv,
	const simd::vect4 (& clp)[2],
	const simd::vect4& color)
{
	// sign-inver z in all rows, for GL screen space
	const GLfloat mvp[4][4] =
	{
		{ mv[0][0], mv[0][1], -mv[0][2], 0.f },
		{ mv[1][0], mv[1][1], -mv[1][2], 0.f },
		{ mv[2][0], mv[2][1], -mv[2][2], 0.f },
		{ mv[3][0], mv[3][1], -mv[3][2], 1.f }
	};

	const GLfloat clip[][4] =
	{
		{ clp[0][0], clp[0][1], clp[0][2], clp[0][3] },
		{ clp[1][0], clp[1][1], clp[1][2], clp[1][3] }
	};

	const GLfloat shade[] =
	{
		color[0],
		color[1],
		color[2],
		color[3]
	};

	// prepare and issue the drawcall
	glUseProgram(g_shader_prog[PROG_SPHERE]);

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_SPHERE][UNI_MVP])
	{
		glUniformMatrix4fv(g_uni[PROG_SPHERE][UNI_MVP], 1, GL_FALSE, mvp[0]);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_SPHERE][UNI_LP_OBJ])
	{
		const GLfloat nonlocal_light[4] =
		{
			mv[0][2],
			mv[1][2],
			mv[2][2],
			0.f
		};

		glUniform4fv(g_uni[PROG_SPHERE][UNI_LP_OBJ], 1, nonlocal_light);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_SPHERE][UNI_VP_OBJ])
	{
		const GLfloat nonlocal_viewer[4] =
		{
			mv[0][2],
			mv[1][2],
			mv[2][2],
			0.f
		};

		glUniform4fv(g_uni[PROG_SPHERE][UNI_VP_OBJ], 1, nonlocal_viewer);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_SPHERE][UNI_LPROD_DIFFUSE])
	{
		glUniform4fv(g_uni[PROG_SPHERE][UNI_LPROD_DIFFUSE], 1, shade);
	}

	DEBUG_GL_ERR()

	if (-1 != g_uni[PROG_SPHERE][UNI_CLIP_PLANES])
	{
		glUniform4fv(g_uni[PROG_SPHERE][UNI_CLIP_PLANES], 2, clip[0]);
	}

	DEBUG_GL_ERR()

	glBindVertexArray(g_vao[PROG_SPHERE]);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[PROG_SPHERE].num_active_attr; ++i)
		glEnableVertexAttribArray(g_active_attr_semantics[PROG_SPHERE].active_attr[i]);

	DEBUG_GL_ERR()

	glDrawElements(GL_TRIANGLES, g_num_faces[MESH_SPHERE] * 3, GL_UNSIGNED_SHORT, 0);

	DEBUG_GL_ERR()

	for (unsigned i = 0; i < g_active_attr_semantics[PROG_SPHERE].num_active_attr; ++i)
		glDisableVertexAttribArray(g_active_attr_semantics[PROG_SPHERE].active_attr[i]);

	DEBUG_GL_ERR()

	return true;
}

} // namespace sphe
} // namespace testbed
