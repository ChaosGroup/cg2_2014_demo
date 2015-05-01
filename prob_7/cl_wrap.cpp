#ifdef __APPLE__
	#include <OpenCL/cl.h>
#else
	#include <CL/cl.h>
#endif	
#if CL_VERSION_1_2 == 0
	#error required CL_VERSION_1_2 to compile
#endif
#include "get_proc_address.hpp"
#include "cl_util.hpp"

using clutil::ocl_ver;

namespace clwrap {

typedef cl_mem (*pf_create_image_t)(
	cl_context,             // context
	cl_mem_flags,           // flags
	const cl_image_format*, // image_format
	const cl_image_desc*,   // image_desc
	void*,                  // host_ptr
	cl_int*);               // errcode_ret

typedef cl_mem (*pf_create_image2d_t)(
	cl_context,             // context
	cl_mem_flags,           // flags
	const cl_image_format*, // image_format
	size_t,                 // image_width
	size_t,                 // image_height
	size_t,                 // image_row_pitch
	void*,                  // host_ptr
	cl_int*);               // errcode_ret

typedef cl_mem (*pf_create_image3d_t)(
	cl_context,             // context
	cl_mem_flags,           // flags
	const cl_image_format*, // image_format
	size_t,                 // image_width
	size_t,                 // image_height
	size_t,                 // image_depth
	size_t,                 // image_row_pitch
	size_t,                 // image_slice_pitch
	void*,                  // host_ptr
	cl_int*);               // errcode_ret
 
static pf_create_image_t genuine_clCreateImage;
static pf_create_image2d_t genuine_clCreateImage2D;
static pf_create_image3d_t genuine_clCreateImage3D;

cl_mem (*clCreateImage)(
	cl_context,             // context
	cl_mem_flags,           // flags
	const cl_image_format*, // image_format
	const cl_image_desc*,   // image_desc
	void*,                  // host_ptr
	cl_int*);               // errcode_ret


static cl_mem clCreateImage_1_1(
	cl_context context,
	cl_mem_flags flags,
	const cl_image_format* image_format,
	const cl_image_desc* image_desc, 
	void* host_ptr,
	cl_int* errcode_ret) {

	if (0 == image_desc) {
		if (0 != errcode_ret)
			*errcode_ret = CL_INVALID_IMAGE_DESCRIPTOR;
		return 0;
	}

	switch (image_desc->image_type) {
	case CL_MEM_OBJECT_IMAGE2D:
		if (0 == genuine_clCreateImage2D)
			break;
		return genuine_clCreateImage2D(
			context,
			flags,
			image_format,
			image_desc->image_width,
			image_desc->image_height,
			image_desc->image_row_pitch,
			host_ptr,
			errcode_ret);

	case CL_MEM_OBJECT_IMAGE3D:
		if (0 == genuine_clCreateImage3D)
			break;
		return genuine_clCreateImage3D(
			context,
			flags,
			image_format,
			image_desc->image_width, 
			image_desc->image_height,
			image_desc->image_depth, 
			image_desc->image_row_pitch, 
			image_desc->image_slice_pitch, 
			host_ptr,
			errcode_ret);
	}

	if (0 != errcode_ret)
		*errcode_ret = CL_INVALID_OPERATION;

	return 0;
}


static cl_mem clCreateImage_1_2(
	cl_context context,
	cl_mem_flags flags,
	const cl_image_format* image_format,
	const cl_image_desc* image_desc, 
	void* host_ptr,
	cl_int* errcode_ret) {

	if (0 != genuine_clCreateImage)
		return genuine_clCreateImage(
			context,
			flags,
			image_format,
			image_desc,
			host_ptr,
			errcode_ret);

	if (0 != errcode_ret)
		*errcode_ret = CL_INVALID_OPERATION;

	return 0;
}


bool init(
	const ocl_ver version) {

	if (version >= ocl_ver(1, 2)) {
		genuine_clCreateImage = reinterpret_cast< pf_create_image_t >(
			getProcAddress("clCreateImage"));
		clCreateImage = clCreateImage_1_2;
	}
	else {
		genuine_clCreateImage2D = reinterpret_cast< pf_create_image2d_t >(
			getProcAddress("clCreateImage2D"));
		genuine_clCreateImage3D = reinterpret_cast< pf_create_image3d_t >(
			getProcAddress("clCreateImage3D"));
		clCreateImage = clCreateImage_1_1;
	}

	return 0 != clCreateImage;
}

} // namespace clwrap

