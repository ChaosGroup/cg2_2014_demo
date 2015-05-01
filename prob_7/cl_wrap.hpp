#ifndef cl_wrap_H__
#define cl_wrap_H__

#include <CL/cl.h>
#include "cl_util.hpp"

namespace clwrap {

bool
init(
	const clutil::ocl_ver);

extern cl_mem (*clCreateImage)(
	cl_context,             // context
	cl_mem_flags,           // flags
	const cl_image_format*, // image_format
	const cl_image_desc*,   // image_desc
	void*,                  // host_ptr
	cl_int*);               // errcode_ret

} // namespace clwrap

#endif // cl_wrap_H__
