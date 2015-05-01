#ifndef cl_util_H__
#define cl_util_H__

#ifdef __APPLE__
	#include <OpenCL/cl.h>
#else
	#include <CL/cl.h>
#endif	
#include "stream.hpp"

namespace clutil
{

class ocl_ver
{
	cl_ulong m;

public:
	ocl_ver()
	: m(0)
	{}

	ocl_ver(
		const cl_uint major,
		const cl_uint minor)
	: m(cl_ulong(major) << 32 | minor)
	{}

	// primary relational ops
	bool
	operator ==(
		const ocl_ver& oth) const;

	bool
	operator <(
		const ocl_ver& oth) const;

	// derivative relational ops
	bool
	operator !=(
		const ocl_ver& oth) const;

	bool
	operator >(
		const ocl_ver& oth) const;

	bool
	operator <=(
		const ocl_ver& oth) const;

	bool
	operator >=(
		const ocl_ver& oth) const;
};

inline bool
ocl_ver::operator ==(
	const ocl_ver& oth) const
{
	return m == oth.m;
}

inline bool
ocl_ver::operator <(
	const ocl_ver& oth) const
{
	return m < oth.m;
}

inline bool
ocl_ver::operator !=(
	const ocl_ver& oth) const
{
	return !this->operator ==(oth);
}

inline bool
ocl_ver::operator >(
	const ocl_ver& oth) const
{
	return oth.operator <(*this);
}

inline bool
ocl_ver::operator <=(
	const ocl_ver& oth) const
{
	return !oth.operator <(*this);
}

inline bool
ocl_ver::operator >=(
	const ocl_ver& oth) const
{
	return !this->operator <(oth);
}

bool
clplatform_version(
	const cl_platform_id id,
	ocl_ver& version);

bool
cldevice_version(
	const cl_device_id id,
	ocl_ver& version);

const char*
cldevice_type_single_string(
	const cl_device_type type);

bool
reportCLError(
	const cl_int code,
	stream::out& out = stream::cerr);

int
reportCLCaps(
	const bool discard_platform_version,
	const bool discard_device_version,
	const size_t cascade_alignment = 5,
	const size_t value_alignment = 64);

} // namespace clutil

#endif // cl_util_H__
