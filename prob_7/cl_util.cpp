#include <CL/cl.h>
#if CL_VERSION_1_2 == 0
	#error required CL_VERSION_1_2 to compile
#endif
#include <sstream>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "scoped.hpp"
#include "cl_util.hpp"
#include "stream.hpp"

// verify iostream-free status
#if _GLIBCXX_IOSTREAM
#error rogue iostream acquired
#endif

namespace testbed
{

template < typename T >
class generic_free
{
public:
	void operator()(T* arg)
	{
		assert(0 != arg);
		std::free(arg);
	}
};

} // namespace testbed

namespace align
{

class ostream
{
	stream::out& m_str;
	size_t m_acc_len;

	ostream();

public:
	ostream(
		stream::out& str)
	: m_str(str)
	, m_acc_len(0)
	{}

	stream::out& operator()() const
	{
		return m_str;
	}

	size_t get_acc_len() const
	{
		return m_acc_len;
	}

	void set_acc_len(
		const size_t len)
	{
		m_acc_len = len;
	}
};


class aligner
{
	const size_t m;

	aligner();

public:
	aligner(
		const size_t a)
	: m(a)
	{
		assert(0 != m);
	}

	size_t operator ()() const
	{
		return m;
	}
};


template < typename T >
static ostream&
operator <<(
	ostream& str,
	const T& a)
{
	std::ostringstream s;
	s << a;
	const std::string& ss = s.str();

	const size_t len = str.get_acc_len() + ss.size();
	str.set_acc_len(len);

	str() << ss;

	return str;
}


template <>
ostream&
operator <<(
	ostream& str,
	const aligner& align)
{
	const size_t pos = str.get_acc_len() + 1;
	const size_t advance = (pos + align() - 1) / align() * align() - pos;

	for (size_t i = 0; i < advance; ++i)
		str() << ' ';

	str.set_acc_len(pos + advance - 1);

	return str;
}


struct endl {
};


ostream&
operator <<(
	ostream& str,
	const endl&)
{
	str() << '\n';
	str.set_acc_len(0);

	return str;
}

} // namespace align

namespace clutil
{

using testbed::scoped_ptr;
using testbed::generic_free;

using align::aligner;

// translate a CL error code to string
// routine written against CL_VERSION_1_2

static const char*
clerror_string(
	const cl_int code)
{
	switch (code)
	{
	case CL_SUCCESS:
		return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:
		return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:
		return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:
		return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:
		return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:
		return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:
		return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:
		return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:
		return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:
		return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case CL_COMPILE_PROGRAM_FAILURE:
		return "CL_COMPILE_PROGRAM_FAILURE";
	case CL_LINKER_NOT_AVAILABLE:
		return "CL_LINKER_NOT_AVAILABLE";
	case CL_LINK_PROGRAM_FAILURE:
		return "CL_LINK_PROGRAM_FAILURE";
	case CL_DEVICE_PARTITION_FAILED:
		return "CL_DEVICE_PARTITION_FAILED";
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
		return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
	case CL_INVALID_VALUE:
		return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:
		return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:
		return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:
		return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:
		return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:
		return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:
		return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:
		return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:
		return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:
		return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:
		return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:
		return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:
		return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:
		return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:
		return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:
		return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:
		return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:
		return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:
		return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:
		return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:
		return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:
		return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:
		return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:
		return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:
		return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:
		return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:
		return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:
		return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:
		return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:
		return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:
		return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:
		return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:
		return "CL_INVALID_GLOBAL_WORK_SIZE";
	case CL_INVALID_PROPERTY:
		return "CL_INVALID_PROPERTY";
	case CL_INVALID_IMAGE_DESCRIPTOR:
		return "CL_INVALID_IMAGE_DESCRIPTOR";
	case CL_INVALID_COMPILER_OPTIONS:
		return "CL_INVALID_COMPILER_OPTIONS";
	case CL_INVALID_LINKER_OPTIONS:
		return "CL_INVALID_LINKER_OPTIONS";
	case CL_INVALID_DEVICE_PARTITION_COUNT:
		return "CL_INVALID_DEVICE_PARTITION_COUNT";
	}

	return "unknown";
}

// translate a platform info code to string
// routine written against CL_VERSION_1_2

static const char*
clplatform_info_string(
	const cl_platform_info info)
{
	switch (info)
	{
	case CL_PLATFORM_PROFILE:
		return "CL_PLATFORM_PROFILE";
	case CL_PLATFORM_VERSION:
		return "CL_PLATFORM_VERSION";
	case CL_PLATFORM_NAME:
		return "CL_PLATFORM_NAME";
	case CL_PLATFORM_VENDOR:
		return "CL_PLATFORM_VENDOR";
	case CL_PLATFORM_EXTENSIONS:
		return "CL_PLATFORM_EXTENSIONS";
	}

	return "unknown";
}

// translate a device info code to string
// routine written against CL_VERSION_1_2

static const char*
cldevice_info_string(
	const cl_device_info info)
{
	switch (info)
	{
	case CL_DEVICE_TYPE:
		return "CL_DEVICE_TYPE";
	case CL_DEVICE_VENDOR_ID:
		return "CL_DEVICE_VENDOR_ID";
	case CL_DEVICE_MAX_COMPUTE_UNITS:
		return "CL_DEVICE_MAX_COMPUTE_UNITS";
	case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
		return "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS";
	case CL_DEVICE_MAX_WORK_GROUP_SIZE:
		return "CL_DEVICE_MAX_WORK_GROUP_SIZE";
	case CL_DEVICE_MAX_WORK_ITEM_SIZES:
		return "CL_DEVICE_MAX_WORK_ITEM_SIZES";
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
		return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR";
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
		return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT";
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
		return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT";
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
		return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG";
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
		return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT";
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
		return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE";
	case CL_DEVICE_MAX_CLOCK_FREQUENCY:
		return "CL_DEVICE_MAX_CLOCK_FREQUENCY";
	case CL_DEVICE_ADDRESS_BITS:
		return "CL_DEVICE_ADDRESS_BITS";
	case CL_DEVICE_MAX_READ_IMAGE_ARGS:
		return "CL_DEVICE_MAX_READ_IMAGE_ARGS";
	case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
		return "CL_DEVICE_MAX_WRITE_IMAGE_ARGS";
	case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
		return "CL_DEVICE_MAX_MEM_ALLOC_SIZE";
	case CL_DEVICE_IMAGE2D_MAX_WIDTH:
		return "CL_DEVICE_IMAGE2D_MAX_WIDTH";
	case CL_DEVICE_IMAGE2D_MAX_HEIGHT:
		return "CL_DEVICE_IMAGE2D_MAX_HEIGHT";
	case CL_DEVICE_IMAGE3D_MAX_WIDTH:
		return "CL_DEVICE_IMAGE3D_MAX_WIDTH";
	case CL_DEVICE_IMAGE3D_MAX_HEIGHT:
		return "CL_DEVICE_IMAGE3D_MAX_HEIGHT";
	case CL_DEVICE_IMAGE3D_MAX_DEPTH:
		return "CL_DEVICE_IMAGE3D_MAX_DEPTH";
	case CL_DEVICE_IMAGE_SUPPORT:
		return "CL_DEVICE_IMAGE_SUPPORT";
	case CL_DEVICE_MAX_PARAMETER_SIZE:
		return "CL_DEVICE_MAX_PARAMETER_SIZE";
	case CL_DEVICE_MAX_SAMPLERS:
		return "CL_DEVICE_MAX_SAMPLERS";
	case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
		return "CL_DEVICE_MEM_BASE_ADDR_ALIGN";
	case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:
		return "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE";
	case CL_DEVICE_SINGLE_FP_CONFIG:
		return "CL_DEVICE_SINGLE_FP_CONFIG";
	case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
		return "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE";
	case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:
		return "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE";
	case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:
		return "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE";
	case CL_DEVICE_GLOBAL_MEM_SIZE:
		return "CL_DEVICE_GLOBAL_MEM_SIZE";
	case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:
		return "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE";
	case CL_DEVICE_MAX_CONSTANT_ARGS:
		return "CL_DEVICE_MAX_CONSTANT_ARGS";
	case CL_DEVICE_LOCAL_MEM_TYPE:
		return "CL_DEVICE_LOCAL_MEM_TYPE";
	case CL_DEVICE_LOCAL_MEM_SIZE:
		return "CL_DEVICE_LOCAL_MEM_SIZE";
	case CL_DEVICE_ERROR_CORRECTION_SUPPORT:
		return "CL_DEVICE_ERROR_CORRECTION_SUPPORT";
	case CL_DEVICE_PROFILING_TIMER_RESOLUTION:
		return "CL_DEVICE_PROFILING_TIMER_RESOLUTION";
	case CL_DEVICE_ENDIAN_LITTLE:
		return "CL_DEVICE_ENDIAN_LITTLE";
	case CL_DEVICE_AVAILABLE:
		return "CL_DEVICE_AVAILABLE";
	case CL_DEVICE_COMPILER_AVAILABLE:
		return "CL_DEVICE_COMPILER_AVAILABLE";
	case CL_DEVICE_EXECUTION_CAPABILITIES:
		return "CL_DEVICE_EXECUTION_CAPABILITIES";
	case CL_DEVICE_QUEUE_PROPERTIES:
		return "CL_DEVICE_QUEUE_PROPERTIES";
	case CL_DEVICE_NAME:
		return "CL_DEVICE_NAME";
	case CL_DEVICE_VENDOR:
		return "CL_DEVICE_VENDOR";
	case CL_DRIVER_VERSION:
		return "CL_DRIVER_VERSION";
	case CL_DEVICE_PROFILE:
		return "CL_DEVICE_PROFILE";
	case CL_DEVICE_VERSION:
		return "CL_DEVICE_VERSION";
	case CL_DEVICE_EXTENSIONS:
		return "CL_DEVICE_EXTENSIONS";
	case CL_DEVICE_PLATFORM:
		return "CL_DEVICE_PLATFORM";
	case CL_DEVICE_DOUBLE_FP_CONFIG:
		return "CL_DEVICE_DOUBLE_FP_CONFIG";
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:
		return "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF";
	case CL_DEVICE_HOST_UNIFIED_MEMORY:
		return "CL_DEVICE_HOST_UNIFIED_MEMORY";
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:
		return "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR";
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:
		return "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT";
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:
		return "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT";
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:
		return "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG";
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:
		return "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT";
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:
		return "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE";
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:
		return "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF";
	case CL_DEVICE_OPENCL_C_VERSION:
		return "CL_DEVICE_OPENCL_C_VERSION";
	case CL_DEVICE_LINKER_AVAILABLE:
		return "CL_DEVICE_LINKER_AVAILABLE";
	case CL_DEVICE_BUILT_IN_KERNELS:
		return "CL_DEVICE_BUILT_IN_KERNELS";
	case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:
		return "CL_DEVICE_IMAGE_MAX_BUFFER_SIZE";
	case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE:
		return "CL_DEVICE_IMAGE_MAX_ARRAY_SIZE";
	case CL_DEVICE_PARENT_DEVICE:
		return "CL_DEVICE_PARENT_DEVICE";
	case CL_DEVICE_PARTITION_MAX_SUB_DEVICES:
		return "CL_DEVICE_PARTITION_MAX_SUB_DEVICES";
	case CL_DEVICE_PARTITION_PROPERTIES:
		return "CL_DEVICE_PARTITION_PROPERTIES";
	case CL_DEVICE_PARTITION_AFFINITY_DOMAIN:
		return "CL_DEVICE_PARTITION_AFFINITY_DOMAIN";
	case CL_DEVICE_PARTITION_TYPE:
		return "CL_DEVICE_PARTITION_TYPE";
	case CL_DEVICE_REFERENCE_COUNT:
		return "CL_DEVICE_REFERENCE_COUNT";
	case CL_DEVICE_PREFERRED_INTEROP_USER_SYNC:
		return "CL_DEVICE_PREFERRED_INTEROP_USER_SYNC";
	case CL_DEVICE_PRINTF_BUFFER_SIZE:
		return "CL_DEVICE_PRINTF_BUFFER_SIZE";
	}

	return "unknown";
}

// translate a singular device type code to string
// routine written against CL_VERSION_1_2

const char*
cldevice_type_single_string(
	const cl_device_type type)
{
	if (CL_DEVICE_TYPE_ALL == type)
		return "CL_DEVICE_TYPE_ALL";

	assert(0 == (type & type - 1));

	switch (type)
	{
	case CL_DEVICE_TYPE_DEFAULT:
		return "CL_DEVICE_TYPE_DEFAULT";
	case CL_DEVICE_TYPE_CPU:
		return "CL_DEVICE_TYPE_CPU";
	case CL_DEVICE_TYPE_GPU:
		return "CL_DEVICE_TYPE_GPU";
	case CL_DEVICE_TYPE_ACCELERATOR:
		return "CL_DEVICE_TYPE_ACCELERATOR";
	case CL_DEVICE_TYPE_CUSTOM:
		return "CL_DEVICE_TYPE_CUSTOM";
	}

	return "unknown";
}

// translate a device type code to string
// routine written against CL_VERSION_1_2

static std::string
cldevice_type_string(
	cl_device_type type)
{
	if (CL_DEVICE_TYPE_ALL == type)
		return cldevice_type_single_string(type);

	std::string multi;

	while (0 != (type & type - 1))
	{
		cl_device_type dropped_bit = type;
		type &= type - 1;
		dropped_bit ^= dropped_bit & type;

		multi += cldevice_type_single_string(dropped_bit);
		multi += ' ';
	}

	return multi + cldevice_type_single_string(type);
}

// translate a singular device fp-config code to string
// routine written against CL_VERSION_1_2

static const char*
cldevice_fpconf_single_string(
	const cl_device_fp_config config)
{
	assert(0 == (config & config - 1));

	if (0 == config)
		return "none";

	switch (config)
	{
	case CL_FP_DENORM:
		return "CL_FP_DENORM";
	case CL_FP_INF_NAN:
		return "CL_FP_INF_NAN";
	case CL_FP_ROUND_TO_NEAREST:
		return "CL_FP_ROUND_TO_NEAREST";
	case CL_FP_ROUND_TO_ZERO:
		return "CL_FP_ROUND_TO_ZERO";
	case CL_FP_ROUND_TO_INF:
		return "CL_FP_ROUND_TO_INF";
	case CL_FP_FMA:
		return "CL_FP_FMA";
	case CL_FP_SOFT_FLOAT:
		return "CL_FP_SOFT_FLOAT";
	case CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT:
		return "CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT";
	}

	return "unknown";
}

// translate a device fp-config code to string
// routine written against CL_VERSION_1_2

static std::string
cldevice_fpconf_string(
	cl_device_fp_config config)
{
	std::string multi;

	while (0 != (config & config - 1))
	{
		cl_device_fp_config dropped_bit = config;
		config &= config - 1;
		dropped_bit ^= dropped_bit & config;

		multi += cldevice_fpconf_single_string(dropped_bit);
		multi += ' ';
	}

	return multi + cldevice_fpconf_single_string(config);
}

// translate a singular device execution capability code to string
// routine written against CL_VERSION_1_2

static const char*
cldevice_exec_single_string(
	const cl_device_exec_capabilities capability)
{
	assert(0 == (capability & capability - 1));

	switch (capability)
	{
	case CL_EXEC_KERNEL:
		return "CL_EXEC_KERNEL";
	case CL_EXEC_NATIVE_KERNEL:
		return "CL_EXEC_NATIVE_KERNEL";
	}

	return "unknown";
}

// translate device execution capabilities to string
// routine written against CL_VERSION_1_2

static std::string
cldevice_exec_string(
	cl_device_exec_capabilities capabilities)
{
	std::string multi;

	while (0 != (capabilities & capabilities - 1))
	{
		cl_device_exec_capabilities dropped_bit = capabilities;
		capabilities &= capabilities - 1;
		dropped_bit ^= dropped_bit & capabilities;

		multi += cldevice_exec_single_string(dropped_bit);
		multi += ' ';
	}

	return multi + cldevice_exec_single_string(capabilities);
}

// translate a singular device command-queue property code to string
// routine written against CL_VERSION_1_2

static const char*
cldevice_queue_single_string(
	const cl_command_queue_properties property)
{
	assert(0 == (property & property - 1));

	switch (property)
	{
	case CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE:
		return "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE";
	case CL_QUEUE_PROFILING_ENABLE:
		return "CL_QUEUE_PROFILING_ENABLE";
	}

	return "unknown";
}

// translate device command-queue properties to string
// routine written against CL_VERSION_1_2

static std::string
cldevice_queue_string(
	cl_command_queue_properties properties)
{
	std::string multi;

	while (0 != (properties & properties - 1))
	{
		cl_command_queue_properties dropped_bit = properties;
		properties &= properties - 1;
		dropped_bit ^= dropped_bit & properties;

		multi += cldevice_queue_single_string(dropped_bit);
		multi += ' ';
	}

	return multi + cldevice_queue_single_string(properties);
}


// translate a device local-mem-type code to string
// routine written agains CL_VERSION_1_2

static const char*
cldevice_mem_type_string(
	const cl_device_local_mem_type type)
{
	switch (type)
	{
	case CL_NONE:
		return "CL_NONE";
	case CL_LOCAL:
		return "CL_LOCAL";
	case CL_GLOBAL:
		return "CL_GLOBAL";
	}

	return "unknown";
}

// translate a device cache-type code to string
// routine written against CL_VERSION_1_2

static const char*
cldevice_cache_type_string(
	const cl_device_mem_cache_type type)
{
	switch (type)
	{
	case CL_NONE:
		return "CL_NONE";
	case CL_READ_ONLY_CACHE:
		return "CL_READ_ONLY_CACHE";
	case CL_READ_WRITE_CACHE:
		return "CL_READ_WRITE_CACHE";
	}

	return "unknown";
}

// translate a device partition property code to string
// routine written against CL_VERSION_1_2

static const char*
clpartition_property_string(
	const cl_device_partition_property property)
{
	if (0 == property)
		return "none";

	switch (property)
	{
	case CL_DEVICE_PARTITION_EQUALLY:
		return "CL_DEVICE_PARTITION_EQUALLY";
	case CL_DEVICE_PARTITION_BY_COUNTS:
		return "CL_DEVICE_PARTITION_BY_COUNTS";
	case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
		return "CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN";
	}

	return "unknown";
}

// translate a singular device affinity to string
// routine written against CL_VERSION_1_2

static const char*
clpartition_affinity_single_string(
	const cl_device_affinity_domain domain)
{
	assert(0 == (domain & domain - 1));

	if (0 == domain)
		return "none";

	switch (domain)
	{
	case CL_DEVICE_AFFINITY_DOMAIN_NUMA:
		return "CL_DEVICE_AFFINITY_DOMAIN_NUMA";
	case CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE:
		return "CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE";
	case CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE:
		return "CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE";
	case CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE:
		return "CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE";
	case CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE:
		return "CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE";
	case CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE:
		return "CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE";
	}

	return "unknown";
}

// translate device affinity domains to string
// routine written against CL_VERSION_1_2

static std::string
clpartition_affinity_string(
	cl_device_affinity_domain domain)
{
	std::string multi;

	while (0 != (domain & domain - 1))
	{
		cl_device_affinity_domain dropped_bit = domain;
		domain &= domain - 1;
		dropped_bit ^= dropped_bit & domain;

		multi += clpartition_affinity_single_string(dropped_bit);
		multi += ' ';
	}

	return multi + clpartition_affinity_single_string(domain);
}


bool
reportCLError(
	const cl_int code,
	stream::out& out)
{
	if (CL_SUCCESS == code)
		return false;

	out << "error: " << clerror_string(code) << " (" << code << ")\n";

	return true;
}

namespace platform_info
{

const cl_platform_info param_name_1_0[] =
{
	CL_PLATFORM_PROFILE,
	CL_PLATFORM_VERSION,
	CL_PLATFORM_NAME,
	CL_PLATFORM_VENDOR,
	CL_PLATFORM_EXTENSIONS
};


static void
print_single_info(
	align::ostream& out,
	const cl_platform_info param_name,
	const void* const buffer,
	const size_t /*buffer_size*/)
{
	switch (param_name)
	{
	case CL_PLATFORM_PROFILE:
	case CL_PLATFORM_VERSION:
	case CL_PLATFORM_NAME:
	case CL_PLATFORM_VENDOR:
	case CL_PLATFORM_EXTENSIONS:
		if (0 != buffer && '\0' != *reinterpret_cast< const char* >(buffer))
			out << reinterpret_cast< const char* >(buffer);
		else
			out << "empty";
		break;
	}

	out << align::endl();
}

} // namespace platform_info

namespace device_info
{

const cl_device_info param_name_1_0[] =
{
	CL_DEVICE_TYPE,
	CL_DEVICE_VENDOR_ID,
	CL_DEVICE_MAX_COMPUTE_UNITS,
	CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
	CL_DEVICE_MAX_WORK_GROUP_SIZE,
	CL_DEVICE_MAX_WORK_ITEM_SIZES,
	CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
	CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
	CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
	CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
	CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
	CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
	CL_DEVICE_MAX_CLOCK_FREQUENCY,
	CL_DEVICE_ADDRESS_BITS,
	CL_DEVICE_MAX_READ_IMAGE_ARGS,
	CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
	CL_DEVICE_MAX_MEM_ALLOC_SIZE,
	CL_DEVICE_IMAGE2D_MAX_WIDTH,
	CL_DEVICE_IMAGE2D_MAX_HEIGHT,
	CL_DEVICE_IMAGE3D_MAX_WIDTH,
	CL_DEVICE_IMAGE3D_MAX_HEIGHT,
	CL_DEVICE_IMAGE3D_MAX_DEPTH,
	CL_DEVICE_IMAGE_SUPPORT,
	CL_DEVICE_MAX_PARAMETER_SIZE,
	CL_DEVICE_MAX_SAMPLERS,
	CL_DEVICE_MEM_BASE_ADDR_ALIGN,
	CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
	CL_DEVICE_SINGLE_FP_CONFIG,
	CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
	CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
	CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
	CL_DEVICE_GLOBAL_MEM_SIZE,
	CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
	CL_DEVICE_MAX_CONSTANT_ARGS,
	CL_DEVICE_LOCAL_MEM_TYPE,
	CL_DEVICE_LOCAL_MEM_SIZE,
	CL_DEVICE_ERROR_CORRECTION_SUPPORT,
	CL_DEVICE_PROFILING_TIMER_RESOLUTION,
	CL_DEVICE_ENDIAN_LITTLE,
	CL_DEVICE_AVAILABLE,
	CL_DEVICE_COMPILER_AVAILABLE,
	CL_DEVICE_EXECUTION_CAPABILITIES,
	CL_DEVICE_QUEUE_PROPERTIES,
	CL_DEVICE_NAME,
	CL_DEVICE_VENDOR,
	CL_DRIVER_VERSION,
	CL_DEVICE_PROFILE,
	CL_DEVICE_VERSION,
	CL_DEVICE_EXTENSIONS,
	CL_DEVICE_PLATFORM
};

const cl_device_info param_name_1_1[] =
{
	CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
	CL_DEVICE_HOST_UNIFIED_MEMORY,
	CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
	CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
	CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
	CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
	CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
	CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
	CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
	CL_DEVICE_OPENCL_C_VERSION
};

const cl_device_info param_name_1_2[] =
{
	CL_DEVICE_DOUBLE_FP_CONFIG,
	CL_DEVICE_LINKER_AVAILABLE,
	CL_DEVICE_BUILT_IN_KERNELS,
	CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
	CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,
	CL_DEVICE_PARENT_DEVICE,
	CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
	CL_DEVICE_PARTITION_PROPERTIES,
	CL_DEVICE_PARTITION_AFFINITY_DOMAIN,
	CL_DEVICE_PARTITION_TYPE,
	CL_DEVICE_REFERENCE_COUNT,
	CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,
	CL_DEVICE_PRINTF_BUFFER_SIZE
};


static void
print_single_info(
	align::ostream& out,
	const cl_device_info param_name,
	const void* const buffer,
	const size_t buffer_size)
{
	switch (param_name)
	{
	case CL_DEVICE_TYPE:
		out << cldevice_type_string(*reinterpret_cast< const cl_device_type* >(buffer));
		break;
	case CL_DEVICE_LOCAL_MEM_TYPE:
		out << cldevice_mem_type_string(*reinterpret_cast< const cl_device_local_mem_type* >(buffer));
		break;
	case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
		out << cldevice_cache_type_string(*reinterpret_cast< const cl_device_mem_cache_type* >(buffer));
		break;
	case CL_DEVICE_SINGLE_FP_CONFIG:
	case CL_DEVICE_DOUBLE_FP_CONFIG:
		out << cldevice_fpconf_string(*reinterpret_cast< const cl_device_fp_config* >(buffer));
		break;
	case CL_DEVICE_EXECUTION_CAPABILITIES:
		out << cldevice_exec_string(*reinterpret_cast< const cl_device_exec_capabilities* >(buffer));
		break;
	case CL_DEVICE_QUEUE_PROPERTIES:
		out << cldevice_queue_string(*reinterpret_cast< const cl_command_queue_properties* >(buffer));
		break;
	case CL_DEVICE_IMAGE_SUPPORT:
	case CL_DEVICE_ERROR_CORRECTION_SUPPORT:
	case CL_DEVICE_HOST_UNIFIED_MEMORY:
	case CL_DEVICE_ENDIAN_LITTLE:
	case CL_DEVICE_AVAILABLE:
	case CL_DEVICE_COMPILER_AVAILABLE:
	case CL_DEVICE_LINKER_AVAILABLE:
	case CL_DEVICE_PREFERRED_INTEROP_USER_SYNC:
		out << (CL_TRUE == *reinterpret_cast< const cl_bool* >(buffer) ? "true" : "false");
		break;
	case CL_DEVICE_MAX_WORK_GROUP_SIZE:
	case CL_DEVICE_IMAGE2D_MAX_WIDTH:
	case CL_DEVICE_IMAGE2D_MAX_HEIGHT:
	case CL_DEVICE_IMAGE3D_MAX_WIDTH:
	case CL_DEVICE_IMAGE3D_MAX_HEIGHT:
	case CL_DEVICE_IMAGE3D_MAX_DEPTH:
	case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:
	case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE:
	case CL_DEVICE_MAX_PARAMETER_SIZE:
	case CL_DEVICE_PROFILING_TIMER_RESOLUTION:
	case CL_DEVICE_PRINTF_BUFFER_SIZE:
		out << *reinterpret_cast< const size_t* >(buffer);
		break;
	case CL_DEVICE_MAX_WORK_ITEM_SIZES:
		out << *reinterpret_cast< const size_t* >(buffer);
		for (size_t i = 1; i < buffer_size / sizeof(size_t); ++i)
			out << ' ' << reinterpret_cast< const size_t* >(buffer)[i];
		break;
	case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
	case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:
	case CL_DEVICE_GLOBAL_MEM_SIZE:
	case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:
	case CL_DEVICE_LOCAL_MEM_SIZE:
		out << *reinterpret_cast< const cl_ulong* >(buffer);
		break;
	case CL_DEVICE_VENDOR_ID:
	case CL_DEVICE_MAX_COMPUTE_UNITS:
	case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:
	case CL_DEVICE_MAX_CLOCK_FREQUENCY:
	case CL_DEVICE_ADDRESS_BITS:
	case CL_DEVICE_MAX_READ_IMAGE_ARGS:
	case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
	case CL_DEVICE_MAX_SAMPLERS:
	case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
	case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:
	case CL_DEVICE_MAX_CONSTANT_ARGS:
	case CL_DEVICE_PARTITION_MAX_SUB_DEVICES:
	case CL_DEVICE_REFERENCE_COUNT:
	case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:
		out << *reinterpret_cast< const cl_uint* >(buffer);
		break;
	case CL_DEVICE_PLATFORM:
		out << *reinterpret_cast< const cl_platform_id* >(buffer);
		break;
	case CL_DEVICE_PARENT_DEVICE:
		out << *reinterpret_cast< const cl_device_id* >(buffer);
		break;
	case CL_DEVICE_PARTITION_PROPERTIES:
		out << clpartition_property_string(
			*reinterpret_cast< const cl_device_partition_property* >(buffer));
		for (size_t i = 1; i < buffer_size / sizeof(cl_device_partition_property); ++i)
			out << ' ' << clpartition_property_string(
				reinterpret_cast< const cl_device_partition_property* >(buffer)[i]);
		break;
	case CL_DEVICE_PARTITION_AFFINITY_DOMAIN:
		out << clpartition_affinity_string(*reinterpret_cast< const cl_device_affinity_domain* >(buffer));
		break;
	case CL_DEVICE_PARTITION_TYPE:
		if (0 == buffer_size ||
			0 != buffer &&
			0 == *reinterpret_cast< const cl_device_partition_property* >(buffer))
		{
			out << "n/a";
		}
		else
			out << "expected a parent device, got something else; ignore";
		break;
	case CL_DEVICE_NAME:
	case CL_DEVICE_VENDOR:
	case CL_DRIVER_VERSION:
	case CL_DEVICE_PROFILE:
	case CL_DEVICE_VERSION:
	case CL_DEVICE_OPENCL_C_VERSION:
	case CL_DEVICE_EXTENSIONS:
	case CL_DEVICE_BUILT_IN_KERNELS:
		if (0 != buffer && '\0' != *reinterpret_cast< const char* >(buffer))
			out << reinterpret_cast< const char* >(buffer);
		else
			out << "empty";
		break;
	}

	out << align::endl();
}

} // namespace device_info

static void
print_info(
	align::ostream& out,
	const cl_platform_id id,
	const cl_platform_info* const param_name,
	const size_t param_count,
	const size_t cascade_alignment,
	const size_t value_alignment)
{
	const size_t buffer_size = 2 << 10;
	const scoped_ptr< void, generic_free > buffer(std::malloc(buffer_size));

	for (size_t i = 0; i < param_count; ++i)
	{
		size_t ret_size = 0;
		const cl_int success = clGetPlatformInfo(
			id,
			param_name[i],
			buffer_size,
			buffer(),
			&ret_size);

		const char* const param_name_string = clplatform_info_string(param_name[i]);

		if (reportCLError(success))
		{
			stream::cerr << "failure at clGetPlatformInfo, param_name: " <<
				param_name_string << "; ignore\n";
			continue;
		}

		out << aligner(cascade_alignment) <<
			param_name_string << ':' << aligner(value_alignment);

		platform_info::print_single_info(
			out,
			param_name[i],
			buffer(),
			ret_size);
	}
}


static void
print_info(
	align::ostream& out,
	const cl_device_id id,
	const cl_device_info* const param_name,
	const size_t param_count,
	const size_t cascade_alignment,
	const size_t value_alignment)
{
	const size_t buffer_size = 2 << 10;
	const scoped_ptr< void, generic_free > buffer(std::malloc(buffer_size));

	for (size_t i = 0; i < param_count; ++i)
	{
		size_t ret_size = 0;
		const cl_int success = clGetDeviceInfo(
			id,
			param_name[i],
			buffer_size,
			buffer(),
			&ret_size);

		const char* const param_name_string = cldevice_info_string(param_name[i]);

		if (reportCLError(success))
		{
			stream::cerr << "failure at clGetDeviceInfo, param_name: " <<
				param_name_string << "; ignore\n";
			continue;
		}

		out << aligner(cascade_alignment * 2 - 1) <<
			param_name_string << ':' << aligner(value_alignment);

		device_info::print_single_info(
			out,
			param_name[i],
			buffer(),
			ret_size);
	}
}


bool
clplatform_version(
	const cl_platform_id id,
	ocl_ver& version)
{
	char platform_version[1024];

	const cl_int success = clGetPlatformInfo(
		id,
		CL_PLATFORM_VERSION,
		sizeof(platform_version),
		platform_version,
		0);

	if (reportCLError(success))
	{
		stream::cerr << "failure at clGetPlatformInfo, param_name: " <<
			clplatform_info_string(CL_PLATFORM_VERSION) << '\n';
		return false;
	}

	cl_uint major, minor;
	if (2 != std::sscanf(platform_version, "OpenCL %u.%u", &major, &minor))
	{
		stream::cerr << "failure at parsing platform version\n";
		return false;
	}

	version = ocl_ver(major, minor);
	return true;
}


bool
cldevice_version(
	const cl_device_id id,
	ocl_ver& version)
{
	char device_version[1024];

	const cl_int success = clGetDeviceInfo(
		id,
		CL_DEVICE_VERSION,
		sizeof(device_version),
		device_version,
		0);

	if (reportCLError(success))
	{
		stream::cerr << "failure at clGetDeviceInfo, param_name: " <<
			cldevice_info_string(CL_DEVICE_VERSION) << '\n';
		return false;
	}

	cl_uint major, minor;
	if (2 != std::sscanf(device_version, "OpenCL %u.%u", &major, &minor))
	{
		stream::cerr << "failure at parsing device version\n";
		return false;
	}

	version = ocl_ver(major, minor);
	return true;
}


int
reportCLCaps(
	const bool discard_platform_version,
	const bool discard_device_version,
	const size_t cascade_alignment,
	const size_t value_alignment)
{
	cl_uint num_platforms = 0;

	const cl_int successA = clGetPlatformIDs(
		num_platforms, 0, &num_platforms);

	if (reportCLError(successA))
	{
		stream::cerr << "failure at clGetPlatformIDs; terminate\n";
		return -1;
	}

	const scoped_ptr< cl_platform_id, generic_free > platform(
		(cl_platform_id*) std::malloc(sizeof(cl_platform_id) * num_platforms));

	const cl_int successB = clGetPlatformIDs(
		num_platforms, platform(), 0);

	if (reportCLError(successB))
	{
		stream::cerr << "failure at clGetPlatformIDs; terminate\n";
		return -1;
	}

	align::ostream cout(stream::cout);

	for (cl_uint i = 0; i < num_platforms; ++i)
	{
		ocl_ver platform_version;

		if (!discard_platform_version &&
			!clplatform_version(platform()[i], platform_version))
		{
			stream::cerr << "failure at getting version of platform " <<
				i << "; ignore\n";
			continue;
		}

		cout << "platform " << i << ':' <<
			aligner(value_alignment) << platform()[i] << align::endl();

		print_info(
			cout,
			platform()[i],
			platform_info::param_name_1_0,
			sizeof(platform_info::param_name_1_0) / sizeof(platform_info::param_name_1_0[0]),
			cascade_alignment,
			value_alignment);

		if (!discard_platform_version && ocl_ver(1, 0) == platform_version)
			goto show_devices;

		// TODO: new versions go here

show_devices:

		cl_uint num_devices = 0;

		const cl_int successA = clGetDeviceIDs(
			platform()[i],
			CL_DEVICE_TYPE_ALL,
			num_devices,
			0,
			&num_devices);

		if (reportCLError(successA))
		{
			stream::cerr << "failure at clGetDeviceIDs; terminate\n";
			return -1;
		}

		const scoped_ptr< cl_device_id, generic_free > device(
			(cl_device_id*) std::malloc(sizeof(cl_device_id) * num_devices));

		const cl_int successB = clGetDeviceIDs(
			platform()[i],
			CL_DEVICE_TYPE_ALL,
			num_devices,
			device(),
			0);

		if (reportCLError(successB))
		{
			stream::cerr << "failure at clGetDeviceIDs; terminate\n";
			return -1;
		}

		for (cl_uint j = 0; j < num_devices; ++j)
		{
			ocl_ver device_version;

			if (!discard_device_version &&
				!cldevice_version(device()[j], device_version))
			{
				stream::cerr << "failure at getting version of device " <<
					j << "; ignore\n";
				continue;
			}

			cout << aligner(cascade_alignment) << "device " << j << ':' <<
				aligner(value_alignment) << device()[j] << align::endl();

			print_info(
				cout,
				device()[j],
				device_info::param_name_1_0,
				sizeof(device_info::param_name_1_0) / sizeof(device_info::param_name_1_0[0]),
				cascade_alignment,
				value_alignment);

			if (!discard_device_version && ocl_ver(1, 0) == device_version)
				continue;

			print_info(
				cout,
				device()[j],
				device_info::param_name_1_1,
				sizeof(device_info::param_name_1_1) / sizeof(device_info::param_name_1_1[0]),
				cascade_alignment,
				value_alignment);

			if (!discard_device_version && ocl_ver(1, 1) == device_version)
				continue;

			print_info(
				cout,
				device()[j],
				device_info::param_name_1_2,
				sizeof(device_info::param_name_1_2) / sizeof(device_info::param_name_1_2[0]),
				cascade_alignment,
				value_alignment);

			if (!discard_device_version && ocl_ver(1, 2) == device_version)
				continue;

			// TODO: new versions go here
		}
	}

	return 0;
}

} // namespace clutil
