#ifndef get_proc_address_H__
#define get_proc_address_H__

#include <dlfcn.h>
#include <iostream>

#ifndef DEFAULT_LOOKUP_LIB
#define DEFAULT_LOOKUP_LIB 0
#endif

static void*
getProcAddress(
	const char* const proc_name,
	const char* const lib_name = DEFAULT_LOOKUP_LIB)
{
	void* library = dlopen(lib_name, RTLD_NOW | RTLD_GLOBAL);

	if (!library)
	{
		std::cerr << "error: " << dlerror() << std::endl;
		return 0;
	}

	void* proc = dlsym(library, proc_name);

	if (!proc)
		std::cerr << "error: " << dlerror() << std::endl;

	return proc;
}

#endif // get_proc_address_H__
