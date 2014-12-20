#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "scoped.hpp"
#include "get_file_size.hpp"

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


template <>
class scoped_functor< FILE >
{
public:

	void operator()(FILE* arg)
	{
		assert(0 != arg);
		fclose(arg);
	}
};


bool
get_file_size(
	const char* const filename,
	size_t& size)
{
	assert(0 != filename);

	struct stat filestat;

	if (-1 == stat(filename, &filestat))
	{
		std::cerr << __FUNCTION__ << " failed to stat file '" <<
			filename << "'" << std::endl;
		return false;
	}

	if (!S_ISREG(filestat.st_mode))
	{
		std::cerr << __FUNCTION__ << " encountered a non-regular file '" <<
			filename << "'" << std::endl;
		return false;
	}

	size = filestat.st_size;
	return true;
}


char*
get_buffer_from_file(
	const char* const filename,
	size_t& length)
{
	assert(0 != filename);

	if (!get_file_size(filename, length))
	{
		std::cerr << __FUNCTION__ <<
			" cannot get size of file '" << filename << "'" << std::endl;
		return 0;
	}

	const scoped_ptr< FILE, scoped_functor > file(fopen(filename, "r"));

	if (0 == file())
	{
		std::cerr << __FUNCTION__ <<
			" cannot open file '" << filename << "'" << std::endl;
		return 0;
	}

	scoped_ptr< char, generic_free > source(
		reinterpret_cast< char* >(malloc(length)));

	if (0 == source())
	{
		std::cerr << __FUNCTION__ <<
			" cannot allocate memory for file '" << filename << "'" << std::endl;
		return 0;
	}

	if (1 != fread(source(), length, 1, file()))
	{
		std::cerr << __FUNCTION__ <<
			" cannot read from file '" << filename << "'" << std::endl;
		return 0;
	}

	char* const ret = source();
	source.reset();

	return ret;
}

} // namespace testbed
