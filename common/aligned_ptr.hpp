#ifndef aligned_ptr_H__
#define aligned_ptr_H__

#include <assert.h>
#include <stdlib.h>
#include <stdint.h>


template < typename T, size_t ALIGNMENT_T >
class aligned_ptr
{
	void* ptr;

	aligned_ptr(
		const aligned_ptr& src); // undefined

	aligned_ptr& operator =(
		const aligned_ptr& src); // undefined

public:
	aligned_ptr()
	: ptr(0)
	{
	}

	aligned_ptr(
		const size_t capacity)
	: ptr(0)
	{
		malloc(capacity);
	}

	~aligned_ptr()
	{
		free();
	}

	aligned_ptr& move(
		aligned_ptr& src)
	{
		free();

		ptr = src.ptr;
		src.ptr = 0;

		return *this;
	}

	void malloc(
		const size_t capacity)
	{
		free();

		if (0 != capacity) {
			void* const unaligned = ::malloc(sizeof(T) * capacity + ALIGNMENT_T - 1);
			ptr = reinterpret_cast< void* >(uintptr_t(unaligned) + uintptr_t(ALIGNMENT_T - 1));
		}
	}

	void free()
	{
		if (0 != ptr) {
			::free(reinterpret_cast< void* >(uintptr_t(ptr) - uintptr_t(ALIGNMENT_T - 1)));
			ptr = 0;
		}
	}

	bool is_null() const
	{
		return 0 == ptr;
	}

	operator T* () const
	{
		return reinterpret_cast< T* >(uintptr_t(ptr) & ~uintptr_t(ALIGNMENT_T - 1));
	}
};

#endif // aligned_ptr_H__

