#ifndef aligned_ptr_H__
#define aligned_ptr_H__

#include <assert.h>
#include <stdlib.h>
#include <stdint.h>


template < typename T, size_t ALIGNMENT_T >
class aligned_ptr
{
	void* unaligned;

	aligned_ptr(
		const aligned_ptr& src); // undefined

	aligned_ptr& operator =(
		const aligned_ptr& src); // undefined

public:
	aligned_ptr()
	: unaligned(0)
	{
	}

	aligned_ptr(
		const size_t capacity)
	: unaligned(0)
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

		unaligned = src.unaligned;
		src.unaligned = 0;

		return *this;
	}

	void malloc(
		const size_t capacity)
	{
		free();

		if (0 != capacity)
		{
			unaligned = ::malloc(sizeof(T) * capacity + ALIGNMENT_T - 1);

			const uintptr_t aligned = uintptr_t(unaligned) + ALIGNMENT_T - 1 & ~uintptr_t(ALIGNMENT_T - 1);
			const uintptr_t offset = aligned - uintptr_t(unaligned);

			unaligned = reinterpret_cast< void* >(aligned + offset);
		}
	}

	void free()
	{
		if (0 != unaligned)
		{
			::free(reinterpret_cast< void* >((uintptr_t(unaligned) & ~uintptr_t(ALIGNMENT_T - 1)) -
				(uintptr_t(unaligned) & uintptr_t(ALIGNMENT_T - 1))));
			unaligned = 0;
		}
	}

	bool is_null() const
	{
		return 0 == unaligned;
	}

	operator T* () const
	{
		return reinterpret_cast< T* >(uintptr_t(unaligned) & ~uintptr_t(ALIGNMENT_T - 1));
	}
};

#endif // aligned_ptr_H__

