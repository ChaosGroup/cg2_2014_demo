#ifndef array_H__
#define array_H__

#include <assert.h>
#include <stdlib.h>
#include <stdint.h>

#include "aligned_ptr.hpp"

template < typename ELEMENT_T >
class Array
{
	enum { alignment = 64 };

protected:
	uint32_t m_capacity;
	uint32_t m_count;
	aligned_ptr< ELEMENT_T, alignment > m_elem;

public:
	Array()
	: m_capacity(0)
	, m_count(0)
	{
	}

	Array(
		const Array& src);

	~Array();

	Array& operator =(
		const Array& src);

	Array&
	move(
		Array& src);

	bool
	setCapacity(
		const size_t n);

	size_t
	getCapacity() const
	{
		return m_capacity;
	}

	size_t
	getCount() const
	{
		return m_count;
	}

	void
	resetCount()
	{
		setCapacity(m_capacity);
	}

	bool
	addElement();

	bool
	addElement(
		const ELEMENT_T &e);

	bool
	addMultiElement(
		const size_t count);

	const ELEMENT_T&
	getElement(
		const size_t i) const;

	ELEMENT_T&
	getMutable(
		const size_t i);
};

template< typename ELEMENT_T >
inline Array< ELEMENT_T >::Array(
	const Array< ELEMENT_T >& src)
: m_capacity(0)
, m_count(0)
{
	if (0 == src.m_capacity)
		return;

	assert(!src.m_elem.is_null());

	m_elem.malloc(src.m_capacity);

	if (m_elem.is_null())
		return;

	for (size_t i = 0; i < src.m_count; ++i)
		new (m_elem + i) ELEMENT_T(src.m_elem[i]);

	m_capacity = src.m_capacity;
	m_count = src.m_count;
}

template< typename ELEMENT_T >
inline Array< ELEMENT_T >::~Array()
{
	assert(0 == m_capacity || !m_elem.is_null());

	for (size_t i = 0; i < m_count; ++i)
		m_elem[i].~ELEMENT_T();

#if !defined(NDEBUG)
	m_capacity = 0;
	m_count = 0;

#endif
}

template< typename ELEMENT_T >
inline Array< ELEMENT_T >&
Array< ELEMENT_T >::operator =(
	const Array< ELEMENT_T>& src)
{
	if (setCapacity(src.m_capacity))
	{
		for (size_t i = 0; i < src.m_count; ++i)
			new (m_elem + i) ELEMENT_T(src.m_elem[i]);

		m_count = src.m_count;
	}

	return *this;
}

template< typename ELEMENT_T >
inline Array< ELEMENT_T >&
Array< ELEMENT_T >::move(
	Array< ELEMENT_T>& src)
{
	m_capacity = src.m_capacity;
	m_count = src.m_count;

	m_elem.move(src.m_elem);

	src.m_capacity = 0;
	src.m_count = 0;

	return *this;
}

template< typename ELEMENT_T >
inline bool
Array< ELEMENT_T >::setCapacity(
	const size_t n)
{
	// setting the capacity resets the content
	for (size_t i = 0; i < m_count; ++i)
		m_elem[i].~ELEMENT_T();

	m_count = 0;

	if (m_capacity == n)
		return true;

	m_capacity = 0;

	if (0 == n)
	{
		m_elem.free();
		return true;
	}

	m_elem.malloc(n);

	if (m_elem.is_null())
		return false;

	m_capacity = n;

	return true;
}

template< typename ELEMENT_T >
inline bool
Array< ELEMENT_T >::addElement()
{
	if (m_count == m_capacity)
		return false;

	assert(!m_elem.is_null());

	new (m_elem + m_count++) ELEMENT_T();

	return true;
}

template< typename ELEMENT_T >
inline bool
Array< ELEMENT_T >::addElement(
	const ELEMENT_T& e)
{
	if (m_count == m_capacity)
		return false;

	assert(!m_elem.is_null());

	new (m_elem + m_count++) ELEMENT_T(e);

	return true;
}

template< typename ELEMENT_T >
inline bool
Array< ELEMENT_T >::addMultiElement(
	const size_t count)
{
	if (m_count + count > m_capacity)
		return false;

	assert(!m_elem.is_null());

	for (size_t i = 0; i < count; ++i)
		new (m_elem + m_count + i) ELEMENT_T();

	m_count += count;

	return true;
}

template< typename ELEMENT_T >
inline const ELEMENT_T&
Array< ELEMENT_T >::getElement(
	const size_t i) const
{
	assert(i < m_count);
	assert(!m_elem.is_null());

	return m_elem[i];
}

template< typename ELEMENT_T >
inline ELEMENT_T&
Array< ELEMENT_T >::getMutable(
	const size_t i)
{
	assert(i < m_count);
	assert(!m_elem.is_null());

	return m_elem[i];
}

#endif // array_H__

