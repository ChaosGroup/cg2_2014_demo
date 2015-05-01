#ifndef array_extern_H__
#define array_extern_H__

#include <assert.h>
#include <stdlib.h>
#include <stdint.h>

template < typename ELEMENT_T >
class ArrayExtern {

	ArrayExtern(
		const ArrayExtern&); // undefined

	ArrayExtern& operator =(
		const ArrayExtern&); // undefined

protected:
	uint32_t m_capacity;
	uint32_t m_count;
	ELEMENT_T* m_elem;

public:
	ArrayExtern()
	: m_capacity(0)
	, m_count(0)
	, m_elem(0) {
	}

	~ArrayExtern();

	bool
	setCapacity(
		const size_t n,
		ELEMENT_T* const elem);

	size_t
	getCapacity() const {
		return m_capacity;
	}

	size_t
	getCount() const {
		return m_count;
	}

	void
	resetCount() {
		setCapacity(m_capacity, m_elem);
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

template < typename ELEMENT_T >
inline ArrayExtern< ELEMENT_T >::~ArrayExtern() {

	assert(0 == m_capacity || 0 != m_elem);

	for (size_t i = 0; i < m_count; ++i)
		m_elem[i].~ELEMENT_T();

#if !defined(NDEBUG)
	m_capacity = 0;
	m_count = 0;
	m_elem = 0;

#endif
}

template < typename ELEMENT_T >
inline bool ArrayExtern< ELEMENT_T >::setCapacity(
	const size_t n,
	ELEMENT_T* const elem) {

	// setting the capacity resets the content
	for (size_t i = 0; i < m_count; ++i)
		m_elem[i].~ELEMENT_T();

	m_capacity = n;
	m_count = 0;
	m_elem = elem;

	return true;
}

template < typename ELEMENT_T >
inline bool ArrayExtern< ELEMENT_T >::addElement() {

	if (m_count == m_capacity)
		return false;

	assert(0 != m_elem);

	new (m_elem + m_count++) ELEMENT_T();

	return true;
}

template < typename ELEMENT_T >
inline bool ArrayExtern< ELEMENT_T >::addElement(
	const ELEMENT_T& e) {

	if (m_count == m_capacity)
		return false;

	assert(0 != m_elem);

	new (m_elem + m_count++) ELEMENT_T(e);

	return true;
}

template < typename ELEMENT_T >
inline bool ArrayExtern< ELEMENT_T >::addMultiElement(
	const size_t count) {

	if (m_count + count > m_capacity)
		return false;

	assert(0 != m_elem);

	for (size_t i = 0; i < count; ++i)
		new (m_elem + m_count + i) ELEMENT_T();

	m_count += count;

	return true;
}

template < typename ELEMENT_T >
inline const ELEMENT_T& ArrayExtern< ELEMENT_T >::getElement(
	const size_t i) const {

	assert(i < m_count);
	assert(0 != m_elem);

	return m_elem[i];
}

template < typename ELEMENT_T >
inline ELEMENT_T& ArrayExtern< ELEMENT_T >::getMutable(
	const size_t i) {

	assert(i < m_count);
	assert(0 != m_elem);

	return m_elem[i];
}

#endif // array_extern_H__

