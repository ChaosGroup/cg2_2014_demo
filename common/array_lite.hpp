#ifndef array_lite_H__
#define array_lite_H__

#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

template < typename ELEMENT_T, size_t CAPACITY_T, size_t PAYLOAD_OFFSET_T >
class ArrayLite {

protected:
	ELEMENT_T* bits() {
		return reinterpret_cast< ELEMENT_T* >(uintptr_t(this) + uintptr_t(PAYLOAD_OFFSET_T));
	}

	const ELEMENT_T* bits() const {
		return reinterpret_cast< const ELEMENT_T* >(uintptr_t(this) + uintptr_t(PAYLOAD_OFFSET_T));
	}

	uint32_t m_count;

public:
	ArrayLite()
	: m_count(0) {
	}

	template < size_t SRC_PAYLOAD_OFFSET_T >
	ArrayLite(
		const ArrayLite< ELEMENT_T, CAPACITY_T, SRC_PAYLOAD_OFFSET_T >& src);

	~ArrayLite();

	template < size_t SRC_PAYLOAD_OFFSET_T >
	ArrayLite& operator =(
		const ArrayLite< ELEMENT_T, CAPACITY_T, SRC_PAYLOAD_OFFSET_T >& src);

	size_t
	getCapacity() const {
		return CAPACITY_T;
	}

	size_t
	getCount() const {
		return m_count;
	}

	void
	resetCount() {
		for (size_t i = 0; i < m_count; ++i)
			bits()[i].~ELEMENT_T();

		m_count = 0;
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

template < typename ELEMENT_T, size_t CAPACITY_T, size_t PAYLOAD_OFFSET_T >
template < size_t SRC_PAYLOAD_OFFSET_T >
inline ArrayLite< ELEMENT_T, CAPACITY_T, PAYLOAD_OFFSET_T >::ArrayLite(
	const ArrayLite< ELEMENT_T, CAPACITY_T, SRC_PAYLOAD_OFFSET_T >& src)
: m_count(0) {

	if (0 == src.m_count)
		return;

	for (size_t i = 0; i < src.m_count; ++i)
		new (bits() + i) ELEMENT_T(src.bits()[i]);

	m_count = src.m_count;
}

template < typename ELEMENT_T, size_t CAPACITY_T, size_t PAYLOAD_OFFSET_T >
inline ArrayLite< ELEMENT_T, CAPACITY_T, PAYLOAD_OFFSET_T >::~ArrayLite() {

	for (size_t i = 0; i < m_count; ++i)
		bits()[i].~ELEMENT_T();

#if !defined(NDEBUG)
	m_count = 0;

#endif
}

template < typename ELEMENT_T, size_t CAPACITY_T, size_t PAYLOAD_OFFSET_T >
template < size_t SRC_PAYLOAD_OFFSET_T >
inline ArrayLite< ELEMENT_T, CAPACITY_T, PAYLOAD_OFFSET_T >&
ArrayLite< ELEMENT_T, CAPACITY_T, PAYLOAD_OFFSET_T >::operator =(
	const ArrayLite< ELEMENT_T, CAPACITY_T, SRC_PAYLOAD_OFFSET_T >& src) {

	for (size_t i = 0; i < m_count; ++i)
		bits()[i].~ELEMENT_T();

	for (size_t i = 0; i < src.m_count; ++i)
		new (bits() + i) ELEMENT_T(src.bits()[i]);

	m_count = src.m_count;
	return *this;
}

template < typename ELEMENT_T, size_t CAPACITY_T, size_t PAYLOAD_OFFSET_T >
inline bool
ArrayLite< ELEMENT_T, CAPACITY_T, PAYLOAD_OFFSET_T >::addElement() {

	if (m_count == CAPACITY_T)
		return false;

	new (bits() + m_count++) ELEMENT_T();
	return true;
}

template < typename ELEMENT_T, size_t CAPACITY_T, size_t PAYLOAD_OFFSET_T >
inline bool
ArrayLite< ELEMENT_T, CAPACITY_T, PAYLOAD_OFFSET_T >::addElement(
	const ELEMENT_T& e) {

	if (m_count == CAPACITY_T)
		return false;

	new (bits() + m_count++) ELEMENT_T(e);
	return true;
}

template < typename ELEMENT_T, size_t CAPACITY_T, size_t PAYLOAD_OFFSET_T >
inline bool
ArrayLite< ELEMENT_T, CAPACITY_T, PAYLOAD_OFFSET_T >::addMultiElement(
	const size_t count) {

	if (m_count + count > CAPACITY_T)
		return false;

	for (size_t i = 0; i < count; ++i)
		new (bits() + m_count + i) ELEMENT_T();

	m_count += count;
	return true;
}

template < typename ELEMENT_T, size_t CAPACITY_T, size_t PAYLOAD_OFFSET_T >
inline const ELEMENT_T&
ArrayLite< ELEMENT_T, CAPACITY_T, PAYLOAD_OFFSET_T >::getElement(
	const size_t i) const {

	assert(i < m_count);
	return bits()[i];
}

template < typename ELEMENT_T, size_t CAPACITY_T, size_t PAYLOAD_OFFSET_T >
inline ELEMENT_T&
ArrayLite< ELEMENT_T, CAPACITY_T, PAYLOAD_OFFSET_T >::getMutable(
	const size_t i) {

	assert(i < m_count);
	return bits()[i];
}

#endif // array_lite_H__

