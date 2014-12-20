#ifndef vect_base_H__
#define vect_base_H__

#include <cassert>

namespace base
{

template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
class vect
{
public:

	typedef SCALTYPE_T scaltype;
	typedef NATIVE_T native;

	enum
	{
		dimension = DIMENSION_T,
		native_count = (sizeof(SCALTYPE_T) * DIMENSION_T + sizeof(NATIVE_T) - 1) / sizeof(NATIVE_T),
		native_dimension = native_count * sizeof(NATIVE_T) / sizeof(SCALTYPE_T),
		scalars_per_native = sizeof(NATIVE_T) / sizeof(SCALTYPE_T)
	};

	vect()
	{
	}

	// element mutator
	void set(
		const size_t i,
		const SCALTYPE_T c);

	// element accessor
	SCALTYPE_T get(
		const size_t i) const;

	// element accessor, array subscript
	SCALTYPE_T operator [](
		const size_t i) const;

	bool operator ==(
		const vect& src) const;

	bool operator !=(
		const vect& src) const;

	bool operator >(
		const vect& src) const;

	bool operator >=(
		const vect& src) const;

	// native vector mutator
	vect& setn(
		const size_t i,
		const NATIVE_T src);

	// native vector accessor
	NATIVE_T getn(
		const size_t i = 0) const;

protected:

	// native element mutator
	void set_native(
		const size_t i,
		const SCALTYPE_T c);

private:

	NATIVE_T n[native_count];
};


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline void
vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >::set(
	const size_t i,
	const SCALTYPE_T c)
{
	assert(i < dimension);
	assert(scalars_per_native == 1);

	n[i] = c;
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline void
vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >::set_native(
	const size_t i,
	const SCALTYPE_T c)
{
	assert(i < native_dimension);
	assert(scalars_per_native == 1);

	n[i] = c;
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline SCALTYPE_T
vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >::get(
	const size_t i) const
{
	assert(i < dimension);
	assert(scalars_per_native == 1);

	return n[i];
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline SCALTYPE_T
vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >::operator [](
	const size_t i) const
{
	return this->get(i);
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >&
vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >::setn(
	const size_t i,
	const NATIVE_T src)
{
	assert(i < native_count);

	n[i] = src;

	return *this;
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline NATIVE_T
vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >::getn(
	const size_t i) const
{
	assert(i < native_count);

	return n[i];
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline bool
vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >::operator ==(
	const vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >& src) const
{
	for (size_t i = 0; i < DIMENSION_T; ++i)
		if (this->operator [](i) != src[i])
			return false;

	return true;
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline bool
vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >::operator !=(
	const vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >& src) const
{
	return !this->operator ==(src);
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline bool
vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >::operator >(
	const vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >& src) const
{
	for (size_t i = 0; i < DIMENSION_T; ++i)
		if (this->operator [](i) <= src[i])
			return false;

	return true;
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline bool
vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >::operator >=(
	const vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >& src) const
{
	for (size_t i = 0; i < DIMENSION_T; ++i)
		if (this->operator [](i) < src[i])
			return false;

	return true;
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
class matx
{
	vect< SCALTYPE_T, DIMENSION_T, NATIVE_T > m[DIMENSION_T];

public:

	typedef SCALTYPE_T scaltype;
	typedef NATIVE_T native;

	enum { dimension = DIMENSION_T };

	matx()
	{
	}

	// element mutator
	void set(
		const size_t i,
		const size_t j,
		const SCALTYPE_T c);

	// element accessor
	SCALTYPE_T get(
		const size_t i,
		const size_t j) const;

	// row mutator
	void set(
		const size_t i,
		const vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >& row);

	// row accessor
	const vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >& get(
		const size_t i) const;

	// row accessor, array subscript
	const vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >& operator [](
		const size_t i) const;

	bool operator ==(
		const matx& src) const;

	bool operator !=(
		const matx& src) const;

	// native mutator
	void setn(
		const size_t i,
		const size_t j,
		const NATIVE_T native);

	// native accessor
	NATIVE_T getn(
		const size_t i,
		const size_t j) const;
};


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline void
matx< SCALTYPE_T, DIMENSION_T, NATIVE_T >::set(
	const size_t i,
	const size_t j,
	const SCALTYPE_T c)
{
	assert(i < dimension && j < dimension);

	m[i].set(j, c);
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline SCALTYPE_T
matx< SCALTYPE_T, DIMENSION_T, NATIVE_T >::get(
	const size_t i,
	const size_t j) const
{
	assert(i < dimension && j < dimension);

	return m[i].get(j);
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline void
matx< SCALTYPE_T, DIMENSION_T, NATIVE_T >::set(
	const size_t i,
	const vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >& row)
{
	assert(i < dimension);

	m[i] = row;
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline const vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >&
matx< SCALTYPE_T, DIMENSION_T, NATIVE_T >::get(
	const size_t i) const
{
	assert(i < dimension);

	return m[i];
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline const vect< SCALTYPE_T, DIMENSION_T, NATIVE_T >&
matx< SCALTYPE_T, DIMENSION_T, NATIVE_T >::operator [](
	const size_t i) const
{
	return this->get(i);
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline bool
matx< SCALTYPE_T, DIMENSION_T, NATIVE_T >::operator ==(
	const matx< SCALTYPE_T, DIMENSION_T, NATIVE_T >& src) const
{
	for (size_t i = 0; i < dimension; ++i)
		if (this->operator [](i) != src[i])
			return false;

	return true;
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline bool
matx< SCALTYPE_T, DIMENSION_T, NATIVE_T >::operator !=(
	const matx< SCALTYPE_T, DIMENSION_T, NATIVE_T >& src) const
{
	return !this->operator ==(src);
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline void
matx< SCALTYPE_T, DIMENSION_T, NATIVE_T >::setn(
	const size_t i,
	const size_t j,
	const NATIVE_T n)
{
	assert(i < DIMENSION_T && j < m[i].native_count);

	m[i].setn(j, n);
}


template < typename SCALTYPE_T, size_t DIMENSION_T, typename NATIVE_T >
inline NATIVE_T
matx< SCALTYPE_T, DIMENSION_T, NATIVE_T >::getn(
	const size_t i,
	const size_t j) const
{
	assert(i < dimension && j < m[i].native_count);

	return m[i].getn(j);
}

} // namespace base

#endif // vect_base_H__
