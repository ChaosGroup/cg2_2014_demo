#ifndef pure_macro_H__
#define pure_macro_H__

#define QUOTE(x) #x
#define XQUOTE(x) QUOTE(x)

#ifdef DEBUG
	#define DEBUG_LITERAL 1
#else
	#define	DEBUG_LITERAL 0
#endif

template < size_t N >
struct confirm_static_array_type {
	char arr[N];
};

template < typename T, size_t N >
inline confirm_static_array_type< N > confirm_static_array(const T(&)[N]) {
	return confirm_static_array_type< N >();
}

#define COUNT_OF(x) sizeof(confirm_static_array(x))

#endif // pure_macro_H_
