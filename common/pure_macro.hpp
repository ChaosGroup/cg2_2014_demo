#ifndef pure_macro_H__
#define pure_macro_H__

#define QUOTE(x) #x
#define XQUOTE(x) QUOTE(x)

#ifdef DEBUG
	#define DEBUG_LITERAL 1
#else
	#define	DEBUG_LITERAL 0
#endif

#if __clang__
struct confirm_static_array {
	template < typename T, size_t N >
	confirm_static_array(const T(&)[N]) {
	}
};

// c++03 forbids the use of comma operator in const expressions even if the operands are
// const expressions themselves; c++11 fixes that, but clang++ also extends that to c++03
#define COUNT_OF(x) (sizeof(confirm_static_array(x)), sizeof(x) / sizeof((x)[0]))

#else
#define COUNT_OF(x) (sizeof(x) / sizeof((x)[0]))

#endif

#endif // pure_macro_H_
