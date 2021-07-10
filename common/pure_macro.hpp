#ifndef pure_macro_H__
#define pure_macro_H__

#define QUOTE(x) #x
#define XQUOTE(x) QUOTE(x)

#ifdef DEBUG
	#define DEBUG_LITERAL 1
#else
	#define	DEBUG_LITERAL 0
#endif

template < typename T, size_t N >
char (& noneval_countof(const T (&)[N]))[N];

#define COUNT_OF(x) sizeof(noneval_countof(x))

#endif // pure_macro_H_
