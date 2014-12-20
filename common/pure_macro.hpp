#ifndef pure_macro_H__
#define pure_macro_H__

#define QUOTE(x) #x
#define XQUOTE(x) QUOTE(x)

#ifdef DEBUG
	#define DEBUG_LITERAL 1
#else
	#define	DEBUG_LITERAL 0
#endif

#endif // pure_macro_H_
