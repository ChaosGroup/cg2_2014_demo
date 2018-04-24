#ifndef stream_H__
#define stream_H__

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#endif
#include <assert.h>
#include <string>
#include <iomanip>
#include <ostream>
#include <cstring>

namespace stream {

class in {
	FILE* file;

public:
	in()
	: file(0) {
	}

	void close() {
		if (0 == file)
			return;

		fclose(file);
		file = 0;
	}

	bool open(const char* const filename) {
		close();

		file = fopen(filename, "r");
		return 0 != file;
	}

	bool open(FILE* const f) {
		close();

		const int fd = fileno(f);
		if (-1 != fd)
			file = fdopen(dup(fd), "r");

		return 0 != file;
	}

	~in() {
		close();
	}

	bool is_eof() const {
		if (0 != file)
			return feof(file) ? true : false;

		return false;
	}

	bool is_good() const {
		return (0 != file) && (0 == ferror(file));
	}

	void set_good() const {
		if (0 != file)
			clearerr(file);
	}

	const in& operator >>(char& a) const {
		if (0 != file)
			a = getc(file);

		return *this;
	}

	const in& operator >>(int16_t& a) const {
		if (0 != file)
			fscanf(file, "%hd", &a);

		return *this;
	}

	const in& operator >>(uint16_t& a) const {
		if (0 != file)
			fscanf(file, "%hu", &a);

		return *this;
	}

	const in& operator >>(int32_t& a) const {
		if (0 != file)
			fscanf(file, "%d", &a);

		return *this;
	}

	const in& operator >>(uint32_t& a) const {
		if (0 != file)
			fscanf(file, "%u", &a);

		return *this;
	}

	const in& operator >>(int64_t& a) const {
		if (0 != file)
#if _MSC_VER || __APPLE__
			fscanf(file, "%lld", &a);

#else
			fscanf(file, "%ld", &a);

#endif
		return *this;
	}

	const in& operator >>(uint64_t& a) const {
		if (0 != file)
#if _MSC_VER || __APPLE__
			fscanf(file, "%llu", &a);

#else
			fscanf(file, "%lu", &a);

#endif
		return *this;
	}

#if __clang__ && __APPLE__ // type size_t is unrelated to same-size type uint*_t
#if _LP64 == 1
	const in& operator >>(size_t& a) const {
		return *this >> reinterpret_cast< uint64_t& >(a);
	}

#else
	const in& operator >>(size_t& a) const {
		return *this >> reinterpret_cast< uint32_t& >(a);
	}

#endif
#endif
	const in& operator >>(float& a) const {
		if (0 != file)
			fscanf(file, "%f", &a);

		return *this;
	}

	const in& operator >>(double& a) const {
		if (0 != file)
			fscanf(file, "%lf", &a);

		return *this;
	}

	const in& operator >>(void*& a) const {
		if (0 != file)
			fscanf(file, "%p", &a);

		return *this;
	}

	const in& operator >>(std::string& a) const {
		if (0 != file) {
			const size_t buffer_inc = 1024;
			size_t buffer_size = 0;
			size_t len = 0;
			char* buffer = 0;

			while (true) {
				if (feof(file) || ferror(file))
					break;

				if (len == buffer_size) {
					buffer_size += buffer_inc;
					buffer = (char*) realloc(buffer, buffer_size);
				}

				const char read = getc(file);
				if (' ' == read || '\t' == read || '\n' == read)
					break;

				buffer[len++] = read;
			}

			a.assign(buffer, len);
			free(buffer);
		}

		return *this;
	}
};

class out {
	FILE* file;
	int width;
	char fillchar;

	enum Base {
		BASE_DEC,
		BASE_HEX,
		BASE_OCT
	} base;

	static size_t setFillInFormatStr(
		char (& format)[64],
		size_t fmtlen,
		const char fillchar) {

		const char fill_space[] = "%*";
		const char fill_zero[] = "%0*";
		const char fill_other[] = "<bad_fill> %*";

		switch (fillchar) {
		case ' ':
			memcpy(format + fmtlen, fill_space, strlen(fill_space));
			fmtlen += strlen(fill_space);
			break;

		case '0':
			memcpy(format + fmtlen, fill_zero, strlen(fill_zero));
			fmtlen += strlen(fill_zero);
			break;

		default:
			memcpy(format + fmtlen, fill_other, strlen(fill_other));
			fmtlen += strlen(fill_other);
			break;
		}

		return fmtlen;
	}

	static size_t setBaseInFormatStr(
		char (& format)[64],
		size_t fmtlen,
		const Base base,
		const char* base_dec,
		const char* base_hex,
		const char* base_oct) {

		switch (base) {
		case BASE_DEC:
			memcpy(format + fmtlen, base_dec, strlen(base_dec));
			fmtlen += strlen(base_dec);
			break;

		case BASE_HEX:
			memcpy(format + fmtlen, base_hex, strlen(base_hex));
			fmtlen += strlen(base_hex);
			break;

		case BASE_OCT:
			memcpy(format + fmtlen, base_oct, strlen(base_oct));
			fmtlen += strlen(base_oct);
			break;
		}

		return fmtlen;
	}

public:
	out()
	: file(0)
	, width(0)
	, fillchar(' ')
	, base(BASE_DEC) {
	}

	void close() {
		if (0 == file)
			return;

		fclose(file);
		file = 0;
	}

	bool open(const char* const filename, const bool append = true) {
		close();

		const char* const mode = append ? "a" : "w";
		file = fopen(filename, mode);
		return 0 != file;
	}

	bool open(FILE* const f) {
		close();

		const int fd = fileno(f);
		if (-1 != fd)
			file = fdopen(dup(fd), "a");

		return 0 != file;
	}

	~out() {
		close();
	}

	void flush() const {
		if (0 != file)
			fflush(file);
	}

	bool is_good() const {
		return (0 != file) && (0 == ferror(file));
	}

	void set_good() const {
		if (0 != file)
			clearerr(file);
	}

	out& operator <<(const char a) {
		if (0 != file)
			putc(a, file);

		return *this;
	}

	out& operator <<(const int16_t a) {
		if (0 == file)
			return *this;

		char format[64];
		size_t fmtlen = 0;

		const char base_dec[] = "hd";
		const char base_hex[] = "hx";
		const char base_oct[] = "ho";

		fmtlen = setFillInFormatStr(format, fmtlen, fillchar);
		fmtlen = setBaseInFormatStr(format, fmtlen, base, base_dec, base_hex, base_oct);

		format[fmtlen++] = '\0';

		fprintf(file, format, width, a);

		// reset width as per std::ostream specs
		width = 0;
		return *this;
	}

	out& operator <<(const uint16_t a) {
		if (0 == file)
			return *this;

		char format[64];
		size_t fmtlen = 0;

		const char base_dec[] = "hu";
		const char base_hex[] = "hx";
		const char base_oct[] = "ho";

		fmtlen = setFillInFormatStr(format, fmtlen, fillchar);
		fmtlen = setBaseInFormatStr(format, fmtlen, base, base_dec, base_hex, base_oct);

		format[fmtlen++] = '\0';

		fprintf(file, format, width, a);

		// reset width as per std::ostream specs
		width = 0;
		return *this;
	}

	out& operator <<(const int32_t a) {
		if (0 == file)
			return *this;

		char format[64];
		size_t fmtlen = 0;

		const char base_dec[] = "d";
		const char base_hex[] = "x";
		const char base_oct[] = "o";

		fmtlen = setFillInFormatStr(format, fmtlen, fillchar);
		fmtlen = setBaseInFormatStr(format, fmtlen, base, base_dec, base_hex, base_oct);

		format[fmtlen++] = '\0';

		fprintf(file, format, width, a);

		// reset width as per std::ostream specs
		width = 0;
		return *this;
	}

	out& operator <<(const uint32_t a) {
		if (0 == file)
			return *this;

		char format[64];
		size_t fmtlen = 0;

		const char base_dec[] = "u";
		const char base_hex[] = "x";
		const char base_oct[] = "o";

		fmtlen = setFillInFormatStr(format, fmtlen, fillchar);
		fmtlen = setBaseInFormatStr(format, fmtlen, base, base_dec, base_hex, base_oct);

		format[fmtlen++] = '\0';

		fprintf(file, format, width, a);

		// reset width as per std::ostream specs
		width = 0;
		return *this;
	}

	out& operator <<(const int64_t a) {
		if (0 == file)
			return *this;

		char format[64];
		size_t fmtlen = 0;

#if _MSC_VER != 0
		const char base_dec[] = "lld";
		const char base_hex[] = "llx";
		const char base_oct[] = "llo";

#else
		const char base_dec[] = "ld";
		const char base_hex[] = "lx";
		const char base_oct[] = "lo";

#endif
		fmtlen = setFillInFormatStr(format, fmtlen, fillchar);
		fmtlen = setBaseInFormatStr(format, fmtlen, base, base_dec, base_hex, base_oct);

		format[fmtlen++] = '\0';

		fprintf(file, format, width, a);

		// reset width as per std::ostream specs
		width = 0;
		return *this;
	}

	out& operator <<(const uint64_t a) {
		if (0 == file)
			return *this;

		char format[64];
		size_t fmtlen = 0;

#if _MSC_VER != 0
		const char base_dec[] = "llu";
		const char base_hex[] = "llx";
		const char base_oct[] = "llo";

#else
		const char base_dec[] = "lu";
		const char base_hex[] = "lx";
		const char base_oct[] = "lo";

#endif
		fmtlen = setFillInFormatStr(format, fmtlen, fillchar);
		fmtlen = setBaseInFormatStr(format, fmtlen, base, base_dec, base_hex, base_oct);

		format[fmtlen++] = '\0';

		fprintf(file, format, width, a);

		// reset width as per std::ostream specs
		width = 0;
		return *this;
	}

#if __clang__ && __APPLE__ // type size_t is unrelated to same-size type uint*_t
#if _LP64 == 1
	out& operator <<(const size_t a) {
		return *this << reinterpret_cast< const uint64_t& >(a);
	}

#else
	out& operator <<(const size_t a) {
		return *this << reinterpret_cast< const uint32_t& >(a);
	}

#endif
#endif
	out& operator <<(const float a) {
		if (0 == file)
			return *this;

		char format[64];
		size_t fmtlen = 0;

		fmtlen = setFillInFormatStr(format, fmtlen, fillchar);

		format[fmtlen++] = 'f';
		format[fmtlen++] = '\0';

		fprintf(file, format, width, a);

		// reset width as per std::ostream specs
		width = 0;
		return *this;
	}

	out& operator <<(const double a) {
		if (0 == file)
			return *this;

		char format[64];
		size_t fmtlen = 0;

		fmtlen = setFillInFormatStr(format, fmtlen, fillchar);

		format[fmtlen++] = 'f';
		format[fmtlen++] = '\0';

		fprintf(file, format, width, a);

		// reset width as per std::ostream specs
		width = 0;
		return *this;
	}

	out& operator <<(const void* const a) {
		if (0 != file)
			fprintf(file, "%p", a);

		return *this;
	}

	out& operator <<(const char* const a) {
		if (0 != file)
			fprintf(file, "%s", a);

		return *this;
	}

	out& operator <<(const std::string& a) {
		if (0 != file)
			fprintf(file, "%s", a.c_str());

		return *this;
	}

#if _MSC_VER != 0
	out& operator <<(const std::_Smanip<std::streamsize>& arg) {
		const std::_Smanip<std::streamsize> etalon = std::setw(42);

		if (etalon._Pfun == arg._Pfun)
			width = arg._Manarg;
		else
			assert(0);

		return *this;
	}

	out& operator <<(const std::_Fillobj<char>& arg) {
		fillchar = arg._Fill;
		return *this;
	}

#elif __clang__ != 0 && __APPLE__ != 0
	out& operator<<(const std::__iom_t6& a) {
		width = reinterpret_cast< const int& >(a); // std::__iom_t6::__n_
		return *this;
	}

	out& operator<<(const std::__iom_t4<char>& a) {
		fillchar = reinterpret_cast< const char& >(a); // std::__iom_t4<char>::__fill_
		return *this;
	}

#else
	out& operator <<(const std::_Setw& a) {
		width = a._M_n;
		return *this;
	}

	out& operator <<(const std::_Setfill<char>& a) {
		fillchar = a._M_c;
		return *this;
	}

#endif
	out& operator <<(std::ios_base& (* f)(std::ios_base&)) {

		if (std::dec == f) {
			base = BASE_DEC;
		}
		else
		if (std::hex == f) {
			base = BASE_HEX;
		}
		else
		if (std::oct == f) {
			base = BASE_OCT;
		}
		else {
			assert(0);
		}

		return *this;
	}

	out& operator <<(std::basic_ostream< char, std::char_traits< char > >& (* f)(std::basic_ostream< char, std::char_traits< char > >&)) {

		if (0 == file)
			return *this;

#if _MSC_VER
		// truly special treatment for msvc -- specializing the template functions of interest
		// does not provide matching function pointers; the following pre-defined symbols do
		if (std::endl == f) {
			putc('\n', file);
		}
		else
		if (std::ends == f) {
			putc('\0', file);
		}
		else
		if (std::flush == f) {
			fflush(file);
		}
		else {
			assert(0);
		}

#else
#if __GNUC__ >= 4 && __clang__ == 0
		// gcc needs an explicit cast of the specialized template-function type
		// to the formal parameter type

#if __cplusplus >= 201103L
#define cast_to_typeof(EXPR, ARG) static_cast< decltype(EXPR) >(ARG)
#else
#define cast_to_typeof(EXPR, ARG) static_cast< __typeof__(EXPR) >(ARG)
#endif

#else	// forgo the typecast for the rest of the compilers
#define cast_to_typeof(EXPR, ARG) (ARG)
#endif
		if (cast_to_typeof(f, (std::endl< char, std::char_traits< char > >)) == f) {
			putc('\n', file);
		}
		else
		if (cast_to_typeof(f, (std::ends< char, std::char_traits< char > >)) == f) {
			putc('\0', file);
		}
		else
		if (cast_to_typeof(f, (std::flush< char, std::char_traits< char > >)) == f) {
			fflush(file);
		}
		else {
			assert(0);
		}

#undef cast_to_typeof
#endif
		return *this;
	}
};

extern in cin;
extern out cout;
extern out cerr;

} // namespace stream

#endif // stream_H__
