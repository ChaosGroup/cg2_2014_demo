#include <stdlib.h>
#include <stdint.h>
#include <png.h>
#include <assert.h>
#include <cmath>
#ifndef Z_BEST_COMPRESSION
#define Z_BEST_COMPRESSION 9
#endif
#include "scoped.hpp"
#include "stream.hpp"
#include "timer.h"

// verify iostream-free status
#if _GLIBCXX_IOSTREAM
#error rogue iostream acquired
#endif

namespace stream {

// deferred initialization by main()
in cin;
out cout;
out cerr;

} // namespace stream

namespace testbed {

template < bool >
struct compile_assert;

template <>
struct compile_assert< true >
{
	compile_assert() {}
};

template < typename T >
class generic_free
{
public:
	void operator()(T* arg) {
		assert(0 != arg);
		std::free(arg);
	}
};

template <>
class scoped_functor< FILE >
{
public:
	void operator()(FILE* arg) {
		assert(0 != arg);
		fclose(arg);
	}
};

} // namespace testbed

static bool
write_png(
	const bool grayscale,
	const unsigned w,
	const unsigned h,
	void* const bits,
	FILE* const fp)
{
	using testbed::scoped_ptr;
	using testbed::generic_free;

	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr)
		return false;

	png_infop info_ptr = png_create_info_struct(png_ptr);

	if (!info_ptr) {
		png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
		return false;
	}

	// declare any RAII before the longjump, lest no destruction at longjump
	const scoped_ptr< png_bytep, generic_free > row((png_bytepp) malloc(sizeof(png_bytep) * h));

	if (setjmp(png_jmpbuf(png_ptr))) {
		png_destroy_write_struct(&png_ptr, &info_ptr);
		return false;
	}

	size_t pixel_size = sizeof(png_byte[3]);
	int color_type = PNG_COLOR_TYPE_RGB;

	if (grayscale) {
		pixel_size = sizeof(png_byte);
		color_type = PNG_COLOR_TYPE_GRAY;
	}

	for (size_t i = 0; i < h; ++i)
		row()[i] = (png_bytep) bits + w * (h - 1 - i) * pixel_size;

	png_init_io(png_ptr, fp);
	png_set_compression_level(png_ptr, Z_BEST_COMPRESSION);
	png_set_IHDR(png_ptr, info_ptr, w, h, 8, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_set_rows(png_ptr, info_ptr, row());
	png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

	png_destroy_write_struct(&png_ptr, &info_ptr);
	return true;
}

int main(int argc, char** argv)
{
	using testbed::scoped_ptr;
	using testbed::scoped_functor;
	using testbed::generic_free;
	using testbed::compile_assert;
	using std::isfinite;

	stream::cin.open(stdin);
	stream::cout.open(stdout);
	stream::cerr.open(stderr);

	const uint32_t image_w = 800;
	const uint32_t image_h = 800;

	const scoped_ptr< uint8_t, generic_free > image_map(
		reinterpret_cast< uint8_t* >(std::calloc(image_w * image_h, sizeof(uint8_t))));

	if (0 == image_map()) {
		stream::cerr << "error allocating image_map\n";
		return -1;
	}

	size_t frame = 0;
	const uint64_t t0 = timer_ns();
	const uint32_t ao_probe_count = 1 << 10;

	srand(1);
	unsigned seed0 = rand();
	unsigned seed1 = rand();

	for (size_t i = 0; i < ao_probe_count; ++i) {
		const compile_assert< 0 == (RAND_MAX & RAND_MAX + 1LL) > assert_rand_pot;
		const int r0 = rand_r(&seed0);
		const int r1 = rand_r(&seed1);

#if ALT == 0
		const float r[] = {
			r0 * float(M_PI_2   / (RAND_MAX + 1LL)), // decl0
			r1 * float(M_PI * 2 / (RAND_MAX + 1LL)), // azim0
		};
		float sin_dazim[2];
		float cos_dazim[2];
		sincosf(r[0], sin_dazim + 0, cos_dazim + 0);
		sincosf(r[1], sin_dazim + 1, cos_dazim + 1);

		const float sin_decl0 = sin_dazim[0];
		const float cos_decl0 = cos_dazim[0];
		const float sin_azim0 = sin_dazim[1];
		const float cos_azim0 = cos_dazim[1];

#elif ALT == 1
		const float r[] = {
			r0 * float(1.0      / (RAND_MAX + 1LL)), // decl0
			r1 * float(M_PI * 2 / (RAND_MAX + 1LL)), // azim0
		};
		const float sin_decl = sqrtf(1.f - r[0] * r[0]);
		const float cos_decl = r[0];
		float sin_azim;
		float cos_azim;
		sincosf(r[1], &sin_azim, &cos_azim);

		const float sin_decl0 = sin_decl;
		const float cos_decl0 = cos_decl;
		const float sin_azim0 = sin_azim;
		const float cos_azim0 = cos_azim;

#elif ALT == 2
		const float r[] = {
			r0 * float(1.0      / (RAND_MAX + 1LL)), // decl0
			r1 * float(M_PI * 2 / (RAND_MAX + 1LL)), // azim0
		};
		const float sin_decl = r[0];
		const float cos_decl = sqrtf(1.f - r[0] * r[0]);
		float sin_azim;
		float cos_azim;
		sincosf(r[1], &sin_azim, &cos_azim);

		const float sin_decl0 = sin_decl;
		const float cos_decl0 = cos_decl;
		const float sin_azim0 = sin_azim;
		const float cos_azim0 = cos_azim;

#elif ALT == 3
		const float r[] = {
			r0 * float(1.0      / (RAND_MAX + 1LL)), // decl0
			r1 * float(M_PI * 2 / (RAND_MAX + 1LL)), // azim0
		};
		const float sin_decl = sqrtf(1.f - r[0]);
		const float cos_decl = sqrtf(r[0]);
		float sin_azim;
		float cos_azim;
		sincosf(r[1], &sin_azim, &cos_azim);

		const float sin_decl0 = sin_decl;
		const float cos_decl0 = cos_decl;
		const float sin_azim0 = sin_azim;
		const float cos_azim0 = cos_azim;

#else
	#error ALT out of range

#endif
		// compute a bounce vector in some TBN space, in this case of an assumed normal along x-axis
		const float hemi0[] = { cos_decl0, cos_azim0 * sin_decl0, sin_azim0 * sin_decl0 };
		assert(isfinite(hemi0[0]));
		assert(isfinite(hemi0[1]));
		assert(isfinite(hemi0[2]));
		assert( 0.f <= hemi0[0] && hemi0[0] <= 1.f);
		assert(-1.f <= hemi0[1] && hemi0[1] <= 1.f);
		assert(-1.f <= hemi0[2] && hemi0[2] <= 1.f);
		const float len = sqrtf(
			hemi0[0] * hemi0[0] +
			hemi0[1] * hemi0[1] +
			hemi0[2] * hemi0[2]);
		const float eps = 1e-6;
		assert(fabsf(1.f - len) < eps);

		const float x = fminf(.999f, hemi0[1] * .5f + .5f); // [0, 1)
		const float y = fminf(.999f, hemi0[2] * .5f + .5f); // [0, 1)

		image_map()[image_w * uint32_t(image_h * y) + uint32_t(image_w * x)] = 0x80;
	}

	image_map()[0] = 0xff;
	image_map()[image_w - 1] = 0xff;
	image_map()[image_w * image_h - image_w] = 0xff;
	image_map()[image_w * image_h - 1] = 0xff;

	const uint64_t dt = timer_ns() - t0;

	stream::cout << "total frames rendered: " << ++frame << '\n';

	if (dt) {
		const double sec = double(dt) * 1e-9;
		stream::cout << "elapsed time: " << sec << " s\naverage FPS: " << frame / sec << '\n';
	}

	const char* const name = "last_frame.png";
	stream::cout << "saving framegrab as '" << name << "'\n";
	const scoped_ptr< FILE, scoped_functor > file(fopen(name, "wb"));

	if (0 == file()) {
		stream::cerr << "failure opening framegrab file '" << name << "'\n";
		return -1;
	}

	if (!write_png(true, image_w, image_h, image_map(), file())) {
		stream::cerr << "failure writing framegrab file '" << name << "'\n";
		return -1;
	}

	return 0;
}
