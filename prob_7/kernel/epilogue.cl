// source_epilogue
#if OCL_OGL_INTEROP
	write_imageui(dst, (int2)(idx, idy), (uint4)(uchar)result);
#else
	dst[get_global_size(0) * idy + idx] = (uchar)result;
#endif
}

