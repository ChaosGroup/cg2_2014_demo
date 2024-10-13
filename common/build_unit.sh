#!/bin/bash

# known issue with clang++-3.6 and the lifespan of temporaries, causing multiple failures at unittest
CXX=${CXX:-clang++}
BINARY=unittest
SOURCE=(
	unittest.cpp
)
CXXFLAGS=(
# g++-4.8 through 4.9 suffer from a name-mangling conflict in cephes headers; switching to the following ABI version resolves that
#	-fabi-version=6
# g++-6 warns about ignored attributes in template arguments (e.g. __m256 as a typename template arg to simd::native8)
#	-Wno-ignored-attributes
	-std=c++11
	-Wno-logical-op-parentheses
	-Wno-bitwise-op-parentheses
	-Wno-parentheses
	-pipe
	-fno-exceptions
	-fno-rtti
# Clang static code analysis:
#	--analyze
)

source cxx_util.sh

if [[ ${MACHTYPE} =~ "-apple-darwin" ]]; then
	if [[ ${HOSTTYPE} == "arm64" ]]; then

		CXXFLAGS+=(
			-march=armv8.4-a
		)

	elif [[ ${HOSTTYPE} == "x86_64" ]]; then

		CXXFLAGS+=(
			-march=native
		)

	fi
	CXXFLAGS+=(
		-mtune=native
		-DCACHELINE_SIZE=`sysctl hw.cachelinesize | sed 's/^hw.cachelinesize: //g'`
	)

elif [[ ${MACHTYPE} =~ "-linux-" ]]; then
	if [[ ${HOSTTYPE} == "arm64" || ${HOSTTYPE} == "aarch64" ]]; then

		cxx_uarch_arm

	elif [[ ${HOSTTYPE} == "x86_64" ]]; then

		CXXFLAGS+=(
			-march=native
			-mtune=native
			-DCACHELINE_SIZE=64
		)
	fi
else
	echo Unknown platform
	exit 255
fi

LFLAGS=(
# Alias some glibc6 symbols to older ones for better portability
#	-Wa,-defsym,memcpy=memcpy@GLIBC_2.2.5
#	-Wa,-defsym,__sqrtf_finite=__sqrtf_finite@GLIBC_2.2.5
)

if [[ $1 == "debug" ]]; then
	CXXFLAGS+=(
		-Wall
		-O0
		-g
		-DDEBUG
	)
else
	CXXFLAGS+=(
		-Ofast
		-fno-unsafe-math-optimizations
		-fstrict-aliasing
		-fstrict-overflow
		-funroll-loops
		-fomit-frame-pointer
		-flto
		-DNDEBUG
	)
fi

CXX_FILENAME=${CXX##*/}
if [[ ${CXX_FILENAME:0:3} == "g++" ]]; then
	if [[ ${HOSTTYPE} == "arm64" || ${HOSTTYPE} == "aarch64" ]]; then
		CXXFLAGS+=(
			-mpc-relative-literal-loads
		)
	fi
elif [[ ${CXX_FILENAME:0:7} == "clang++" ]]; then
	CXXFLAGS+=(
		# no idea yet how to enforce PC-relative literals
	)
fi

BUILD_CMD=${CXX}" -o "${BINARY}" "${CXXFLAGS[@]}" "${SOURCE[@]}" "${LFLAGS[@]}
echo ${BUILD_CMD}
CCC_ANALYZER_CPLUSPLUS=1 ${BUILD_CMD}
