Description
-----------

This is the invitation demo for Chaos Group's CG2 2014 coders competition, part of [CG2 'Code for Art' 2014](http://cg2.chaosgroup.com/conf2014/), which took place on Oct 25th at IEC Sofia. Organized and partially sponsored by Chaos Group Ltd., CG2 is an annual conference event which brings together computer graphics professionals and enthusiasts alike, and offers a tightly-packed program of lectures and presentations from the field. The 2014 coders competition was about [procedurally-generated demoscene-style visual productions](http://cg2.chaosgroup.com/dev-competition/). The invitation demo runs entirely on the CPU (some post-processing done on the GPU), and does Monte-Carlo-based object-space ambient occlusion by tracing rays across dynamic-content (frame-by-frame generated) voxel trees. The version used in the [youtube video](https://www.youtube.com/watch?v=Fs5zvCip2uI) was rendered at non-realtime settings - 1024x1024 resolution, 64 AO rays per pixel. In comparison, the version on the floor machine was configured to run at 768x768 and 24 AO rays, though it never got to run in public due to event schedule overrun. For reference, the stage machine was a 24-core/48-thread Xeon NUMA.

Project Tree
------------

The demo itself is in prob_6; it shares a good deal of code (read: identical headers and translation units) with prob_4, which is a game built using the technology from the demo. Its unofficial name is 'An Occlusion Game Unlike No 0ther', for short - aogun0.

* CL     - OpenCL 1.2 headers from khronos.org
* common - shared code, not specific to any particular project
* prob_4 - aogun0
* prob_6 - CG2 2014 invitation demo
* prob_7 - CG2 2014 invitation demo, OpenCL redux (in progress)

How to Build
------------

Running the build_glx.sh script in the respective directiory builds the resident project.

Build Prerequisites
-------------------

Linux with glibc 2.2.5, glibcxx 3.4.11 and X11; OpenGL 3.1, GLX 1.3

clang++, preferably 3.5, still 3.4 works fine. Older versions might do as well, but have not been tested.

Default build options are set for link-time optimisations (-flto), which require the presence of a gold linker and LLVMGold.so module. To disable this optimisation just comment out -flto from the build script.

Many things are controlled at build time, via macro definitions in the build script. Here are a few defines one might want to adjust:

* WORKFORCE_NUM_THREADS - Number of workforce threads (normally equating the number of logical cores)
* WORKFORCE_THREADS_STICKY - Make workforce threads sticky (NUMA, etc)
* AO_NUM_RAYS - Number of AO rays per pixel

Screengrabs of aogun0
---------------------

![suboptimal arrangement](images/ao064_default_hmm_t.png "suboptimal block arrangement per default seed") ![optimal arrangement](images/ao064_default_opt_t.png "optimal arrangement per default seed")

prob_7 openCL 1.2 benchmark
---------------------------

*'Have raycaster -- will benchmark'* -- [Robert A. Heinlein](https://en.wikipedia.org/wiki/Have_Space_Suit%E2%80%94Will_Travel)

Results from headless build run to the 1000th frame, resolution 3840x2160; on hardware with multiple device types the name of the soc/cpu is given, followed by device in parentheses:

| hardware (device)                 | device multiplicity         | device mem, GB/s  | FPS      | remarks                                                                                         |
| --------------------------------- | --------------------------- | ----------------- | -------- | ----------------------------------------------------------------------------------------------- |
| Rockchip RK3399 (Mali-T860MP4)    |   1x Mali-T860              | 12.8              | 0.900    | ARM Mali-T860 OpenCL 1.2 v1.r14p0-01rel0-git(a79caef).8ddfd7584149d9238dced4e406610de7, 800 MHz |
| Rockchip RK3399 (Cortex-A72)      |   2x Cortex-A72             | 12.8              | 0.379    | pocl 1.3, LLVM 8.0.0, OCL_KERNEL_TARGET_CPU: cortex-a72, 1800 MHz                               |
| Amlogic S922X (Mali-G52MP6)       |   1x Mali-G52               | 10.56             | 0.982    | ARM Mali-G52 OpenCL 2.0 git.c8adbf9.122c9daed32dbba4b3056f41a2f23c58, 750 MHz                   |
| Amlogic S922X (Cortex-A73)        |   4x Cortex-A73             | 10.56             | 0.763    | pocl 1.3, LLVM 8.0.0, OCL_KERNEL_TARGET_CPU: cortex-a73, 1800 MHz                               |
| Marvell ARMADA 8040 (Cortex-A72)  |   4x Cortex-A72             | 19.2              | 0.824    | pocl 1.3, LLVM 8.0.0, OCL_KERNEL_TARGET_CPU: cortex-a72, 2000 MHz                               |
| AWS Graviton (Cortex-A72)         |  16x Cortex-A72             | 38.4              | 3.639    | pocl 1.3, LLVM 8.0.0, OCL_KERNEL_TARGET_CPU: cortex-a72, 2290 MHz                               |
| NXP LX2160A (Cortex-A72)          |  16x Cortex-A72             | 19.2              | 3.200    | pocl 1.3, LLVM 8.0.0, OCL_KERNEL_TARGET_CPU: cortex-a72, 2000 MHz                               |
| Intel Xeon E5-2687W (SNB), 2S     |  16x Sandy Bridge (32x SMT) | 51.2 (25.6 1S)    | 2.330    | Intel(R) Corporation, Intel(R) Xeon(R) CPU, OpenCL 1.2 (Build 67279), 3100 MHz                  |
| Intel Xeon E5-2687W (GF108GL)     |   1x GF108GL                | 25.6              | 3.325    | NVIDIA Corporation Quadro 600 OpenCL 1.1 CUDA 375.66, 1280 MHz                                  |
| Intel Xeon E3-1270v2 (GK208B)     |   1x GK208B                 | 40                | 3.677    | NVIDIA Corporation GeForce GT 720 OpenCL 1.1 CUDA 340.102, 797 MHz                              |
| Intel Xeon E3-1270v2 (GT218)      |   1x GT218                  | 9.6               | 0.612    | NVIDIA Corporation GeForce 210 OpenCL 1.1 CUDA 340.102, 1230 MHz                                |
| Intel P8600 (MCP89)               |   1x MCP89                  | 8.53              | 1.427    | NVIDIA GeForce 320M OpenCL 1.0 10.2.37 310.90.10.05b54, CLI: -use_images, 950 MHz               |
| NVIDIA Jetson Nano (GM20B)        |   1x GM20B (128 cores)      | 25.6              | 4.754    | pocl 1.6-pre, LLVM 8.0.0, CUDA-sm_53, NVIDIA Tegra X1, 922 MHz                                  |
| NVIDIA Jetson Xavier NX (GV10B)   |   1x GV10B (348 cores)      | 51.2              | 14.744   | pocl 1.6-pre, LLVM 8.0.0, CUDA-sm_72, NVIDIA Xavier, 1100 MHz                                   |
| Apple M1 (AppleFamily7)           |   1x AppleFamily7 (7 cores) | 68.25             | 60.565   | Apple M1 OpenCL 1.2 (Nov 23 2020 03:06:28), 1000 MHz                                            |
| Apple M1 (AppleFamily7)           |   1x AppleFamily7 (8 cores) | 68.25             | 64.963   | Apple M1 OpenCL 1.2 (Dec 21 2020 17:26:51), 1000 MHz                                            |

Same as above but from branch `better_cpu` and `pocl` patched for good-codegen `convert_T` function:

| hardware (device)                 | device multiplicity         | device mem, GB/s  | FPS      | remarks                                                                                         |
| --------------------------------- | --------------------------- | ----------------- | -------- | ----------------------------------------------------------------------------------------------- |
| Rockchip RK3399 (Cortex-A72)      |   2x Cortex-A72             | 12.8              | 0.410    | pocl 1.3, LLVM 8.0.0, OCL_KERNEL_TARGET_CPU: cortex-a72, 1800 MHz                               |
| Amlogic S922X (Cortex-A73)        |   4x Cortex-A73             | 10.56             | 0.819    | pocl 1.3, LLVM 8.0.0, OCL_KERNEL_TARGET_CPU: cortex-a73, 1800 MHz                               |
| Marvell ARMADA 8040 (Cortex-A72)  |   4x Cortex-A72             | 19.2              | 0.912    | pocl 1.3, LLVM 8.0.0, OCL_KERNEL_TARGET_CPU: cortex-a72, 2000 MHz                               |
| Snapdragon 835 (Cortex-A73)       |   4x Cortex-A73             | 14.93             | 0.958    | pocl 1.5, LLVM 9.0.0, OCL_KERNEL_TARGET_CPU: cortex-a73, 2200 MHz                               |
| Snapdragon SQ1 (Cortex-A76)       |   4x Cortex-A76 + 4x A55    | 68.25             | 2.652    | pocl 1.5, LLVM 9.0.1, OCL_KERNEL_TARGET_CPU: cortex-a76, 3000 MHz + 1800 MHz                    |
| Apple M1 (Firestorm)              |   4x Firestorm (passive)    | 68.25             | 3.499    | pocl 1.7, LLVM 11.0.0, OCL_KERNEL_TARGET_CPU: cyclone, 3200 MHz                                 |
| Apple M1 (Firestorm)              |   4x Firestorm (active)     | 68.25             | 3.580    | pocl 1.7, LLVM 11.0.0, OCL_KERNEL_TARGET_CPU: cyclone, 3200 MHz                                 |
| Snapdragon x1e78100 (Oryon)       |  12x Oryon                  | 85 (tinymembench) | 12.223   | pocl 1.7, LLVM 12.0.1, OCL_KERNEL_TARGET_CPU: cyclone, 3400 MHz                                 |
| NXP LX2160A (Cortex-A72)          |  16x Cortex-A72             | 38.4              | 3.551    | pocl 1.4, LLVM 8.0.0, OCL_KERNEL_TARGET_CPU: cortex-a72, 2000 MHz                               |
| AWS Graviton (Cortex-A72)         |  16x Cortex-A72             | 38.4              | 4.049    | pocl 1.3, LLVM 8.0.0, OCL_KERNEL_TARGET_CPU: cortex-a72, 2290 MHz                               |
| AWS Graviton2 (Cortex-A76)        |  64x Cortex-A76             | 40 (tinymembench) | 26.491   | pocl 1.4, LLVM 8.0.0, OCL_KERNEL_TARGET_CPU: cortex-a75, 2500 MHz                               |
| AWS Xeon Platinum 8175M (Skylake) |  24x Skylake (48x SMT)      | 19 (tinymembench) | 10.335   | pocl 1.5, LLVM 8.0.0, OCL_KERNEL_TARGET_CPU: skylake-avx512, 2500 MHz                           |
| NVIDIA Tegra Orin (Cortex-A78AE)  |  12x Cortex-A76AE           | 26 (tinymembench) | 5.488    | pocl 1.7, LLVM 9.0.1, OCL_KERNEL_TARGET_CPU: cortex-a76, 2200 MHz                               |

benchmark build directions
--------------------------
To get results relevant to the 1st table use commit 474efe0 from branch `master`. Detailed directions for getting results relevant to the 2nd table:
```
$ sudo apt-get install --no-install-recommends libhwloc-dev # this is optional as pocl 1.5 has alternative means to collect hw info than via libhwloc
$ sudo apt-get install build-essential llvm-8-dev llvm-8 clang-8 libclang-8-dev libpng-dev cmake pkg-config
$ git clone -b release_1_5 --single-branch https://github.com/pocl/pocl.git # R1.5 has an important performance fix, also applied to all entries in this test
$ mkdir pocl_build
$ cd pocl_build
$ cmake ../pocl -DLLC_HOST_CPU=cortex-a73 -DDEFAULT_ENABLE_ICD=0 -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-8 # CPU choice is crucial as it controls kernel codegen -- pick something close to your uarch
$ make
$ sudo make install
$ cd ..
$ git clone -b better_cpu --single-branch https://github.com/ChaosGroup/cg2_2014_demo.git # branch has certain CPU-centric optimisations (see OCL_QUIRK below)
$ cd cg2_2014_demo/prob_7
$ ./build_headless.sh # on Arm clang needs adjusting target-arch options first -- comment out 'native' arch/cpu, uncomment relevant arch/cpu; you can also experiment with OCL_QUIRK_0004
$ LD_LIBRARY_PATH=/usr/local/lib/ ./problem_4 -screen "3840 2160 60" -frames 1000 -device 0 # have patience; subsequent runs use cached kernels, which can improve times
```
