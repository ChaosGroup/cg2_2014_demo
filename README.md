cg2_2014_demo
-------------


This is the invitation demo for Chaos Group's CG2 2014 coders competition, part of [CG2 'Code for Art' 2014](http://cg2.chaosgroup.com/conf2014/), which took place on Oct 25th at IEC Sofia. Organized and partially sponsored by Chaos Group Ltd., CG2 is an annual conference event, meant to bring together computer graphics professionals and enthusiasts alike, and offering a tightly-packed program of lectures and presentations from the field. The 2014 coders competition was about [procedurally-generated demo-scene-style visual productions](http://cg2.chaosgroup.com/dev-competition/). The invitation demo runs entirely on the CPU (some post-processing done on the GPU), and does Monte-Carlo-based object-space ambient occlusion by tracing rays across dynamic-content (frame-by-frame generated) voxel trees. The version used in the [youtube video](https://www.youtube.com/watch?v=Fs5zvCip2uI) was rendered at non-realtime settings - 1024x1024 resolution, 64 AO rays per pixel. In comparison, the version on the floor machine was configured to run at 768x768 and 24 AO rays, though it never got to run before public due to event schedule overrun. For reference, the stage machine was a 24-core/48-thread Xeon NUMA beast.

Project Tree
------------

The demo itself is in prob_6; it shares a good deal of code (read: identical headers and translation units) with prob_4, which is a game of sorts, built using the technology from the demo. Its unofficial name is 'An Occlusion Game Unlike No 0ther', for short - aogun0.

* common - various shared functionality, not specific to prob_4 or prob_6
* prob_4 - aogun0
* prob_6 - CG2 2014 invitation demo

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
