#!/bin/bash

###########################################################################################
# launch the sound; any low-latency player would do but here we use play from sox

play -q ../music/cg2_2014_demo.wav &

###########################################################################################
# for 736x736 use the params below
#
#./problem_4 -screen "736 736 60"

# the default 512x512
./problem_4
