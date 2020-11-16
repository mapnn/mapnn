#!/bin/bash

````
############################
##  set the build directory
build_DIR=build_linux_x86-64


################################################
##  remove build directory and make the new one.
rm -r $build_DIR 2>/dev/null
mkdir -p $build_DIR 


#############################
## enter the build directory
pushd $build_DIR

#####################
## cmake with params
cmake ../../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=../../toolchains/host-amd64.toolchain.cmake

#######################################################
## make with 8 thread and install to default directory
make install -j16

#######################
## pop build directory
popd


````
