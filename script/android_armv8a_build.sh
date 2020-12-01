#!/bin/bash

````
############################
##  set the build directory.
build_DIR=build_android_armv8a

################################################
##  remove build directory and make the new one.
rm -r $build_DIR
mkdir $build_DIR


#############################
## enter the build directory.
pushd $build_DIR

#####################
## cmake with params.
cmake ../../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=../../toolchains/Android-armv8a.toolchain.cmake 

#######################################################
## make with 8 thread and install to default directory.
make install -j16

#######################
## pop build directory.
popd

````
