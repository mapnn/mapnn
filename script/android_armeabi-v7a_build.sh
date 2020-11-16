#!/bin/bash

````

##  set the build directory
build_DIR=build_android_armeabi-v7a

##  remove build directory and make the new one.
rm -r $build_DIR
mkdir $build_DIR


## enter the build directory
pushd $build_DIR

## cmake with params
cmake ../../ \
    -DCMAKE_TOOLCHAIN_FILE=../../toolchains/android-armv7a.toolchain.cmake  \
    -DPROTOBUF_ROOT_DIR=/home/geewoo/Documents/nnwork/android-protobuf/ 


## make with 8 thread and install to default directory
make install -j8


## pop build directory
popd


````
