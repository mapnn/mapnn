#!/bin/bash

build_DIR=build_android_armeabi-v7l

#rm -r $build_DIR
mkdir $build_DIR
pushd $build_DIR
cmake ../../ \
    -DCMAKE_TOOLCHAIN_FILE=../../toolchains/linux-armv7l.toolchain.cmake \
    -DPROTOBUF_ROOT_DIR=/home/geewoo/Documents/nnwork/pi3b-protobuf/ \
    -DCMAKE_C_COMPILER=/opt/raspberry/tools/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian-x64/bin/arm-linux-gnueabihf-gcc \
    -DCMAKE_CXX_COMPILER=/opt/raspberry/tools/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian-x64/bin/arm-linux-gnueabihf-g++ 
make install -j8
popd
