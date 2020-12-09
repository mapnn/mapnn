#!/bin/bash

build_DIR=build_android_armeabi-v7l

rm -r $build_DIR
mkdir $build_DIR
pushd $build_DIR
cmake ../../ \
    -DCMAKE_INSTALL_PREFIX=install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=../../toolchains/Linux-armv7hl.toolchain.cmake
make install -j16
popd