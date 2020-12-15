mapnn
---
[![License](https://img.shields.io/badge/license-Apache2.0-blue)](https://github.com/mapnn/mapnn) 
[![Build Status](https://github.com/mapnn/mapnn/workflows/.github/workflows/Ubuntu-x64-gcc.yml/badge.svg?branch=master)](https://github.com/mapnn/mapnn)

## 1. Introduction

mapnn is designed to combine the strengths of different high-performance neutral network, such as ncnn, MNN or Tengine.  With this framework, one  can easily test the performance of each kernel from different inference frameworks.  Of cause, this framework provide a simple method that map some operator(defined by training) to kernel(defined by inference).  So that user given the ability to choose the best kernel for inference task throught the map.

## 2. Support most commonly used CNN network

* caffe: alexnet, googlenet inceptionv3/v4 mobilenetv1/v2, resnet, vgg16
* onnx : alexnet, googlenet inceptionv1/v2 mobilenetv2, resnet, vgg16, shufflenet1.1, yolov2

## 3. Build status matrix

| System         | armv7 | armv8 | x86 | amd64 |
| :------------: | :---: | :---: | :--: | :--: |
| Ubuntu(GCC)    | — | — | — | ![Build Status](https://github.com/mapnn/mapnn/workflows/.github/workflows/Ubuntu-x64-gcc.yml/badge.svg?branch=master) |
| Ubuntu(Clang)  | — | — | — | ![Build Status](https://github.com/mapnn/mapnn/workflows/.github/workflows/Ubuntu-x64-clang.yml/badge.svg?branch=master) |
| Linux          | ![Build Status](https://github.com/mapnn/mapnn/workflows/.github/workflows/linux-armv7l.yml/badge.svg?branch=master) | ![Build Status](https://github.com/mapnn/mapnn/workflows/.github/workflows/Linux-armv8.yml/badge.svg?branch=master) | — | — |
| Windows(MSVC)  | — | — | ![Build Status](https://github.com/mapnn/mapnn/workflows/.github/workflows/Windows-x86-msvc.yml/badge.svg?branch=master) | ![Build Status](https://github.com/mapnn/mapnn/workflows/.github/workflows/windows-amd64-msvc.yml/badge.svg?branch=master) |
| Android        | ![Build Status](https://github.com/mapnn/mapnn/workflows/.github/workflows/Android-armv7a.yml/badge.svg?branch=master) | ![Build Status](https://github.com/mapnn/mapnn/workflows/.github/workflows/Android-armv8a.yml/badge.svg?branch=master) | ![Build Status](https://github.com/mapnn/mapnn/workflows/.github/workflows/Android-x86.yml/badge.svg?branch=master) | ![Build Status](https://github.com/mapnn/mapnn/workflows/.github/workflows/Android-x64.yml/badge.svg?branch=master) |
| MacOS(Clang)   | — | — | — | ![Build Status](https://github.com/mapnn/mapnn/workflows/.github/workflows/MacOS-amd64-clang.yml/badge.svg?branch=master) |

## 4. How to build

* [Build for Linux-x86-64](script/linux_x86-64_build.sh)
* [Build for Linux-armv7l](script/linux_armeabi-v7l_build.sh)
* [Build for Linux-armv7hl](script/linux_armeabi-v7hl_build.sh)
* [Build for android-v7a](script/android_armv7a_build.sh)
* [Build for android-v8a](script/android_armv8a_build.sh)

## 5. How to test kernel

According to the step 4, Compiling and install this library with your toolchain.  And copy(if need) the layer_test to your platform.  Then run ther layer_test to get perf.txt.

```sh
./layer_test conv -h
# Usage: layer_test conv -k 3 -s 3 -c 1 -g 2
# Option:
#       -k kernel INT
#       -s stride INT
#       -c cycle  INT
#       -g gap    INT
#       -o output INT
#       -h this help
```

## 6. How to use the API

```c++
int ret; 
mapnn::Net* net = new mapnn::Net(); // new object
ret = net->load("model_path.onnx"); // load model
//ret = net->load("model_path.proto", "model_path.caffemodel");
ret = net->prepare(3, 224, 224);    // prepare net
ret = net->inference(float_data, 3, 224, 224); // inference
Tensor& output = net->getTensor("output_name");
delete net;
```

## 7. How to contribute

* Add new operation from other training frameworks.
* Add new kernel from other inference frameworks.
* Improve frameworks and do good pull Request.
* Fix and Report the issue on the Github issues page. 
* Star or fork this project.

> * notice A: if you contribute a good code, please append(or apply for new code) the follow boilerplate notice to your code:
>        Copyright [yyyy] [name of copyright owner]
>
> * notice B: mapnn include some third party libraries as following:
>   a. [3rdparty/flatbuffers](3rdparty/flabuffers/LICENSE.txt)
>   b. [3rdparty/MNN](3rdparty/README.md)
>   c. [3rdparty/ncnn](3rdparty/ncnn/LICENSE.txt)
>   d. [3rdparty/protobuf](3rdparty/protobuf/LICENSE)
>   e. [3rdparty/stb](3rdparty/stb/LICENSE)
>   f. [3rdparty/Tengine](3rdparty/tengine/LICENSE)

