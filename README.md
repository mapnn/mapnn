mapnn
---
[![License](https://img.shields.io/badge/license-Apache2.0-blue)](https://github.com/mapnn/mapnn) 
![Build](https://github.com/mapnn/mapnn/workflows/.github/workflows/linux-amd64-gcc.yml/badge.svg?branch=main&event=status)

## 1. Introduction

mapnn is designed to combine the strengths of different high-performance neutral network, such as ncnn, MNN or Tengine.  With this framework, one  can easily test the performance of each kernel from different inference frameworks.  Of cause, this framework provide a simple method that map some operator(defined by training) to kernel(defined by inference).  So that user given the ability to choose the best kernel for inference task throught the map.

## 2. Support most commonly used CNN network

* caffe: alexnet, googlenet inceptionv3/v4 mobilenetv1/v2, resnet, vgg16
* onnx : alexnet, googlenet inceptionv1/v2 mobilenetv2, resnet, vgg16, shufflenet1.1, yolov2, vgg

## 3. Build status matrix

| System         | armv7 | armv8 | x86 | amd64 |
| :------------: | :---: | :---: | :--: | :--: |
| Ubuntu(GCC)    | — | — | — | ![.github/workflows/linux-amd64-gcc.yml](https://github.com/mapnn/mapnn/workflows/.github/workflows/linux-amd64-gcc.yml/badge.svg?branch=main&event=status) |
| Ubuntu(Clang)  | — | — | — | ![.github/workflows/Ubuntu-x64-clang.yml](https://github.com/mapnn/mapnn/workflows/.github/workflows/Ubuntu-x64-clang.yml/badge.svg?branch=master) |
| Linux          | ![.github/workflows/linux-armv7l.yml](https://github.com/mapnn/mapnn/workflows/.github/workflows/linux-armv7l.yml/badge.svg?branch=master) | ![.github/workflows/Linux-armv8.yml](https://github.com/mapnn/mapnn/workflows/.github/workflows/Linux-armv8.yml/badge.svg?branch=master) | — | — |
| Windows(MSVC)  | — | — | ![.github/workflows/Windows-x86-msvc.yml](https://github.com/mapnn/mapnn/workflows/.github/workflows/Windows-x86-msvc.yml/badge.svg?branch=master) | ![.github/workflows/windows-amd64-msvc.yml](https://github.com/mapnn/mapnn/workflows/.github/workflows/windows-amd64-msvc.yml/badge.svg?branch=master) |
| Android        | ![.github/workflows/Android-armv7a.yml](https://github.com/mapnn/mapnn/workflows/.github/workflows/Android-armv7a.yml/badge.svg?branch=master) | ![.github/workflows/Android-armv8a.yml](https://github.com/mapnn/mapnn/workflows/.github/workflows/Android-armv8a.yml/badge.svg?branch=master) | ![.github/workflows/Android-x86.yml](https://github.com/mapnn/mapnn/workflows/.github/workflows/Android-x86.yml/badge.svg?branch=master) | ![.github/workflows/Android-x64.yml](https://github.com/mapnn/mapnn/workflows/.github/workflows/Android-x64.yml/badge.svg?branch=master) |
| MacOS(Clang)   | — | — | — | ![.github/workflows/MacOS-amd64-clang.yml](https://github.com/mapnn/mapnn/workflows/.github/workflows/MacOS-amd64-clang.yml/badge.svg?branch=master) |

## 4. How To build

* [Build for Linux-x86-64](script/Linux_x86-64_build.sh)
* [Build for android-v7a](script/Android_armv7a_build.sh)
* [Build for android-v8a](script/Android_armv8a_build.sh)

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

## 7. How to contribution 

* Add new operation from other training frameworks.
* Add new kernel from other inference frameworks.
* Improve this frameworks and do good pull Request..
* Fix and Report the issue on the Github issues page. 
* Star this project.
