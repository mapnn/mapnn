/* Copyright 2020 The Mapnn Team. All Rights Reserved. 
 *                                                                            
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *                                                                            
 *     http://www.apache.org/licenses/LICENSE-2.0
 *                                                                            
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __MAPNN_NCNN_KERNEL_H__
#define __MAPNN_NCNN_KERNEL_H__

#include "reference.h"
#if defined(MAPNN_USE_NEON32) || defined(MAPNN_USE_NEON64)
#include <arm_neon.h>
#define __ARM_NEON 1
#if defined(MAPNN_USE_NEON64)
#define __aarch64__ 1
#endif
DECLARE_KERNEL_BASE(ncnn_conv1x1s1_neon, RefConv)                      
DECLARE_KERNEL(ncnn_conv1x1s1_sgemm_neon)
DECLARE_KERNEL(ncnn_conv1x1s1_sgemm_pack4_neon)
DECLARE_KERNEL(ncnn_conv1x1s1_sgemm_pack4to1_neon)
DECLARE_KERNEL(ncnn_conv1x1s1_sgemm_transform_kernel_neon)
DECLARE_KERNEL(ncnn_conv1x1s1_sgemm_transform_kernel_pack4_neon)
DECLARE_KERNEL(ncnn_conv1x1s1_sgemm_transform_kernel_pack4to1_neon)
DECLARE_KERNEL_BASE(ncnn_conv1x1s2_neon, RefConv)
DECLARE_KERNEL(ncnn_conv1x1s2_pack4_neon)
DECLARE_KERNEL(ncnn_conv1x1s2_pack4to1_neon)
DECLARE_KERNEL_BASE(ncnn_conv2x2s1_neon, RefConv)
DECLARE_KERNEL_BASE(ncnn_conv3x3s1_neon, RefConv)
DECLARE_KERNEL(ncnn_conv3x3s1_pack1to4_neon)
DECLARE_KERNEL(ncnn_conv3x3s1_winograd64_neon4)
DECLARE_KERNEL(ncnn_conv3x3s1_winograd64_neon5)
DECLARE_KERNEL(ncnn_conv3x3s1_winograd64_pack4_neon)
DECLARE_KERNEL(ncnn_conv3x3s1_winograd64_transform_kernel_neon)
DECLARE_KERNEL(ncnn_conv3x3s1_winograd64_transform_kernel_neon5)
DECLARE_KERNEL(ncnn_conv3x3s1_winograd64_transform_kernel_pack4_neon)
DECLARE_KERNEL_BASE(ncnn_conv3x3s2_neon, RefConv)
DECLARE_KERNEL(ncnn_conv3x3s2_pack1to4_neon)
DECLARE_KERNEL(ncnn_conv3x3s2_pack4_neon)
DECLARE_KERNEL_BASE(ncnn_conv3x3s2_packed_neon, RefConv)
DECLARE_KERNEL(ncnn_conv3x3s2_transform_kernel_neon)
DECLARE_KERNEL_BASE(ncnn_conv4x4s4_neon, RefConv)
DECLARE_KERNEL_BASE(ncnn_conv5x5s1_neon, RefConv)
DECLARE_KERNEL(ncnn_conv5x5s1_pack4_neon)
DECLARE_KERNEL_BASE(ncnn_conv5x5s2_neon, RefConv)
DECLARE_KERNEL(ncnn_conv5x5s2_pack4_neon)
DECLARE_KERNEL_BASE(ncnn_conv7x7s1_neon, RefConv)
DECLARE_KERNEL_BASE(ncnn_conv7x7s2_neon, RefConv)
DECLARE_KERNEL(ncnn_conv7x7s2_pack1to4_neon)
DECLARE_KERNEL(ncnn_conv_im2col_sgemm_neon)
DECLARE_KERNEL(ncnn_conv_im2col_sgemm_transform_kernel_neon)
DECLARE_KERNEL(ncnn_conv_weight_pack1x4_neon)
DECLARE_KERNEL(ncnn_conv_weight_pack4x1_neon)
DECLARE_KERNEL(ncnn_conv_weight_pack4x4_neon)
DECLARE_KERNEL_BASE(ncnn_convdw3x3s1_neon, RefConv)
DECLARE_KERNEL(ncnn_convdw3x3s1_pack4_neon)
DECLARE_KERNEL_BASE(ncnn_convdw3x3s2_neon, RefConv)
DECLARE_KERNEL(ncnn_convdw3x3s2_pack4_neon)
DECLARE_KERNEL_BASE(ncnn_convdw5x5s1_neon, RefConv)
DECLARE_KERNEL(ncnn_convdw5x5s1_pack4_neon)
DECLARE_KERNEL_BASE(ncnn_convdw5x5s2_neon, RefConv)
DECLARE_KERNEL(ncnn_convdw5x5s2_pack4_neon)
DECLARE_KERNEL_BASE(ncnn_eltwise_add_neon, RefAdd)
DECLARE_KERNEL_BASE(ncnn_pooling2x2s2_max_neon, RefMaxPool)
DECLARE_KERNEL_BASE(ncnn_pooling3x3s2_max_neon, RefMaxPool)
#else
#error("none")
#endif
#endif // __MAPNN_NCNN_KERNEL_H__
