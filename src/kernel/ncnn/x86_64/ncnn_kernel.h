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
// x86
#ifdef MAPNN_USE_SSE
#include <immintrin.h>
DECLARE_KERNEL(ncnn_conv3x3s1_winograd23_sse)
DECLARE_KERNEL(ncnn_conv3x3s1_winograd23_transform_kernel_sse)
DECLARE_KERNEL(ncnn_conv_im2col_sgemm_sse)
DECLARE_KERNEL(ncnn_conv_im2col_sgemm_transform_kernel_sse)

DECLARE_KERNEL_BASE(ncnn_conv1x1s1_sse, RefConv)
DECLARE_KERNEL_BASE(ncnn_conv1x1s2_sse, RefConv)
DECLARE_KERNEL_BASE(ncnn_conv3x3s1_sse, RefConv)
DECLARE_KERNEL_BASE(ncnn_conv3x3s2_sse, RefConv)
DECLARE_KERNEL_BASE(ncnn_conv5x5s1_sse, RefConv)
DECLARE_KERNEL_BASE(ncnn_convdw3x3s1_sse, RefConv)
DECLARE_KERNEL_BASE(ncnn_convdw3x3s2_sse, RefConv)
#else
#error("none")
#endif
#endif // __MAPNN_NCNN_KERNEL_H__
