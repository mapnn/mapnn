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
#ifndef __MAPNN_MNN_KERNEL_H__
#define __MAPNN_MNN_KERNEL_H__


#include "reference.h"
// x86
#ifdef MCNN_USE_SSE
#include <immintrin.h>
#define MNN_USE_SSE
#elif defined(MCNN_USE_NEON32) || defined(MCNN_USE_NEON64)
#include <arm_neon.h>
#define MNN_USE_NEON
#else
#error("none")
#endif

DECLARE_KERNEL(mnn_ConvolutionDepthwise3x3)
DECLARE_KERNEL(mnn_ConvolutionTiledExecutorBasic)
DECLARE_KERNEL(mnn_ConvolutionTiledExecutorBasic2)
DECLARE_KERNEL(mnn_ConvolutionWinogradF23)
DECLARE_KERNEL(mnn_ConvolutionWinogradF63)
DECLARE_KERNEL(mnn_convolution3x3_gemm)
DECLARE_KERNEL(mnn_depthwise3x3Weight)
DECLARE_KERNEL(mnn_reorderWeight)
DECLARE_KERNEL(mnn_transformWeightF23)
DECLARE_KERNEL(mnn_transformWeightF63)

#endif // __MAPNN_MNN_KERNEL_H__
