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
#ifndef __MAPNN_TENGINE_KERNEL_H__
#define __MAPNN_TENGINE_KERNEL_H__

#include "reference.h"
#if defined(NNOPM_USE_NEON32)
#include <arm_neon.h>
DECLARE_KERNEL_BASE(tengine_conv_2d_direct_3x3_dilation, RefConv)
DECLARE_KERNEL_BASE(tengine_conv_2d_dw_3x3, RefConv)
DECLARE_KERNEL_BASE(tengine_conv_2d_dw_dilation, RefConv)
DECLARE_KERNEL_BASE(tengine_conv_2d_dw, RefConv)
DECLARE_KERNEL_BASE(tengine_conv_2d_dw_k5s1, RefConv)
DECLARE_KERNEL_BASE(tengine_conv_2d_dw_k5s2, RefConv)
DECLARE_KERNEL_BASE(tengine_conv_2d_dw_k7s1, RefConv)
DECLARE_KERNEL_BASE(tengine_conv_2d_dw_k7s2, RefConv)
//DECLARE_KERNEL_BASE(tengine_conv_fast_direct, RefConv)
//DECLARE_KERNEL(tengine_conv_fast_interleave)
//DECLARE_KERNEL(tengine_conv_fast_gemm)
DECLARE_KERNEL(tengine_conv_2d_wino_interleave)
DECLARE_KERNEL(tengine_conv_2d_wino)
#else
#error("none")
#endif
#endif // __MAPNN_TENGINE_KERNEL_H__
