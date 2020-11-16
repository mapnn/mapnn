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

#ifndef __MAPNN_REFERENCE_H__
#define __MAPNN_REFERENCE_H__

#include "kernel.h"

// Not Support.
#include "priorbox.h"
#include "argmax.h"

// Support

DECLARE_KERNEL(RefMul)
DECLARE_KERNEL(RefNop)
DECLARE_KERNEL(RefSigmoid)
DECLARE_KERNEL(RefCHW1ToCHW4)
DECLARE_KERNEL(RefCHW4ToCHW1)
DECLARE_KERNEL(RefAdd)
DECLARE_KERNEL(RefGlobalAvgPool)
DECLARE_KERNEL(RefGlobalMaxPool)
DECLARE_KERNEL(RefConcat)

#include "pool.h"
DECLARE_KERNEL(RefAvgPool)
DECLARE_KERNEL(RefMaxPool)

#include "conv.h"
DECLARE_KERNEL(RefConv)
DECLARE_KERNEL(RefConvTranspose)

#include "crop.h"
DECLARE_KERNEL(RefCrop)

#include "flatten.h"
DECLARE_KERNEL(RefFlatten)

#include "gemm.h"
DECLARE_KERNEL(RefGemm)

#include "pad.h"
DECLARE_KERNEL(RefPad)
#include "reshape.h"

DECLARE_KERNEL(RefReshape)
#include "transpose.h"

DECLARE_KERNEL(RefTranspose)
#include "slice.h"
DECLARE_KERNEL(RefSlice)

DECLARE_KERNEL_BASE(RefBatchnormal, RefNop)
DECLARE_KERNEL_BASE(RefBias,  RefNop)
DECLARE_KERNEL_BASE(RefDiv, RefNop)
DECLARE_KERNEL_BASE(RefMath, RefNop)

#include "clip.h"
DECLARE_KERNEL_BASE(RefClip, RefNop)

#include "dropout.h"
DECLARE_KERNEL_BASE(RefDropout, RefNop)

#include "leakyrelu.h"
DECLARE_KERNEL_BASE(RefLeakyRelu, RefNop)

#include "elu.h"
DECLARE_KERNEL_BASE(RefElu, RefNop)

#include "shuffle_channel.h"
DECLARE_KERNEL_BASE(RefShuffleChannel, RefNop)

#include "LRN.h"
DECLARE_KERNEL_BASE(RefLRN, RefNop)

#include "MVN.h"
DECLARE_KERNEL_BASE(RefMVN,  RefNop)

#include "power.h"
DECLARE_KERNEL_BASE(RefPow, RefNop)
DECLARE_KERNEL_BASE(RefPower, RefNop)

#include "prelu.h"
DECLARE_KERNEL_BASE(RefRelu, RefNop)
DECLARE_KERNEL_BASE(RefRelu6, RefNop)
DECLARE_KERNEL_BASE(RefPRelu, RefNop)
DECLARE_KERNEL_BASE(RefScale, RefNop)
DECLARE_KERNEL_BASE(RefSoftmax, RefNop)
DECLARE_KERNEL_BASE(RefMin, RefNop)
DECLARE_KERNEL_BASE(RefMax, RefNop)
#endif // __MAPNN_REFERENCE_H__
