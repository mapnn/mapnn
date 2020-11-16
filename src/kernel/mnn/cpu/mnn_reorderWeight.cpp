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

#include "mnn_kernel.h"

#include <backend/cpu/compute/CommonOptFunction.h>
#include <backend/cpu/compute/ConvOpt.h>
#include <core/Macro.h>
#include <core/Concurrency.h>
#include <math/Vec4.hpp>

#include "conv.h"

using namespace MNN;
using namespace MNN::Math;

void mnn_reorderWeight::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LUVAB output(out);
    L111W input(ins[0]); 
    LUVAB temp(tmp[0]);
    const int maxk = conv.wkernel*conv.hkernel;
    int inch = conv.inch;
    int outch = conv.outch;
    output.u = (outch+3)/4;
    output.v = (inch+3)/4;
    output.a = maxk;
    output.b = 16;
    temp.u = (outch+3)/4;
    temp.v = (inch+3)/4;
    temp.a = maxk;
    temp.b = 16;
}
void mnn_reorderWeight::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LUVAB output(out);
    L111W input(ins[0]); 
    LUVAB temp(tmp[0]);

    auto depth = conv.inch;
    auto outputCount = conv.outch;
    auto kernelSize = conv.wkernel*conv.hkernel;
    auto alignDepth = ALIGN_UP4(depth);
    for (int b = 0; b < outputCount; ++b) {
        auto dst = temp.data + b * alignDepth * kernelSize;
        auto src = input.data + b * depth * kernelSize;
        MNNPackC4(dst, src, kernelSize, depth);
    }
    MNNPackC4(output.data, temp.data, kernelSize * ALIGN_UP4(depth), outputCount);
    auto count = UP_DIV(depth, 4) * kernelSize * UP_DIV(outputCount, 4);
    MNNReorder4x4ByPlatform(output.data, count);
}
