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
#include <math/Matrix.hpp>

#include "conv.h"

using namespace MNN;
using namespace MNN::Math;
namespace mapnn {
void mnn_depthwise3x3Weight::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LUVAB output(out);
    L111W input(ins[0]); 
    output.u = UP_DIV(conv.inch, 4);
    output.v = 3;
    output.a = 4;
    output.b = 4;

}
void mnn_depthwise3x3Weight::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LUVAB output(out);
    L111W input(ins[0]); 

    auto channel   = conv.inch;
    //auto channelC4 = UP_DIV(channel, 4);
    auto weightHost = output.data;
    ::memset(weightHost, 0, output.uvab*sizeof(float));

    /* 1D-Winograd F(2,3) and tiling */
    for (int c = 0; c < channel; ++c) {
        auto cIndex     = c / 4;
        auto cRemain    = c % 4;
        auto weightDstZ = weightHost + cIndex * 4 * 4 * 3 + cRemain;
        auto weightSrcZ = input.data + c * 9;
        for (int y = 0; y < 3; ++y) {
            auto k0 = weightSrcZ[3 * y + 0];
            auto k1 = weightSrcZ[3 * y + 1];
            auto k2 = weightSrcZ[3 * y + 2];

            auto m0 = k0;
            auto m1 = 0.5f * (k0 + k1 + k2);
            auto m2 = 0.5f * (k0 - k1 + k2);
            auto m3 = k2;

            weightDstZ[y * 16 + 4 * 0] = m0;
            weightDstZ[y * 16 + 4 * 1] = m1;
            weightDstZ[y * 16 + 4 * 2] = m2;
            weightDstZ[y * 16 + 4 * 3] = m3;
        }
    }

}
}
