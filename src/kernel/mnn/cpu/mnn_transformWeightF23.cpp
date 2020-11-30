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
void mnn_transformWeightF23::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LUVAB output(out);
    L111W input(ins[0]); 
    int ci = conv.inch;
    int co = conv.outch;
    int alpha = 4;
    output.u = alpha*alpha;
    output.v = UP_DIV(co, 4);
    output.a = UP_DIV(ci, 4);
    output.b = 16;

}
void mnn_transformWeightF23::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LUVAB output(out);
    L111W input(ins[0]); 

    const float G[] = {
           1.0f,    0.0f,    0.0f,
           1.0f,    0.5f,    0.25f,
           1.0f,   -0.5f,    0.25f,
           0.0f,    0.0f,    1.0f
    };
    const float GT[] = {
           1.0f, 1.0f,  1.0f, 0.0f,
           0.0f, 0.5f, -0.5f, 0.0f,
           0.0f, 0.25f, 0.25f, 1.0f
    };
    int ci          = conv.inch;
    int co          = conv.outch;
    int kernelCount = conv.hkernel;
    int unitCi      = 4;
    int unitCo      = 4;
    auto alpha      = 4;
    auto alpha2     = alpha * alpha;
    float M[12];
    float KT[16];

    if (ci % unitCi != 0 || co % unitCo != 0) {
        ::memset(output.data, 0, output.uvab*sizeof(float));
    }
    auto weightPtr      = input.data;
    for (int oz = 0; oz < co; ++oz) {
        auto srcOz = weightPtr + oz * ci * kernelCount * kernelCount;

        int ozC4 = oz / unitCo;
        int mx   = oz % unitCo;

        auto dstOz = output.data + output.ab * ozC4 + mx;
        for (int sz = 0; sz < ci; ++sz) {
            int szC4         = sz / unitCi;
            int my           = sz % unitCi;
            auto srcSz       = srcOz + kernelCount * kernelCount * sz;

            // M = G * K
            Math::Matrix::multi(M, G, srcSz, 4, 3, 3);
            // K_Transform = M*GT
            Math::Matrix::multi(KT, M, GT, 4, 3, 4);

            auto dstSz = dstOz + szC4 * output.b + unitCo * my;

            for (int i = 0; i < alpha2; ++i) {
                dstSz[i * output.vab] = KT[i];
            }
        }
    }
    int ic4 = UP_DIV(ci, 4);
    int oc4 = UP_DIV(co, 4);
    MNNReorder4x4ByPlatform(output.data, ic4 * oc4 * alpha2);
}
}
