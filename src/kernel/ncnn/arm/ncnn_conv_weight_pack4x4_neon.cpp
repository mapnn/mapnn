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

#include "ncnn_kernel.h"
void ncnn_conv_weight_pack4x4_neon::init(const Tensors& /*ins*/, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LUVAB output(out);
    const int maxk = conv.wkernel*conv.hkernel;
    int inch = conv.inch;
    int outch = conv.outch;
    output.u = (outch+3)/4;
    output.v = (inch+3)/4;
    output.a = maxk;
    output.b = 16;
}
void ncnn_conv_weight_pack4x4_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L111W input(ins[0]); 
    LUVAB output(out); 
    const int maxk = conv.wkernel*conv.hkernel;
    for (int q=0; q+3<conv.outch; q+=4) {
        const float* k0 = input.data + 9*conv.inch*(q);
        const float* k1 = input.data + 9*conv.inch*(q+1);
        const float* k2 = input.data + 9*conv.inch*(q+2);
        const float* k3 = input.data + 9*conv.inch*(q+3);

        float* g0 = output.data + output.vab*(q/4);

        for (int p=0; p+3<conv.inch; p+=4)
        {
            const float* k00 = k0 + 9*(p);
            const float* k01 = k0 + 9*(p+1);
            const float* k02 = k0 + 9*(p+2);
            const float* k03 = k0 + 9*(p+3);

            const float* k10 = k1 + 9*(p);
            const float* k11 = k1 + 9*(p+1);
            const float* k12 = k1 + 9*(p+2);
            const float* k13 = k1 + 9*(p+3);

            const float* k20 = k2 + 9*(p);
            const float* k21 = k2 + 9*(p+1);
            const float* k22 = k2 + 9*(p+2);
            const float* k23 = k2 + 9*(p+3);

            const float* k30 = k3 + 9*(p);
            const float* k31 = k3 + 9*(p+1);
            const float* k32 = k3 + 9*(p+2);
            const float* k33 = k3 + 9*(p+3);

            float* g00 = g0 + output.ab*(p/4);

            for (int k=0; k<maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00[4] = k01[k];
                g00[5] = k11[k];
                g00[6] = k21[k];
                g00[7] = k31[k];

                g00[8] = k02[k];
                g00[9] = k12[k];
                g00[10] = k22[k];
                g00[11] = k32[k];

                g00[12] = k03[k];
                g00[13] = k13[k];
                g00[14] = k23[k];
                g00[15] = k33[k];

                g00 += 16;
            }
        }
    }
}
