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
void ncnn_conv3x3s2_transform_kernel_neon::init(const Tensors& /*ins*/, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1VAB output(out);
    int inch = conv.inch;
    int outch = conv.outch;
    output.u = outch/8 + outch%8;
    output.v = inch;
    output.a = 8*9;
}
void ncnn_conv3x3s2_transform_kernel_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1VAB input(ins[0]); 
    L1VAB output(out); 

    //int w = input.w;
    int inch = conv.inch;
    int outch = conv.outch;

    const float* kernel = input.data;

    int p=0;
    for (; p+7<outch; p+=8)
    {
        const float* k0 = kernel + (p+0)*inch*9;
        const float* k1 = kernel + (p+1)*inch*9;
        const float* k2 = kernel + (p+2)*inch*9;
        const float* k3 = kernel + (p+3)*inch*9;
        const float* k4 = kernel + (p+4)*inch*9;
        const float* k5 = kernel + (p+5)*inch*9;
        const float* k6 = kernel + (p+6)*inch*9;
        const float* k7 = kernel + (p+7)*inch*9;

        float* ktmp = output.data + output.va*(p/8);

        for (int q=0; q<inch; q++)
        {
            for (int k=0; k<9; k++)
            {
                ktmp[0] = k0[k];
                ktmp[1] = k1[k];
                ktmp[2] = k2[k];
                ktmp[3] = k3[k];
                ktmp[4] = k4[k];
                ktmp[5] = k5[k];
                ktmp[6] = k6[k];
                ktmp[7] = k7[k];
                ktmp += 8;
            }

            k0 += 9;
            k1 += 9;
            k2 += 9;
            k3 += 9;
            k4 += 9;
            k5 += 9;
            k6 += 9;
            k7 += 9;
        }
    }
    for (; p<outch; p++)
    {
        const float* k0 = kernel + (p+0)*inch*9;

        float* ktmp = output.data + output.va*(p/8 + p%8);

        for (int q=0; q<inch; q++)
        {
            for (int k=0; k<9; k++)
            {
                ktmp[k] = k0[k];
            }
            ktmp += 9;

            k0 += 9;
        }
    }

}
