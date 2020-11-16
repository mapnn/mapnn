// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "option.h"
#include "mat.h"
namespace ncnn{
static void conv1x1s1_sgemm_transform_kernel_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    const float* kernel = _kernel;

    // interleave
#if __ARM_NEON && __aarch64__
    kernel_tm.create(4*8, inch/4 + inch%4, outch/8 + (outch%8)/4 + outch%4);
#else
    kernel_tm.create(4*4, inch/4 + inch%4, outch/4 + outch%4);
#endif // __ARM_NEON && __aarch64__

    int p = 0;
#if __ARM_NEON && __aarch64__
    for (; p+7<outch; p+=8)
    {
        const float* kernel0 = kernel + (p+0)*inch;
        const float* kernel1 = kernel + (p+1)*inch;
        const float* kernel2 = kernel + (p+2)*inch;
        const float* kernel3 = kernel + (p+3)*inch;
        const float* kernel4 = kernel + (p+4)*inch;
        const float* kernel5 = kernel + (p+5)*inch;
        const float* kernel6 = kernel + (p+6)*inch;
        const float* kernel7 = kernel + (p+7)*inch;

        float* ktmp = kernel_tm.channel(p/8);

        for (int q=0; q<inch; q++)
        {
            // kernel0...7 0
            ktmp[0] = kernel0[0];
            ktmp[1] = kernel1[0];
            ktmp[2] = kernel2[0];
            ktmp[3] = kernel3[0];
            ktmp[4] = kernel4[0];
            ktmp[5] = kernel5[0];
            ktmp[6] = kernel6[0];
            ktmp[7] = kernel7[0];

            ktmp += 8;
            kernel0 += 1;
            kernel1 += 1;
            kernel2 += 1;
            kernel3 += 1;
            kernel4 += 1;
            kernel5 += 1;
            kernel6 += 1;
            kernel7 += 1;
        }
    }
#endif // __ARM_NEON && __aarch64__
    for (; p+3<outch; p+=4)
    {
        const float* kernel0 = kernel + (p+0)*inch;
        const float* kernel1 = kernel + (p+1)*inch;
        const float* kernel2 = kernel + (p+2)*inch;
        const float* kernel3 = kernel + (p+3)*inch;

#if __ARM_NEON && __aarch64__
        float* ktmp = kernel_tm.channel(p/8 + (p%8)/4);
#else
        float* ktmp = kernel_tm.channel(p/4);
#endif // __ARM_NEON && __aarch64__

        for (int q=0; q<inch; q++)
        {
            // kernel0...3 0
            ktmp[0] = kernel0[0];
            ktmp[1] = kernel1[0];
            ktmp[2] = kernel2[0];
            ktmp[3] = kernel3[0];

            ktmp += 4;
            kernel0 += 1;
            kernel1 += 1;
            kernel2 += 1;
            kernel3 += 1;
        }
    }
    for (; p<outch; p++)
    {
        const float* kernel0 = kernel + p*inch;

#if __ARM_NEON && __aarch64__
        float* ktmp = kernel_tm.channel(p/8 + (p%8)/4 + p%4);
#else
        float* ktmp = kernel_tm.channel(p/4 + p%4);
#endif // __ARM_NEON && __aarch64__

        for (int q=0; q<inch; q++)
        {
            ktmp[0] = kernel0[0];
            ktmp++;
            kernel0++;
        }
    }
}
}
