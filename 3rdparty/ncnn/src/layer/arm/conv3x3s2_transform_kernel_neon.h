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
static void conv3x3s2_transform_kernel_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(8*9, inch, outch/8 + outch%8);

    const float* kernel = _kernel;

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

        float* ktmp = kernel_tm.channel(p/8);

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

        float* ktmp = kernel_tm.channel(p/8 + p%8);

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
}
