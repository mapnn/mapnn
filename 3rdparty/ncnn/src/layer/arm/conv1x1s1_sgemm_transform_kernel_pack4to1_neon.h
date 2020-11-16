// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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
static void conv1x1s1_sgemm_transform_kernel_pack4to1_neon(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch)
{
    // interleave
    // src = inch-outch
    // dst = 4a-inch/4a-outch
#if __aarch64__
    kernel_tm_pack4.create(8, inch/4, outch/8 + (outch%8)/4 + outch%4, (size_t)4u*4, 4);
#else
    kernel_tm_pack4.create(4, inch/4, outch/4 + outch%4, (size_t)4u*4, 4);
#endif

    int p=0;
#if __aarch64__
    for (; p+7<outch; p+=8)
    {
        const float* k0 = (const float*)kernel + (p+0)*inch;
        const float* k1 = (const float*)kernel + (p+1)*inch;
        const float* k2 = (const float*)kernel + (p+2)*inch;
        const float* k3 = (const float*)kernel + (p+3)*inch;
        const float* k4 = (const float*)kernel + (p+4)*inch;
        const float* k5 = (const float*)kernel + (p+5)*inch;
        const float* k6 = (const float*)kernel + (p+6)*inch;
        const float* k7 = (const float*)kernel + (p+7)*inch;

        float* ktmp = kernel_tm_pack4.channel(p/8);

        for (int q=0; q+3<inch; q+=4)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp[4] = k4[0];
            ktmp[5] = k5[0];
            ktmp[6] = k6[0];
            ktmp[7] = k7[0];

            ktmp[8] = k0[1];
            ktmp[9] = k1[1];
            ktmp[10] = k2[1];
            ktmp[11] = k3[1];
            ktmp[12] = k4[1];
            ktmp[13] = k5[1];
            ktmp[14] = k6[1];
            ktmp[15] = k7[1];

            ktmp[16] = k0[2];
            ktmp[17] = k1[2];
            ktmp[18] = k2[2];
            ktmp[19] = k3[2];
            ktmp[20] = k4[2];
            ktmp[21] = k5[2];
            ktmp[22] = k6[2];
            ktmp[23] = k7[2];

            ktmp[24] = k0[3];
            ktmp[25] = k1[3];
            ktmp[26] = k2[3];
            ktmp[27] = k3[3];
            ktmp[28] = k4[3];
            ktmp[29] = k5[3];
            ktmp[30] = k6[3];
            ktmp[31] = k7[3];

            k0 += 4;
            k1 += 4;
            k2 += 4;
            k3 += 4;
            k4 += 4;
            k5 += 4;
            k6 += 4;
            k7 += 4;
            ktmp += 32;
        }
    }
#endif
    for (; p+3<outch; p+=4)
    {
        const float* k0 = (const float*)kernel + (p+0)*inch;
        const float* k1 = (const float*)kernel + (p+1)*inch;
        const float* k2 = (const float*)kernel + (p+2)*inch;
        const float* k3 = (const float*)kernel + (p+3)*inch;

#if __aarch64__
        float* ktmp = kernel_tm_pack4.channel(p/8 + (p%8)/4);
#else
        float* ktmp = kernel_tm_pack4.channel(p/4);
#endif

        for (int q=0; q+3<inch; q+=4)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];

            ktmp[4] = k0[1];
            ktmp[5] = k1[1];
            ktmp[6] = k2[1];
            ktmp[7] = k3[1];

            ktmp[8] = k0[2];
            ktmp[9] = k1[2];
            ktmp[10] = k2[2];
            ktmp[11] = k3[2];

            ktmp[12] = k0[3];
            ktmp[13] = k1[3];
            ktmp[14] = k2[3];
            ktmp[15] = k3[3];

            k0 += 4;
            k1 += 4;
            k2 += 4;
            k3 += 4;
            ktmp += 16;
        }
    }
    for (; p<outch; p++)
    {
        const float* k0 = (const float*)kernel + p*inch;

#if __aarch64__
        float* ktmp = kernel_tm_pack4.channel(p/8 + (p%8)/4 + p%4);
#else
        float* ktmp = kernel_tm_pack4.channel(p/4 + p%4);
#endif

        for (int q=0; q+3<inch; q+=4)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k0[1];
            ktmp[2] = k0[2];
            ktmp[3] = k0[3];

            k0 += 4;
            ktmp += 4;
        }
    }
}
}
