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
static void conv1x1s1_sgemm_transform_kernel_pack4_neon(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch)
{
    // interleave
    // src = inch-outch
    // dst = 4b-4a-inch/4a-outch/4b
#if __aarch64__
    kernel_tm_pack4.create(2 * 1, inch/4, (outch/4)/2 + (outch/4)%2, (size_t)4u*16, 16);
#else
    kernel_tm_pack4.create(1, inch/4, outch/4, (size_t)4u*16, 16);
#endif

    int q=0;
#if __aarch64__
    for (; q+7<outch; q+=8)
    {
        const float* k0 = (const float*)kernel + (q+0)*inch;
        const float* k1 = (const float*)kernel + (q+1)*inch;
        const float* k2 = (const float*)kernel + (q+2)*inch;
        const float* k3 = (const float*)kernel + (q+3)*inch;
        const float* k4 = (const float*)kernel + (q+4)*inch;
        const float* k5 = (const float*)kernel + (q+5)*inch;
        const float* k6 = (const float*)kernel + (q+6)*inch;
        const float* k7 = (const float*)kernel + (q+7)*inch;

        float* g0 = kernel_tm_pack4.channel(q/8);

        for (int p=0; p+3<inch; p+=4)
        {
            g0[0] = k0[0];
            g0[1] = k1[0];
            g0[2] = k2[0];
            g0[3] = k3[0];

            g0[4] = k4[0];
            g0[5] = k5[0];
            g0[6] = k6[0];
            g0[7] = k7[0];

            g0[8] = k0[1];
            g0[9] = k1[1];
            g0[10] = k2[1];
            g0[11] = k3[1];

            g0[12] = k4[1];
            g0[13] = k5[1];
            g0[14] = k6[1];
            g0[15] = k7[1];

            g0[16] = k0[2];
            g0[17] = k1[2];
            g0[18] = k2[2];
            g0[19] = k3[2];

            g0[20] = k4[2];
            g0[21] = k5[2];
            g0[22] = k6[2];
            g0[23] = k7[2];

            g0[24] = k0[3];
            g0[25] = k1[3];
            g0[26] = k2[3];
            g0[27] = k3[3];

            g0[28] = k4[3];
            g0[29] = k5[3];
            g0[30] = k6[3];
            g0[31] = k7[3];

            k0 += 4;
            k1 += 4;
            k2 += 4;
            k3 += 4;
            k4 += 4;
            k5 += 4;
            k6 += 4;
            k7 += 4;
            g0 += 32;
        }
    }
#endif // __aarch64__
    for (; q+3<outch; q+=4)
    {
        const float* k0 = (const float*)kernel + (q+0)*inch;
        const float* k1 = (const float*)kernel + (q+1)*inch;
        const float* k2 = (const float*)kernel + (q+2)*inch;
        const float* k3 = (const float*)kernel + (q+3)*inch;

#if __aarch64__
        float* g0 = kernel_tm_pack4.channel(q/8+(q%8)/4);
#else
        float* g0 = kernel_tm_pack4.channel(q/4);
#endif

        for (int p=0; p+3<inch; p+=4)
        {
            g0[0] = k0[0];
            g0[1] = k1[0];
            g0[2] = k2[0];
            g0[3] = k3[0];

            g0[4] = k0[1];
            g0[5] = k1[1];
            g0[6] = k2[1];
            g0[7] = k3[1];

            g0[8] = k0[2];
            g0[9] = k1[2];
            g0[10] = k2[2];
            g0[11] = k3[2];

            g0[12] = k0[3];
            g0[13] = k1[3];
            g0[14] = k2[3];
            g0[15] = k3[3];

            k0 += 4;
            k1 += 4;
            k2 += 4;
            k3 += 4;
            g0 += 16;
        }
    }
}
}
