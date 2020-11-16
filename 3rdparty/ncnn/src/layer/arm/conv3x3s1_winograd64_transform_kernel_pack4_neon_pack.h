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
static void conv3x3s1_winograd64_transform_kernel_pack4_neon_pack(const Mat& kernel_tm, Mat& kernel_tm_pack4, int inch, int outch)
{
    // interleave
    // src = 64-inch-outch
    // dst = 4b-4a-inch/4a-64-outch/4b;
#if __aarch64__
    kernel_tm_pack4.create(2 * inch/4, 64, (outch/4)/2 + (outch/4)%2, (size_t)4u*16, 16);
#else
    kernel_tm_pack4.create(inch/4, 64, outch/4, (size_t)4u*16, 16);
#endif

    int q=0;
#if __aarch64__
    for (; q+7<outch; q+=8)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q+1);
        const Mat k2 = kernel_tm.channel(q+2);
        const Mat k3 = kernel_tm.channel(q+3);
        const Mat k4 = kernel_tm.channel(q+4);
        const Mat k5 = kernel_tm.channel(q+5);
        const Mat k6 = kernel_tm.channel(q+6);
        const Mat k7 = kernel_tm.channel(q+7);

        Mat g0 = kernel_tm_pack4.channel(q/8);

        for (int k=0; k<64; k++)
        {
            float* g00 = g0.row(k);

            for (int p=0; p+3<inch; p+=4)
            {
                const float* k00 = k0.row(p);
                const float* k01 = k0.row(p+1);
                const float* k02 = k0.row(p+2);
                const float* k03 = k0.row(p+3);

                const float* k10 = k1.row(p);
                const float* k11 = k1.row(p+1);
                const float* k12 = k1.row(p+2);
                const float* k13 = k1.row(p+3);

                const float* k20 = k2.row(p);
                const float* k21 = k2.row(p+1);
                const float* k22 = k2.row(p+2);
                const float* k23 = k2.row(p+3);

                const float* k30 = k3.row(p);
                const float* k31 = k3.row(p+1);
                const float* k32 = k3.row(p+2);
                const float* k33 = k3.row(p+3);

                const float* k40 = k4.row(p);
                const float* k41 = k4.row(p+1);
                const float* k42 = k4.row(p+2);
                const float* k43 = k4.row(p+3);

                const float* k50 = k5.row(p);
                const float* k51 = k5.row(p+1);
                const float* k52 = k5.row(p+2);
                const float* k53 = k5.row(p+3);

                const float* k60 = k6.row(p);
                const float* k61 = k6.row(p+1);
                const float* k62 = k6.row(p+2);
                const float* k63 = k6.row(p+3);

                const float* k70 = k7.row(p);
                const float* k71 = k7.row(p+1);
                const float* k72 = k7.row(p+2);
                const float* k73 = k7.row(p+3);

                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00[4] = k40[k];
                g00[5] = k50[k];
                g00[6] = k60[k];
                g00[7] = k70[k];

                g00[8] = k01[k];
                g00[9] = k11[k];
                g00[10] = k21[k];
                g00[11] = k31[k];

                g00[12] = k41[k];
                g00[13] = k51[k];
                g00[14] = k61[k];
                g00[15] = k71[k];

                g00[16] = k02[k];
                g00[17] = k12[k];
                g00[18] = k22[k];
                g00[19] = k32[k];

                g00[20] = k42[k];
                g00[21] = k52[k];
                g00[22] = k62[k];
                g00[23] = k72[k];

                g00[24] = k03[k];
                g00[25] = k13[k];
                g00[26] = k23[k];
                g00[27] = k33[k];

                g00[28] = k43[k];
                g00[29] = k53[k];
                g00[30] = k63[k];
                g00[31] = k73[k];

                g00 += 32;
            }
        }
    }
#endif // __aarch64__
    for (; q+3<outch; q+=4)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q+1);
        const Mat k2 = kernel_tm.channel(q+2);
        const Mat k3 = kernel_tm.channel(q+3);

#if __aarch64__
        Mat g0 = kernel_tm_pack4.channel(q/8+(q%8)/4);
#else
        Mat g0 = kernel_tm_pack4.channel(q/4);
#endif

        for (int k=0; k<64; k++)
        {
            float* g00 = g0.row(k);

            for (int p=0; p+3<inch; p+=4)
            {
                const float* k00 = k0.row(p);
                const float* k01 = k0.row(p+1);
                const float* k02 = k0.row(p+2);
                const float* k03 = k0.row(p+3);

                const float* k10 = k1.row(p);
                const float* k11 = k1.row(p+1);
                const float* k12 = k1.row(p+2);
                const float* k13 = k1.row(p+3);

                const float* k20 = k2.row(p);
                const float* k21 = k2.row(p+1);
                const float* k22 = k2.row(p+2);
                const float* k23 = k2.row(p+3);

                const float* k30 = k3.row(p);
                const float* k31 = k3.row(p+1);
                const float* k32 = k3.row(p+2);
                const float* k33 = k3.row(p+3);

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
}
