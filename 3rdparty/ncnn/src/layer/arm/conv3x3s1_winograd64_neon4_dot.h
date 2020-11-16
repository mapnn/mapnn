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
static void conv3x3s1_winograd64_neon4_dot(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Option& opt,
        int outch, int inch, int outh, int outw)
{
    Mat bottom_blob_tm = bottom_blob;
    Mat top_blob_tm = top_blob;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        top_blob_tm.create(4, 16 * w_tm/8 * h_tm/8, outch, 4u, opt.workspace_allocator);

        const int tiles = h_tm/8 * w_tm/8;

        int nn_outch = outch >> 2;
        int remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 4;

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p+1);
            Mat out2_tm = top_blob_tm.channel(p+2);
            Mat out3_tm = top_blob_tm.channel(p+3);

            const float* ktm = kernel_tm.channel(pp);

            out0_tm.fill(0.f);
            out1_tm.fill(0.f);
            out2_tm.fill(0.f);
            out3_tm.fill(0.f);

            int q = 0;

#if __ARM_NEON && __aarch64__
            for (; q+3<inch; q+=4)
            {
                const float* r0 = bottom_blob_tm.channel(q);
                const float* r1 = bottom_blob_tm.channel(q+1);
                const float* r2 = bottom_blob_tm.channel(q+2);
                const float* r3 = bottom_blob_tm.channel(q+3);

                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;
                float* output2_tm = out2_tm;
                float* output3_tm = out3_tm;

                asm volatile(
                    "mov    w0, #16                     \n"// w0 = r = 16
                    "0:                                 \n"

                    "prfm   pldl1keep, [%8, #512]                       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%8], #64     \n"// v0  v1  v2  v3  = _k00 _k01 _k02 _k03

                    "prfm   pldl1keep, [%8, #512]                       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%8], #64     \n"// v4  v5  v6  v7  = _k10 _k11 _k12 _k13

                    "prfm   pldl1keep, [%8, #512]                       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64   \n"// v8  v9  v10 v11 = _k20 _k21 _k22 _k23

                    "prfm   pldl1keep, [%8, #512]                       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%8], #64 \n"// v12 v13 v14 v15 = _k30 _k31 _k32 _k33

                    // tile loop
                    "lsr    w1, %w18, #2                \n"// w1 = nn = tiles >> 2
                    "cmp    w1, #0                      \n"
                    "beq    2f                          \n"

                    //BEGIN tile loop
                    "prfm   pldl1keep, [%4, #128]       \n"//
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "1:                                 \n"

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v20.4s}, [%0]              \n"
                    "add    x4, %0, #16                 \n"// x4 = %0 next

                    "fmla   v20.4s, v16.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v21.4s}, [%1]              \n"
                    "add    x5, %1, #16                 \n"// x5 = %1 next

                    "fmla   v21.4s, v16.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v22.4s}, [%2]              \n"
                    "add    x6, %2, #16                 \n"// x6 = %2 next

                    "fmla   v22.4s, v16.4s, v8.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v23.4s}, [%3]              \n"
                    "add    x7, %3, #16                 \n"// x7 = %3 next

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v17.4s}, [%5], #16         \n"

                    "fmla   v23.4s, v16.4s, v12.4s      \n"

                    "prfm   pldl1keep, [x4, #128]       \n"
                    "ld1    {v24.4s}, [x4]              \n"

                    "fmla   v20.4s, v17.4s, v1.4s       \n"
                    "fmla   v21.4s, v17.4s, v5.4s       \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v18.4s}, [%6], #16         \n"

                    "fmla   v22.4s, v17.4s, v9.4s       \n"
                    "fmla   v23.4s, v17.4s, v13.4s      \n"

                    "prfm   pldl1keep, [x5, #128]       \n"
                    "ld1    {v25.4s}, [x5]              \n"

                    "fmla   v20.4s, v18.4s, v2.4s       \n"
                    "fmla   v21.4s, v18.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v19.4s}, [%7], #16         \n"

                    "fmla   v22.4s, v18.4s, v10.4s      \n"
                    "fmla   v23.4s, v18.4s, v14.4s      \n"

                    "prfm   pldl1keep, [x6, #128]       \n"
                    "ld1    {v26.4s}, [x6]              \n"

                    "fmla   v20.4s, v19.4s, v3.4s       \n"
                    "fmla   v21.4s, v19.4s, v7.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "fmla   v22.4s, v19.4s, v11.4s      \n"
                    "fmla   v23.4s, v19.4s, v15.4s      \n"

                    ///////

                    "prfm   pldl1keep, [x7, #128]       \n"
                    "ld1    {v27.4s}, [x7]              \n"

                    "st1    {v20.4s}, [%0]              \n"
                    "add    %0, %0, #32                 \n"

                    "fmla   v24.4s, v16.4s, v0.4s       \n"
                    "fmla   v25.4s, v16.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v17.4s}, [%5], #16         \n"

                    "fmla   v26.4s, v16.4s, v8.4s       \n"
                    "fmla   v27.4s, v16.4s, v12.4s      \n"

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v20.4s}, [%0]              \n"

                    "st1    {v21.4s}, [%1]              \n"
                    "add    %1, %1, #32                 \n"

                    "fmla   v24.4s, v17.4s, v1.4s       \n"
                    "fmla   v25.4s, v17.4s, v5.4s       \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v18.4s}, [%6], #16         \n"

                    "fmla   v26.4s, v17.4s, v9.4s       \n"
                    "fmla   v27.4s, v17.4s, v13.4s      \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v21.4s}, [%1]              \n"

                    "st1    {v22.4s}, [%2]              \n"
                    "add    %2, %2, #32                 \n"

                    "fmla   v24.4s, v18.4s, v2.4s       \n"
                    "fmla   v25.4s, v18.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v19.4s}, [%7], #16         \n"

                    "fmla   v26.4s, v18.4s, v10.4s      \n"
                    "fmla   v27.4s, v18.4s, v14.4s      \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v22.4s}, [%2]              \n"

                    "st1    {v23.4s}, [%3]              \n"
                    "add    %3, %3, #32                 \n"

                    "fmla   v24.4s, v19.4s, v3.4s       \n"
                    "fmla   v25.4s, v19.4s, v7.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "fmla   v26.4s, v19.4s, v11.4s      \n"
                    "fmla   v27.4s, v19.4s, v15.4s      \n"

                    ///////

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v23.4s}, [%3]              \n"

                    "st1    {v24.4s}, [x4]              \n"
                    "add    x4, x4, #32                 \n"

                    "fmla   v20.4s, v16.4s, v0.4s       \n"
                    "fmla   v21.4s, v16.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v17.4s}, [%5], #16         \n"

                    "fmla   v22.4s, v16.4s, v8.4s       \n"
                    "fmla   v23.4s, v16.4s, v12.4s      \n"

                    "prfm   pldl1keep, [x4, #128]       \n"
                    "ld1    {v24.4s}, [x4]              \n"

                    "st1    {v25.4s}, [x5]              \n"
                    "add    x5, x5, #32                 \n"

                    "fmla   v20.4s, v17.4s, v1.4s       \n"
                    "fmla   v21.4s, v17.4s, v5.4s       \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v18.4s}, [%6], #16         \n"

                    "fmla   v22.4s, v17.4s, v9.4s       \n"
                    "fmla   v23.4s, v17.4s, v13.4s      \n"

                    "prfm   pldl1keep, [x5, #128]       \n"
                    "ld1    {v25.4s}, [x5]              \n"

                    "st1    {v26.4s}, [x6]              \n"
                    "add    x6, x6, #32                 \n"

                    "fmla   v20.4s, v18.4s, v2.4s       \n"
                    "fmla   v21.4s, v18.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v19.4s}, [%7], #16         \n"

                    "fmla   v22.4s, v18.4s, v10.4s      \n"
                    "fmla   v23.4s, v18.4s, v14.4s      \n"

                    "prfm   pldl1keep, [x6, #128]       \n"
                    "ld1    {v26.4s}, [x6]              \n"

                    "st1    {v27.4s}, [x7]              \n"
                    "add    x7, x7, #32                 \n"

                    "fmla   v20.4s, v19.4s, v3.4s       \n"
                    "fmla   v21.4s, v19.4s, v7.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "fmla   v22.4s, v19.4s, v11.4s      \n"
                    "fmla   v23.4s, v19.4s, v15.4s      \n"

                    ///////

                    "prfm   pldl1keep, [x7, #128]       \n"
                    "ld1    {v27.4s}, [x7]              \n"

                    "st1    {v20.4s}, [%0]              \n"

                    "fmla   v24.4s, v16.4s, v0.4s       \n"
                    "fmla   v25.4s, v16.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v17.4s}, [%5], #16         \n"

                    "fmla   v26.4s, v16.4s, v8.4s       \n"
                    "fmla   v27.4s, v16.4s, v12.4s      \n"

                    "st1    {v21.4s}, [%1]              \n"

                    "fmla   v24.4s, v17.4s, v1.4s       \n"
                    "fmla   v25.4s, v17.4s, v5.4s       \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v18.4s}, [%6], #16         \n"

                    "fmla   v26.4s, v17.4s, v9.4s       \n"
                    "fmla   v27.4s, v17.4s, v13.4s      \n"

                    "st1    {v22.4s}, [%2]              \n"

                    "fmla   v24.4s, v18.4s, v2.4s       \n"
                    "fmla   v25.4s, v18.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v19.4s}, [%7], #16         \n"

                    "fmla   v26.4s, v18.4s, v10.4s      \n"
                    "fmla   v27.4s, v18.4s, v14.4s      \n"

                    "st1    {v23.4s}, [%3]              \n"

                    "fmla   v24.4s, v19.4s, v3.4s       \n"
                    "fmla   v25.4s, v19.4s, v7.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "fmla   v26.4s, v19.4s, v11.4s      \n"
                    "fmla   v27.4s, v19.4s, v15.4s      \n"

                    "st1    {v24.4s}, [x4], #16         \n"
                    "mov    %0, x4                      \n"

                    "st1    {v25.4s}, [x5], #16         \n"
                    "mov    %1, x5                      \n"

                    "subs   w1, w1, #1                  \n"

                    "st1    {v26.4s}, [x6], #16         \n"
                    "mov    %2, x6                      \n"

                    "st1    {v27.4s}, [x7], #16         \n"
                    "mov    %3, x7                      \n"

                    "bne    1b                          \n"
                    "sub    %4, %4, #16                 \n"
                    //END tile loop

                    "2:                                 \n"

                    // remain loop
                    "and    w1, %w18, #3                \n"// w1 = remain = tiles & 3;
                    "cmp    w1, #0                      \n"
                    "beq    4f                          \n"

                    //BEGIN remain loop
                    "3:                                 \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v20.4s}, [%0]              \n"

                    "fmla   v20.4s, v16.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v21.4s}, [%1]              \n"

                    "fmla   v21.4s, v16.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v22.4s}, [%2]              \n"

                    "fmla   v22.4s, v16.4s, v8.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v23.4s}, [%3]              \n"

                    "fmla   v23.4s, v16.4s, v12.4s      \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v17.4s}, [%5], #16         \n"

                    "fmla   v20.4s, v17.4s, v1.4s       \n"
                    "fmla   v21.4s, v17.4s, v5.4s       \n"

                    "fmla   v22.4s, v17.4s, v9.4s       \n"
                    "fmla   v23.4s, v17.4s, v13.4s      \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v18.4s}, [%6], #16         \n"

                    "fmla   v20.4s, v18.4s, v2.4s       \n"
                    "fmla   v21.4s, v18.4s, v6.4s       \n"

                    "fmla   v22.4s, v18.4s, v10.4s      \n"
                    "fmla   v23.4s, v18.4s, v14.4s      \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v19.4s}, [%7], #16         \n"

                    "fmla   v20.4s, v19.4s, v3.4s       \n"
                    "fmla   v21.4s, v19.4s, v7.4s       \n"
                    "fmla   v22.4s, v19.4s, v11.4s      \n"
                    "fmla   v23.4s, v19.4s, v15.4s      \n"

                    "st1    {v20.4s}, [%0], #16         \n"
                    "st1    {v21.4s}, [%1], #16         \n"

                    "subs   w1, w1, #1                  \n"

                    "st1    {v22.4s}, [%2], #16         \n"
                    "st1    {v23.4s}, [%3], #16         \n"

                    "bne    3b                          \n"
                    //END remain loop

                    "4:                                 \n"

                    "subs   w0, w0, #1                  \n"
                    "bne    0b                          \n"

                    : "=r"(output0_tm), // %0
                      "=r"(output1_tm), // %1
                      "=r"(output2_tm), // %2
                      "=r"(output3_tm), // %3
                      "=r"(r0),         // %4
                      "=r"(r1),         // %5
                      "=r"(r2),         // %6
                      "=r"(r3),         // %7
                      "=r"(ktm)         // %8
                    : "0"(output0_tm),
                      "1"(output1_tm),
                      "2"(output2_tm),
                      "3"(output3_tm),
                      "4"(r0),
                      "5"(r1),
                      "6"(r2),
                      "7"(r3),
                      "8"(ktm),
                      "r"(tiles)        // %18
                    : "cc", "memory", "x0", "x1", "x4", "x5", "x6", "x7", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27"
                );
            }
#endif // __ARM_NEON && __aarch64__

            for (; q+1<inch; q+=2)
            {
                const float* r0 = bottom_blob_tm.channel(q);
                const float* r1 = bottom_blob_tm.channel(q+1);

                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;
                float* output2_tm = out2_tm;
                float* output3_tm = out3_tm;

#if __ARM_NEON
#if __aarch64__
                asm volatile(
                    "mov    w0, #16                     \n"// w0 = r = 16
                    "0:                                 \n"

                    "prfm   pldl1keep, [%6, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%6], #32   \n"// v0 v1 = _k00 _k01

                    "prfm   pldl1keep, [%6, #256]       \n"
                    "ld1    {v2.4s, v3.4s}, [%6], #32   \n"// v2 v3 = _k10 _k11

                    "prfm   pldl1keep, [%6, #256]       \n"
                    "ld1    {v4.4s, v5.4s}, [%6], #32   \n"// v4 v5 = _k20 _k21

                    "prfm   pldl1keep, [%6, #256]       \n"
                    "ld1    {v6.4s, v7.4s}, [%6], #32   \n"// v6 v7 = _k30 _k31

                    // tile loop
                    "lsr    w1, %w14, #2                \n"// w1 = nn = tiles >> 2
                    "cmp    w1, #0                      \n"
                    "beq    2f                          \n"

                    //BEGIN tile loop
                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v20.4s}, [%4], #16         \n"

                    "1:                                 \n"

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v16.4s}, [%0]              \n"

                    "fmla   v16.4s, v20.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v17.4s}, [%1]              \n"

                    "fmla   v17.4s, v20.4s, v2.4s       \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v18.4s}, [%2]              \n"

                    "fmla   v18.4s, v20.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v19.4s}, [%3]              \n"

                    "fmla   v19.4s, v20.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v21.4s}, [%5], #16         \n"

                    "fmla   v16.4s, v21.4s, v1.4s       \n"
                    "fmla   v17.4s, v21.4s, v3.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v20.4s}, [%4], #16         \n"

                    "fmla   v18.4s, v21.4s, v5.4s       \n"
                    "fmla   v19.4s, v21.4s, v7.4s       \n"

                    "st1    {v16.4s}, [%0], #16         \n"
                    "st1    {v17.4s}, [%1], #16         \n"

                    ////

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v16.4s}, [%0]              \n"

                    "fmla   v16.4s, v20.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v17.4s}, [%1]              \n"

                    "fmla   v17.4s, v20.4s, v2.4s       \n"

                    "st1    {v18.4s}, [%2], #16         \n"
                    "st1    {v19.4s}, [%3], #16         \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v18.4s}, [%2]              \n"

                    "fmla   v18.4s, v20.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v19.4s}, [%3]              \n"

                    "fmla   v19.4s, v20.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v21.4s}, [%5], #16         \n"

                    "fmla   v16.4s, v21.4s, v1.4s       \n"
                    "fmla   v17.4s, v21.4s, v3.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v20.4s}, [%4], #16         \n"

                    "fmla   v18.4s, v21.4s, v5.4s       \n"
                    "fmla   v19.4s, v21.4s, v7.4s       \n"

                    "st1    {v16.4s}, [%0], #16         \n"
                    "st1    {v17.4s}, [%1], #16         \n"

                    ////

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v16.4s}, [%0]              \n"

                    "fmla   v16.4s, v20.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v17.4s}, [%1]              \n"

                    "fmla   v17.4s, v20.4s, v2.4s       \n"

                    "st1    {v18.4s}, [%2], #16         \n"
                    "st1    {v19.4s}, [%3], #16         \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v18.4s}, [%2]              \n"

                    "fmla   v18.4s, v20.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v19.4s}, [%3]              \n"

                    "fmla   v19.4s, v20.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v21.4s}, [%5], #16         \n"

                    "fmla   v16.4s, v21.4s, v1.4s       \n"
                    "fmla   v17.4s, v21.4s, v3.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v20.4s}, [%4], #16         \n"

                    "fmla   v18.4s, v21.4s, v5.4s       \n"
                    "fmla   v19.4s, v21.4s, v7.4s       \n"

                    "st1    {v16.4s}, [%0], #16         \n"
                    "st1    {v17.4s}, [%1], #16         \n"

                    ////

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v16.4s}, [%0]              \n"

                    "fmla   v16.4s, v20.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v17.4s}, [%1]              \n"

                    "fmla   v17.4s, v20.4s, v2.4s       \n"

                    "st1    {v18.4s}, [%2], #16         \n"
                    "st1    {v19.4s}, [%3], #16         \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v18.4s}, [%2]              \n"

                    "fmla   v18.4s, v20.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v19.4s}, [%3]              \n"

                    "fmla   v19.4s, v20.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v21.4s}, [%5], #16         \n"

                    "fmla   v16.4s, v21.4s, v1.4s       \n"
                    "fmla   v17.4s, v21.4s, v3.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v20.4s}, [%4], #16         \n"

                    "fmla   v18.4s, v21.4s, v5.4s       \n"
                    "fmla   v19.4s, v21.4s, v7.4s       \n"

                    "st1    {v16.4s}, [%0], #16         \n"
                    "st1    {v17.4s}, [%1], #16         \n"

                    "subs   w1, w1, #1                  \n"

                    "st1    {v18.4s}, [%2], #16         \n"
                    "st1    {v19.4s}, [%3], #16         \n"

                    "bne    1b                          \n"
                    "sub    %4, %4, #16                 \n"
                    //END tile loop

                    "2:                                 \n"

                    // remain loop
                    "and    w1, %w14, #3                \n"// w1 = remain = tiles & 3;
                    "cmp    w1, #0                      \n"
                    "beq    4f                          \n"

                    //BEGIN remain loop
                    "3:                                 \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v20.4s}, [%4], #16         \n"

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v16.4s}, [%0]              \n"

                    "fmla   v16.4s, v20.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v17.4s}, [%1]              \n"

                    "fmla   v17.4s, v20.4s, v2.4s       \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v18.4s}, [%2]              \n"

                    "fmla   v18.4s, v20.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v19.4s}, [%3]              \n"

                    "fmla   v19.4s, v20.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v21.4s}, [%5], #16         \n"

                    "fmla   v16.4s, v21.4s, v1.4s       \n"
                    "fmla   v17.4s, v21.4s, v3.4s       \n"
                    "fmla   v18.4s, v21.4s, v5.4s       \n"
                    "fmla   v19.4s, v21.4s, v7.4s       \n"

                    "st1    {v16.4s}, [%0], #16         \n"
                    "st1    {v17.4s}, [%1], #16         \n"

                    "subs   w1, w1, #1                  \n"

                    "st1    {v18.4s}, [%2], #16         \n"
                    "st1    {v19.4s}, [%3], #16         \n"

                    "bne    3b                          \n"
                    //END remain loop

                    "4:                                 \n"

                    "subs   w0, w0, #1                  \n"
                    "bne    0b                          \n"

                    : "=r"(output0_tm), // %0
                      "=r"(output1_tm), // %1
                      "=r"(output2_tm), // %2
                      "=r"(output3_tm), // %3
                      "=r"(r0),         // %4
                      "=r"(r1),         // %5
                      "=r"(ktm)         // %6
                    : "0"(output0_tm),
                      "1"(output1_tm),
                      "2"(output2_tm),
                      "3"(output3_tm),
                      "4"(r0),
                      "5"(r1),
                      "6"(ktm),
                      "r"(tiles)        // %14
                    : "cc", "memory", "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21"
                );
#else
                asm volatile(
                    "mov        r0, #16                 \n"// r0 = r = 16
                    "0:                                 \n"

                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d0-d3}, [%6 :128]!     \n"// q0 q1 = _k00 _k01

                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d4-d7}, [%6 :128]!     \n"// q2 q3 = _k10 _k11

                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d8-d11}, [%6 :128]!    \n"// q4 q5 = _k20 _k21

                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d12-d15}, [%6 :128]!   \n"// q6 q7 = _k30 _k31

                    // tile loop
                    "lsr        r1, %14, #2             \n"// r1 = nn = tiles >> 2
                    "cmp        r1, #0                  \n"
                    "beq        2f                      \n"

                    //BEGIN tile loop
                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "1:                                 \n"

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                    "vmla.f32   q8, q12, q0             \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1 :128]    \n"// q9 = _output1_tm

                    "vmla.f32   q9, q12, q2             \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2 :128]    \n"// q10 = _output2_tm

                    "vmla.f32   q10, q12, q4            \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3 :128]    \n"// q11 = _output3_tm

                    "vmla.f32   q11, q12, q6            \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5 :128]!   \n"// q13 = _r1

                    "vmla.f32   q8, q13, q1             \n"
                    "vmla.f32   q9, q13, q3             \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "vmla.f32   q10, q13, q5            \n"
                    "vmla.f32   q11, q13, q7            \n"

                    "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%1 :128]!   \n"

                    ////

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                    "vmla.f32   q8, q12, q0             \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1 :128]    \n"// q9 = _output1_tm

                    "vmla.f32   q9, q12, q2             \n"

                    "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2 :128]    \n"// q10 = _output2_tm

                    "vmla.f32   q10, q12, q4            \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3 :128]    \n"// q11 = _output3_tm

                    "vmla.f32   q11, q12, q6            \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5 :128]!   \n"// q13 = _r1

                    "vmla.f32   q8, q13, q1             \n"
                    "vmla.f32   q9, q13, q3             \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "vmla.f32   q10, q13, q5            \n"
                    "vmla.f32   q11, q13, q7            \n"

                    "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%1 :128]!   \n"

                    ////

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                    "vmla.f32   q8, q12, q0             \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1 :128]    \n"// q9 = _output1_tm

                    "vmla.f32   q9, q12, q2             \n"

                    "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2 :128]    \n"// q10 = _output2_tm

                    "vmla.f32   q10, q12, q4            \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3 :128]    \n"// q11 = _output3_tm

                    "vmla.f32   q11, q12, q6            \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5 :128]!   \n"// q13 = _r1

                    "vmla.f32   q8, q13, q1             \n"
                    "vmla.f32   q9, q13, q3             \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "vmla.f32   q10, q13, q5            \n"
                    "vmla.f32   q11, q13, q7            \n"

                    "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%1 :128]!   \n"

                    ////

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                    "vmla.f32   q8, q12, q0             \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1 :128]    \n"// q9 = _output1_tm

                    "vmla.f32   q9, q12, q2             \n"

                    "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2 :128]    \n"// q10 = _output2_tm

                    "vmla.f32   q10, q12, q4            \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3 :128]    \n"// q11 = _output3_tm

                    "vmla.f32   q11, q12, q6            \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5 :128]!   \n"// q13 = _r1

                    "vmla.f32   q8, q13, q1             \n"
                    "vmla.f32   q9, q13, q3             \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "vmla.f32   q10, q13, q5            \n"
                    "vmla.f32   q11, q13, q7            \n"

                    "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%1 :128]!   \n"

                    "subs       r1, #1                  \n"

                    "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                    "bne        1b                      \n"
                    "sub        %4, %4, #16             \n"
                    //END tile loop

                    "2:                                 \n"

                    // remain loop
                    "and        r1, %14, #3             \n"// r1 = remain = tiles & 3;
                    "cmp        r1, #0                  \n"
                    "beq        4f                      \n"

                    //BEGIN remain loop
                    "3:                                 \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                    "vmla.f32   q8, q12, q0             \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1 :128]    \n"// q9 = _output1_tm

                    "vmla.f32   q9, q12, q2             \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2 :128]    \n"// q10 = _output2_tm

                    "vmla.f32   q10, q12, q4            \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3 :128]    \n"// q11 = _output3_tm

                    "vmla.f32   q11, q12, q6            \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5 :128]!   \n"// q13 = _r1

                    "vmla.f32   q8, q13, q1             \n"
                    "vmla.f32   q9, q13, q3             \n"
                    "vmla.f32   q10, q13, q5            \n"
                    "vmla.f32   q11, q13, q7            \n"

                    "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%1 :128]!   \n"

                    "subs       r1, #1                  \n"

                    "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                    "bne        3b                      \n"
                    //END remain loop

                    "4:                                 \n"

                    "subs       r0, #1                  \n"
                    "bne        0b                      \n"

                    : "=r"(output0_tm), // %0
                      "=r"(output1_tm), // %1
                      "=r"(output2_tm), // %2
                      "=r"(output3_tm), // %3
                      "=r"(r0),         // %4
                      "=r"(r1),         // %5
                      "=r"(ktm)         // %6
                    : "0"(output0_tm),
                      "1"(output1_tm),
                      "2"(output2_tm),
                      "3"(output3_tm),
                      "4"(r0),
                      "5"(r1),
                      "6"(ktm),
                      "r"(tiles)        // %14
                    : "cc", "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13"
                );
#endif // __aarch64__
#else
                for (int r=0; r<16; r++)
                {
                    for (int t=0; t<tiles; t++)
                    {
                        for (int m=0; m<4; m++)
                        {
                            output0_tm[m] += r0[m] * ktm[0 +m];
                            output0_tm[m] += r1[m] * ktm[4 +m];
                            output1_tm[m] += r0[m] * ktm[8 +m];
                            output1_tm[m] += r1[m] * ktm[12+m];
                            output2_tm[m] += r0[m] * ktm[16+m];
                            output2_tm[m] += r1[m] * ktm[20+m];
                            output3_tm[m] += r0[m] * ktm[24+m];
                            output3_tm[m] += r1[m] * ktm[28+m];
                        }

                        r0 += 4;
                        r1 += 4;
                        output0_tm += 4;
                        output1_tm += 4;
                        output2_tm += 4;
                        output3_tm += 4;
                    }

                    ktm += 32;
                }
#endif // __ARM_NEON
            }

            for (; q<inch; q++)
            {
                const float* r0 = bottom_blob_tm.channel(q);

                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;
                float* output2_tm = out2_tm;
                float* output3_tm = out3_tm;

#if __ARM_NEON
#if __aarch64__
                asm volatile(
                    "mov    w0, #16                     \n"// w0 = r = 16
                    "0:                                 \n"

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%5], #32   \n"// v0 v1 = _k00 _k10

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v2.4s, v3.4s}, [%5], #32   \n"// v2 v3 = _k20 _k30

                    // tile loop
                    "mov    w1, %w12                    \n"// w1 = tiles
                    "cmp    w1, #0                      \n"
                    "beq    2f                          \n"

                    //BEGIN tile loop
                    "1:                                 \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v17.4s}, [%0]              \n"

                    "fmla   v17.4s, v16.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v18.4s}, [%1]              \n"

                    "fmla   v18.4s, v16.4s, v1.4s       \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v19.4s}, [%2]              \n"

                    "fmla   v19.4s, v16.4s, v2.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v20.4s}, [%3]              \n"

                    "fmla   v20.4s, v16.4s, v3.4s       \n"

                    "st1    {v17.4s}, [%0], #16         \n"
                    "st1    {v18.4s}, [%1], #16         \n"

                    "subs   w1, w1, #1                  \n"

                    "st1    {v19.4s}, [%2], #16         \n"
                    "st1    {v20.4s}, [%3], #16         \n"

                    "bne    1b                          \n"
                    //END tile loop

                    "2:                                 \n"

                    "subs   w0, w0, #1                  \n"
                    "bne    0b                          \n"

                    : "=r"(output0_tm), // %0
                      "=r"(output1_tm), // %1
                      "=r"(output2_tm), // %2
                      "=r"(output3_tm), // %3
                      "=r"(r0),         // %4
                      "=r"(ktm)         // %5
                    : "0"(output0_tm),
                      "1"(output1_tm),
                      "2"(output2_tm),
                      "3"(output3_tm),
                      "4"(r0),
                      "5"(ktm),
                      "r"(tiles)        // %12
                    : "cc", "memory", "x0", "x1", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20"
                );
#else
                asm volatile(
                    "mov        r0, #16                 \n"// r0 = r = 16
                    "0:                                 \n"

                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d0-d3}, [%5 :128]!     \n"// q0 q1 = _k00 _k10

                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d4-d7}, [%5 :128]!     \n"// q2 q3 = _k20 _k30

                    // tile loop
                    "mov        r1, %12                 \n"// r1 = tiles
                    "cmp        r1, #0                  \n"
                    "beq        2f                      \n"

                    //BEGIN tile loop
                    "1:                                 \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                    "vmla.f32   q8, q12, q0             \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1 :128]    \n"// q9 = _output1_tm

                    "vmla.f32   q9, q12, q1             \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2 :128]    \n"// q10 = _output2_tm

                    "vmla.f32   q10, q12, q2            \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3 :128]    \n"// q11 = _output3_tm

                    "vmla.f32   q11, q12, q3            \n"

                    "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%1 :128]!   \n"

                    "subs       r1, #1                  \n"

                    "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                    "bne        1b                      \n"
                    //END tile loop

                    "2:                                 \n"

                    "subs       r0, #1                  \n"
                    "bne        0b                      \n"

                    : "=r"(output0_tm), // %0
                      "=r"(output1_tm), // %1
                      "=r"(output2_tm), // %2
                      "=r"(output3_tm), // %3
                      "=r"(r0),         // %4
                      "=r"(ktm)         // %5
                    : "0"(output0_tm),
                      "1"(output1_tm),
                      "2"(output2_tm),
                      "3"(output3_tm),
                      "4"(r0),
                      "5"(ktm),
                      "r"(tiles)        // %12
                    : "cc", "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13"
                );
#endif // __aarch64__
#else
                for (int r=0; r<16; r++)
                {
                    for (int t=0; t<tiles; t++)
                    {
                        for (int m=0; m<4; m++)
                        {
                            output0_tm[m] += r0[m] * ktm[0 +m];
                            output1_tm[m] += r0[m] * ktm[4 +m];
                            output2_tm[m] += r0[m] * ktm[8 +m];
                            output3_tm[m] += r0[m] * ktm[12+m];
                        }

                        r0 += 4;
                        output0_tm += 4;
                        output1_tm += 4;
                        output2_tm += 4;
                        output3_tm += 4;
                    }

                    ktm += 16;
                }
#endif // __ARM_NEON
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p<outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);

            const float* ktm = (const float*)kernel_tm.channel(nn_outch) + 8*8 * inch * (p-remain_outch_start);

            out0_tm.fill(0.f);

            int q = 0;

            for (; q<inch; q++)
            {
                const float* r0 = bottom_blob_tm.channel(q);

                float* output0_tm = out0_tm;

                for (int r=0; r<16; r++)
                {
#if __ARM_NEON
                float32x4_t _k00 = vld1q_f32(ktm); ktm += 4;
#endif // __ARM_NEON

                // tile
                for (int i=0; i<tiles; i++)
                {
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #128]   \n"
                        "ld1    {v17.4s}, [%1], #16     \n"

                        "prfm   pldl1keep, [%0, #128]   \n"
                        "ld1    {v16.4s}, [%0]          \n"

                        "fmla   v16.4s, v17.4s, %4.4s   \n"

                        "st1    {v16.4s}, [%0], #16     \n"
                        : "=r"(output0_tm), // %0
                          "=r"(r0)          // %1
                        : "0"(output0_tm),
                          "1"(r0),
                          "w"(_k00)         // %4
                        : "cc", "memory", "v16", "v17"
                    );
#else
                    asm volatile(
                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d18-d19}, [%1 :128]!   \n"// q9 = _r0

                        "pld        [%0, #128]              \n"
                        "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                        "vmla.f32   q8, q9, %q4             \n"

                        "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                        : "=r"(output0_tm), // %0
                          "=r"(r0)          // %1
                        : "0"(output0_tm),
                          "1"(r0),
                          "w"(_k00)         // %4
                        : "cc", "memory", "q8", "q9"
                    );
#endif // __aarch64__
#else
                    for (int m=0; m<4; m++)
                    {
                        output0_tm[m] += r0[m] * ktm[m];
                    }

                    r0 += 4;
                    output0_tm += 4;
#endif // __ARM_NEON
                }

#if !__ARM_NEON
                ktm += 4;
#endif // __ARM_NEON
                }
            }
        }
    }
}
}
