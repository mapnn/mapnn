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
static void conv3x3s1_winograd64_neon5_dot(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Option& opt,
        int inch, int outw, int outh, int outch)
{
    {
        Mat bottom_blob_tm2 = bottom_blob;
        Mat top_blob_tm = top_blob;
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm/8 * h_tm/8;

        int nn_outch = 0;
        int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
        nn_outch = outch >> 3;
        remain_outch_start = nn_outch << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 8;

            const Mat kernel_tm0 = kernel_tm.channel(p/8);

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p+1);
            Mat out2_tm = top_blob_tm.channel(p+2);
            Mat out3_tm = top_blob_tm.channel(p+3);
            Mat out4_tm = top_blob_tm.channel(p+4);
            Mat out5_tm = top_blob_tm.channel(p+5);
            Mat out6_tm = top_blob_tm.channel(p+6);
            Mat out7_tm = top_blob_tm.channel(p+7);

            float* output0_tm = out0_tm;
            float* output1_tm = out1_tm;
            float* output2_tm = out2_tm;
            float* output3_tm = out3_tm;
            float* output4_tm = out4_tm;
            float* output5_tm = out5_tm;
            float* output6_tm = out6_tm;
            float* output7_tm = out7_tm;

            for (int r=0; r<64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                // tile
                int i=0;
                for (; i+7<tiles; i+=8)
                {
                    const float* bb2p0 = bb2.row(i/8);

                    const float* ktm0 = kernel_tm0.row(r);

                    asm volatile(
                        "eor    v16.16b, v16.16b, v16.16b  \n"
                        "eor    v17.16b, v17.16b, v17.16b  \n"
                        "eor    v18.16b, v18.16b, v18.16b  \n"
                        "eor    v19.16b, v19.16b, v19.16b  \n"
                        "eor    v20.16b, v20.16b, v20.16b  \n"
                        "eor    v21.16b, v21.16b, v21.16b  \n"
                        "eor    v22.16b, v22.16b, v22.16b  \n"
                        "eor    v23.16b, v23.16b, v23.16b  \n"
                        "eor    v24.16b, v24.16b, v24.16b  \n"
                        "eor    v25.16b, v25.16b, v25.16b  \n"
                        "eor    v26.16b, v26.16b, v26.16b  \n"
                        "eor    v27.16b, v27.16b, v27.16b  \n"
                        "eor    v28.16b, v28.16b, v28.16b  \n"
                        "eor    v29.16b, v29.16b, v29.16b  \n"
                        "eor    v30.16b, v30.16b, v30.16b  \n"
                        "eor    v31.16b, v31.16b, v31.16b  \n"

                        // inch loop
                        "lsr    w4, %w20, #2            \n"// w4 = nn = inch >> 2
                        "cmp    w4, #0                  \n"
                        "beq    1f                      \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%8, #512]   \n"
                        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64   \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64   \n"

                        "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                        "fmla   v17.4s, v9.4s, v0.s[0]  \n"
                        "fmla   v18.4s, v8.4s, v0.s[1]  \n"
                        "fmla   v19.4s, v9.4s, v0.s[1]  \n"
                        "fmla   v20.4s, v8.4s, v0.s[2]  \n"
                        "fmla   v21.4s, v9.4s, v0.s[2]  \n"
                        "fmla   v22.4s, v8.4s, v0.s[3]  \n"
                        "fmla   v23.4s, v9.4s, v0.s[3]  \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64   \n"

                        "fmla   v24.4s, v8.4s, v1.s[0]  \n"
                        "fmla   v25.4s, v9.4s, v1.s[0]  \n"
                        "fmla   v26.4s, v8.4s, v1.s[1]  \n"
                        "fmla   v27.4s, v9.4s, v1.s[1]  \n"
                        "fmla   v28.4s, v8.4s, v1.s[2]  \n"
                        "fmla   v29.4s, v9.4s, v1.s[2]  \n"
                        "fmla   v30.4s, v8.4s, v1.s[3]  \n"
                        "fmla   v31.4s, v9.4s, v1.s[3]  \n"

                        "fmla   v16.4s, v10.4s, v2.s[0] \n"
                        "fmla   v17.4s, v11.4s, v2.s[0] \n"
                        "fmla   v18.4s, v10.4s, v2.s[1] \n"
                        "fmla   v19.4s, v11.4s, v2.s[1] \n"
                        "fmla   v20.4s, v10.4s, v2.s[2] \n"
                        "fmla   v21.4s, v11.4s, v2.s[2] \n"
                        "fmla   v22.4s, v10.4s, v2.s[3] \n"
                        "fmla   v23.4s, v11.4s, v2.s[3] \n"

                        "prfm   pldl1keep, [%8, #512]   \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%8], #64 \n"

                        "fmla   v24.4s, v10.4s, v3.s[0] \n"
                        "fmla   v25.4s, v11.4s, v3.s[0] \n"
                        "fmla   v26.4s, v10.4s, v3.s[1] \n"
                        "fmla   v27.4s, v11.4s, v3.s[1] \n"
                        "fmla   v28.4s, v10.4s, v3.s[2] \n"
                        "fmla   v29.4s, v11.4s, v3.s[2] \n"
                        "fmla   v30.4s, v10.4s, v3.s[3] \n"
                        "fmla   v31.4s, v11.4s, v3.s[3] \n"

                        "fmla   v16.4s, v12.4s, v4.s[0] \n"
                        "fmla   v17.4s, v13.4s, v4.s[0] \n"
                        "fmla   v18.4s, v12.4s, v4.s[1] \n"
                        "fmla   v19.4s, v13.4s, v4.s[1] \n"
                        "fmla   v20.4s, v12.4s, v4.s[2] \n"
                        "fmla   v21.4s, v13.4s, v4.s[2] \n"
                        "fmla   v22.4s, v12.4s, v4.s[3] \n"
                        "fmla   v23.4s, v13.4s, v4.s[3] \n"

                        "fmla   v24.4s, v12.4s, v5.s[0] \n"
                        "fmla   v25.4s, v13.4s, v5.s[0] \n"
                        "fmla   v26.4s, v12.4s, v5.s[1] \n"
                        "fmla   v27.4s, v13.4s, v5.s[1] \n"
                        "fmla   v28.4s, v12.4s, v5.s[2] \n"
                        "fmla   v29.4s, v13.4s, v5.s[2] \n"
                        "fmla   v30.4s, v12.4s, v5.s[3] \n"
                        "fmla   v31.4s, v13.4s, v5.s[3] \n"

                        "fmla   v16.4s, v14.4s, v6.s[0] \n"
                        "fmla   v17.4s, v15.4s, v6.s[0] \n"
                        "fmla   v18.4s, v14.4s, v6.s[1] \n"
                        "fmla   v19.4s, v15.4s, v6.s[1] \n"
                        "fmla   v20.4s, v14.4s, v6.s[2] \n"
                        "fmla   v21.4s, v15.4s, v6.s[2] \n"
                        "fmla   v22.4s, v14.4s, v6.s[3] \n"
                        "fmla   v23.4s, v15.4s, v6.s[3] \n"

                        "subs   w4, w4, #1              \n"

                        "fmla   v24.4s, v14.4s, v7.s[0] \n"
                        "fmla   v25.4s, v15.4s, v7.s[0] \n"
                        "fmla   v26.4s, v14.4s, v7.s[1] \n"
                        "fmla   v27.4s, v15.4s, v7.s[1] \n"
                        "fmla   v28.4s, v14.4s, v7.s[2] \n"
                        "fmla   v29.4s, v15.4s, v7.s[2] \n"
                        "fmla   v30.4s, v14.4s, v7.s[3] \n"
                        "fmla   v31.4s, v15.4s, v7.s[3] \n"

                        "bne    0b                      \n"

                        "1:                             \n"

                        // remain loop
                        "and    w4, %w20, #3            \n"// w4 = remain = tiles & 3;
                        "cmp    w4, #0                  \n"
                        "beq    3f                      \n"

                        "2:                             \n"

                        "prfm   pldl1keep, [%8, #256]   \n"
                        "ld1    {v8.4s, v9.4s}, [%8], #32   \n"

                        "prfm   pldl1keep, [%9, #256]   \n"
                        "ld1    {v0.4s, v1.4s}, [%9], #32   \n"

                        "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                        "fmla   v17.4s, v9.4s, v0.s[0]  \n"
                        "fmla   v18.4s, v8.4s, v0.s[1]  \n"
                        "fmla   v19.4s, v9.4s, v0.s[1]  \n"
                        "fmla   v20.4s, v8.4s, v0.s[2]  \n"
                        "fmla   v21.4s, v9.4s, v0.s[2]  \n"
                        "fmla   v22.4s, v8.4s, v0.s[3]  \n"
                        "fmla   v23.4s, v9.4s, v0.s[3]  \n"

                        "subs   w4, w4, #1              \n"

                        "fmla   v24.4s, v8.4s, v1.s[0]  \n"
                        "fmla   v25.4s, v9.4s, v1.s[0]  \n"
                        "fmla   v26.4s, v8.4s, v1.s[1]  \n"
                        "fmla   v27.4s, v9.4s, v1.s[1]  \n"
                        "fmla   v28.4s, v8.4s, v1.s[2]  \n"
                        "fmla   v29.4s, v9.4s, v1.s[2]  \n"
                        "fmla   v30.4s, v8.4s, v1.s[3]  \n"
                        "fmla   v31.4s, v9.4s, v1.s[3]  \n"

                        "bne    2b                      \n"

                        "3:                             \n"

                        "st1    {v16.4s, v17.4s}, [%0], #32 \n"
                        "st1    {v18.4s, v19.4s}, [%1], #32 \n"
                        "st1    {v20.4s, v21.4s}, [%2], #32 \n"
                        "st1    {v22.4s, v23.4s}, [%3], #32 \n"
                        "st1    {v24.4s, v25.4s}, [%4], #32 \n"
                        "st1    {v26.4s, v27.4s}, [%5], #32 \n"
                        "st1    {v28.4s, v29.4s}, [%6], #32 \n"
                        "st1    {v30.4s, v31.4s}, [%7], #32 \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(output4_tm), // %4
                          "=r"(output5_tm), // %5
                          "=r"(output6_tm), // %6
                          "=r"(output7_tm), // %7
                          "=r"(bb2p0),      // %8
                          "=r"(ktm0)        // %9
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(output4_tm),
                          "5"(output5_tm),
                          "6"(output6_tm),
                          "7"(output7_tm),
                          "8"(bb2p0),
                          "9"(ktm0),
                          "r"(inch)         // %20
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                    );
                }
                for (; i+3<tiles; i+=4)
                {
                    const float* bb2p0 = bb2.row(i/8+(i%8)/4);

                    const float* ktm0 = kernel_tm0.row(r);

                    asm volatile(
                        "eor    v16.16b, v16.16b, v16.16b  \n"
                        "eor    v17.16b, v17.16b, v17.16b  \n"
                        "eor    v18.16b, v18.16b, v18.16b  \n"
                        "eor    v19.16b, v19.16b, v19.16b  \n"
                        "eor    v20.16b, v20.16b, v20.16b  \n"
                        "eor    v21.16b, v21.16b, v21.16b  \n"
                        "eor    v22.16b, v22.16b, v22.16b  \n"
                        "eor    v23.16b, v23.16b, v23.16b  \n"

                        // inch loop
                        "lsr    w4, %w20, #2            \n"// w4 = nn = inch >> 2
                        "cmp    w4, #0                  \n"
                        "beq    1f                      \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%8, #512]   \n"
                        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64 \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64   \n"

                        "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                        "fmla   v17.4s, v8.4s, v0.s[1]  \n"
                        "fmla   v18.4s, v8.4s, v0.s[2]  \n"
                        "fmla   v19.4s, v8.4s, v0.s[3]  \n"
                        "fmla   v20.4s, v8.4s, v1.s[0]  \n"
                        "fmla   v21.4s, v8.4s, v1.s[1]  \n"
                        "fmla   v22.4s, v8.4s, v1.s[2]  \n"
                        "fmla   v23.4s, v8.4s, v1.s[3]  \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64   \n"

                        "fmla   v16.4s, v9.4s, v2.s[0]  \n"
                        "fmla   v17.4s, v9.4s, v2.s[1]  \n"
                        "fmla   v18.4s, v9.4s, v2.s[2]  \n"
                        "fmla   v19.4s, v9.4s, v2.s[3]  \n"
                        "fmla   v20.4s, v9.4s, v3.s[0]  \n"
                        "fmla   v21.4s, v9.4s, v3.s[1]  \n"
                        "fmla   v22.4s, v9.4s, v3.s[2]  \n"
                        "fmla   v23.4s, v9.4s, v3.s[3]  \n"

                        "fmla   v16.4s, v10.4s, v4.s[0] \n"
                        "fmla   v17.4s, v10.4s, v4.s[1] \n"
                        "fmla   v18.4s, v10.4s, v4.s[2] \n"
                        "fmla   v19.4s, v10.4s, v4.s[3] \n"
                        "fmla   v20.4s, v10.4s, v5.s[0] \n"
                        "fmla   v21.4s, v10.4s, v5.s[1] \n"
                        "fmla   v22.4s, v10.4s, v5.s[2] \n"
                        "fmla   v23.4s, v10.4s, v5.s[3] \n"

                        "subs   w4, w4, #1              \n"

                        "fmla   v16.4s, v11.4s, v6.s[0] \n"
                        "fmla   v17.4s, v11.4s, v6.s[1] \n"
                        "fmla   v18.4s, v11.4s, v6.s[2] \n"
                        "fmla   v19.4s, v11.4s, v6.s[3] \n"
                        "fmla   v20.4s, v11.4s, v7.s[0] \n"
                        "fmla   v21.4s, v11.4s, v7.s[1] \n"
                        "fmla   v22.4s, v11.4s, v7.s[2] \n"
                        "fmla   v23.4s, v11.4s, v7.s[3] \n"

                        "bne    0b                      \n"

                        "1:                             \n"

                        // remain loop
                        "and    w4, %w20, #3            \n"// w4 = remain = tiles & 3;
                        "cmp    w4, #0                  \n"
                        "beq    3f                      \n"

                        "2:                             \n"

                        "prfm   pldl1keep, [%8, #128]   \n"
                        "ld1    {v8.4s}, [%8], #16      \n"

                        "prfm   pldl1keep, [%9, #256]   \n"
                        "ld1    {v0.4s, v1.4s}, [%9], #32   \n"

                        "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                        "fmla   v17.4s, v8.4s, v0.s[1]  \n"
                        "fmla   v18.4s, v8.4s, v0.s[2]  \n"
                        "fmla   v19.4s, v8.4s, v0.s[3]  \n"

                        "subs   w4, w4, #1              \n"

                        "fmla   v20.4s, v8.4s, v1.s[0]  \n"
                        "fmla   v21.4s, v8.4s, v1.s[1]  \n"
                        "fmla   v22.4s, v8.4s, v1.s[2]  \n"
                        "fmla   v23.4s, v8.4s, v1.s[3]  \n"

                        "bne    2b                      \n"

                        "3:                             \n"

                        "st1    {v16.4s}, [%0], #16     \n"
                        "st1    {v17.4s}, [%1], #16     \n"
                        "st1    {v18.4s}, [%2], #16     \n"
                        "st1    {v19.4s}, [%3], #16     \n"
                        "st1    {v20.4s}, [%4], #16     \n"
                        "st1    {v21.4s}, [%5], #16     \n"
                        "st1    {v22.4s}, [%6], #16     \n"
                        "st1    {v23.4s}, [%7], #16     \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(output4_tm), // %4
                          "=r"(output5_tm), // %5
                          "=r"(output6_tm), // %6
                          "=r"(output7_tm), // %7
                          "=r"(bb2p0),      // %8
                          "=r"(ktm0)        // %9
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(output4_tm),
                          "5"(output5_tm),
                          "6"(output6_tm),
                          "7"(output7_tm),
                          "8"(bb2p0),
                          "9"(ktm0),
                          "r"(inch)         // %20
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                    );
                }
                for (; i<tiles; i++)
                {
                    const float* bb2p0 = bb2.row(i/8+(i%8)/4+i%4);

                    const float* ktm0 = kernel_tm0.row(r);

                    float32x4_t _sum0123 = vdupq_n_f32(0.f);
                    float32x4_t _sum4567 = vdupq_n_f32(0.f);

                    int q=0;
                    for (; q+3<inch; q+=4)
                    {
//                         asm volatile("prfm pldl1keep, [%0, #128] \n" : :"r"(bb2p0) :);
                        float32x4_t _bb2p0 = vld1q_f32(bb2p0);
                        bb2p0 += 4;

//                         asm volatile("prfm pldl1keep, [%0, #512] \n" : :"r"(ktm0) :);
                        float32x4_t _ktm0 = vld1q_f32(ktm0 + 0);
                        float32x4_t _ktm1 = vld1q_f32(ktm0 + 4);
                        float32x4_t _ktm2 = vld1q_f32(ktm0 + 8);
                        float32x4_t _ktm3 = vld1q_f32(ktm0 + 12);
                        ktm0 += 16;

                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm0, _bb2p0, 0);
                        _sum4567 = vmlaq_laneq_f32(_sum4567, _ktm1, _bb2p0, 0);
                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm2, _bb2p0, 1);
                        _sum4567 = vmlaq_laneq_f32(_sum4567, _ktm3, _bb2p0, 1);

//                         asm volatile("prfm pldl1keep, [%0, #512] \n" : :"r"(ktm0) :);
                        float32x4_t _ktm4 = vld1q_f32(ktm0 + 0);
                        float32x4_t _ktm5 = vld1q_f32(ktm0 + 4);
                        float32x4_t _ktm6 = vld1q_f32(ktm0 + 8);
                        float32x4_t _ktm7 = vld1q_f32(ktm0 + 12);
                        ktm0 += 16;

                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm4, _bb2p0, 2);
                        _sum4567 = vmlaq_laneq_f32(_sum4567, _ktm5, _bb2p0, 2);
                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm6, _bb2p0, 3);
                        _sum4567 = vmlaq_laneq_f32(_sum4567, _ktm7, _bb2p0, 3);
                    }

                    for (; q<inch; q++)
                    {
                        float32x4_t _bb2p0 = vld1q_dup_f32(bb2p0);
                        float32x4_t _ktm0123 = vld1q_f32(ktm0 + 0);
                        float32x4_t _ktm4567 = vld1q_f32(ktm0 + 4);

                        _sum0123 = vmlaq_f32(_sum0123, _bb2p0, _ktm0123);
                        _sum4567 = vmlaq_f32(_sum4567, _bb2p0, _ktm4567);

                        bb2p0 += 1;
                        ktm0 += 8;
                    }

                    float sum0 = vgetq_lane_f32(_sum0123, 0);
                    float sum1 = vgetq_lane_f32(_sum0123, 1);
                    float sum2 = vgetq_lane_f32(_sum0123, 2);
                    float sum3 = vgetq_lane_f32(_sum0123, 3);
                    float sum4 = vgetq_lane_f32(_sum4567, 0);
                    float sum5 = vgetq_lane_f32(_sum4567, 1);
                    float sum6 = vgetq_lane_f32(_sum4567, 2);
                    float sum7 = vgetq_lane_f32(_sum4567, 3);

                    output0_tm[0] = sum0;
                    output1_tm[0] = sum1;
                    output2_tm[0] = sum2;
                    output3_tm[0] = sum3;
                    output4_tm[0] = sum4;
                    output5_tm[0] = sum5;
                    output6_tm[0] = sum6;
                    output7_tm[0] = sum7;

                    output0_tm += 1;
                    output1_tm += 1;
                    output2_tm += 1;
                    output3_tm += 1;
                    output4_tm += 1;
                    output5_tm += 1;
                    output6_tm += 1;
                    output7_tm += 1;
                }
            }
        }
#endif // __aarch64__

        nn_outch = (outch - remain_outch_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = remain_outch_start + pp * 4;

#if __ARM_NEON && __aarch64__
            const Mat kernel_tm0 = kernel_tm.channel(p/8+(p%8)/4);
#else
            const Mat kernel_tm0 = kernel_tm.channel(p/4);
#endif

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p+1);
            Mat out2_tm = top_blob_tm.channel(p+2);
            Mat out3_tm = top_blob_tm.channel(p+3);

            float* output0_tm = out0_tm;
            float* output1_tm = out1_tm;
            float* output2_tm = out2_tm;
            float* output3_tm = out3_tm;

            for (int r=0; r<64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                // tile
                int i=0;
                for (; i+7<tiles; i+=8)
                {
                    const float* bb2p0 = bb2.row(i/8);

                    const float* ktm0 = kernel_tm0.row(r);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b     \n"
                        "eor    v9.16b, v9.16b, v9.16b     \n"
                        "eor    v10.16b, v10.16b, v10.16b  \n"
                        "eor    v11.16b, v11.16b, v11.16b  \n"
                        "eor    v12.16b, v12.16b, v12.16b  \n"
                        "eor    v13.16b, v13.16b, v13.16b  \n"
                        "eor    v14.16b, v14.16b, v14.16b  \n"
                        "eor    v15.16b, v15.16b, v15.16b  \n"

                        // inch loop
                        "lsr    w4, %w12, #2            \n"// w4 = nn = inch >> 2
                        "cmp    w4, #0                  \n"
                        "beq    1f                      \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%4, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

                        "prfm   pldl1keep, [%5, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v9.4s, v5.4s, v0.s[0]   \n"
                        "fmla   v10.4s, v4.4s, v0.s[1]  \n"
                        "fmla   v11.4s, v5.4s, v0.s[1]  \n"
                        "fmla   v12.4s, v4.4s, v0.s[2]  \n"
                        "fmla   v13.4s, v5.4s, v0.s[2]  \n"
                        "fmla   v14.4s, v4.4s, v0.s[3]  \n"
                        "fmla   v15.4s, v5.4s, v0.s[3]  \n"

                        "prfm   pldl1keep, [%4, #512]   \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmla   v8.4s, v6.4s, v1.s[0]   \n"
                        "fmla   v9.4s, v7.4s, v1.s[0]   \n"
                        "fmla   v10.4s, v6.4s, v1.s[1]  \n"
                        "fmla   v11.4s, v7.4s, v1.s[1]  \n"
                        "fmla   v12.4s, v6.4s, v1.s[2]  \n"
                        "fmla   v13.4s, v7.4s, v1.s[2]  \n"
                        "fmla   v14.4s, v6.4s, v1.s[3]  \n"
                        "fmla   v15.4s, v7.4s, v1.s[3]  \n"

                        "fmla   v8.4s, v16.4s, v2.s[0]  \n"
                        "fmla   v9.4s, v17.4s, v2.s[0]  \n"
                        "fmla   v10.4s, v16.4s, v2.s[1] \n"
                        "fmla   v11.4s, v17.4s, v2.s[1] \n"
                        "fmla   v12.4s, v16.4s, v2.s[2] \n"
                        "fmla   v13.4s, v17.4s, v2.s[2] \n"
                        "fmla   v14.4s, v16.4s, v2.s[3] \n"
                        "fmla   v15.4s, v17.4s, v2.s[3] \n"

                        "fmla   v8.4s, v18.4s, v3.s[0]  \n"
                        "fmla   v9.4s, v19.4s, v3.s[0]  \n"
                        "fmla   v10.4s, v18.4s, v3.s[1] \n"
                        "fmla   v11.4s, v19.4s, v3.s[1] \n"
                        "fmla   v12.4s, v18.4s, v3.s[2] \n"
                        "fmla   v13.4s, v19.4s, v3.s[2] \n"
                        "fmla   v14.4s, v18.4s, v3.s[3] \n"
                        "fmla   v15.4s, v19.4s, v3.s[3] \n"

                        "subs   w4, w4, #1              \n"
                        "bne    0b                      \n"

                        "1:                             \n"

                        // remain loop
                        "and    w4, %w12, #3            \n"// w4 = remain = tiles & 3;
                        "cmp    w4, #0                  \n"
                        "beq    3f                      \n"

                        "2:                             \n"

                        "prfm   pldl1keep, [%4, #256]   \n"
                        "ld1    {v4.4s, v5.4s}, [%4], #32      \n"

                        "prfm   pldl1keep, [%5, #128]   \n"
                        "ld1    {v0.4s}, [%5], #16      \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v9.4s, v5.4s, v0.s[0]   \n"
                        "fmla   v10.4s, v4.4s, v0.s[1]  \n"
                        "fmla   v11.4s, v5.4s, v0.s[1]  \n"
                        "fmla   v12.4s, v4.4s, v0.s[2]  \n"
                        "fmla   v13.4s, v5.4s, v0.s[2]  \n"
                        "fmla   v14.4s, v4.4s, v0.s[3]  \n"
                        "fmla   v15.4s, v5.4s, v0.s[3]  \n"

                        "subs   w4, w4, #1              \n"
                        "bne    2b                      \n"

                        "3:                             \n"

                        "st1    {v8.4s, v9.4s}, [%0], #32       \n"
                        "st1    {v10.4s, v11.4s}, [%1], #32     \n"
                        "st1    {v12.4s, v13.4s}, [%2], #32     \n"
                        "st1    {v14.4s, v15.4s}, [%3], #32     \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(bb2p0),      // %4
                          "=r"(ktm0)        // %5
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(bb2p0),
                          "5"(ktm0),
                          "r"(inch)         // %12
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
                    );
#else // __aarch64__
                    asm volatile(
                        "veor       q8, q8, q8      \n"
                        "veor       q9, q9, q9      \n"
                        "veor       q10, q10, q10   \n"
                        "veor       q11, q11, q11   \n"
                        "veor       q12, q12, q12   \n"
                        "veor       q13, q13, q13   \n"
                        "veor       q14, q14, q14   \n"
                        "veor       q15, q15, q15   \n"

                        // inch loop
                        "lsr        r4, %12, #2     \n"// r4 = nn = inch >> 2
                        "cmp        r4, #0          \n"
                        "beq        1f              \n"

                        "0:                         \n"

                        "pld        [%4, #512]      \n"
                        "vldm       %4!, {d8-d15}   \n"
//                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
//                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

                        "pld        [%5, #512]      \n"
                        "vldm       %5!, {d0-d7}    \n"
//                         "vld1.f32   {d0-d3}, [%5 :128]!  \n"
//                         "vld1.f32   {d4-d7}, [%5 :128]!  \n"

                        "vmla.f32   q8, q4, d0[0]   \n"
                        "vmla.f32   q9, q5, d0[0]   \n"
                        "vmla.f32   q10, q4, d0[1]  \n"
                        "vmla.f32   q11, q5, d0[1]  \n"
                        "vmla.f32   q12, q4, d1[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q4, d1[1]  \n"
                        "vmla.f32   q15, q5, d1[1]  \n"

                        "vmla.f32   q8, q6, d2[0]   \n"
                        "vmla.f32   q9, q7, d2[0]   \n"
                        "vmla.f32   q10, q6, d2[1]  \n"
                        "vmla.f32   q11, q7, d2[1]  \n"
                        "vmla.f32   q12, q6, d3[0]  \n"
                        "vmla.f32   q13, q7, d3[0]  \n"
                        "vmla.f32   q14, q6, d3[1]  \n"
                        "vmla.f32   q15, q7, d3[1]  \n"

                        "pld        [%4, #512]      \n"
                        "vldm       %4!, {d8-d15}   \n"
//                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
//                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

                        "vmla.f32   q8, q4, d4[0]   \n"
                        "vmla.f32   q9, q5, d4[0]   \n"
                        "vmla.f32   q10, q4, d4[1]  \n"
                        "vmla.f32   q11, q5, d4[1]  \n"
                        "vmla.f32   q12, q4, d5[0]  \n"
                        "vmla.f32   q13, q5, d5[0]  \n"
                        "vmla.f32   q14, q4, d5[1]  \n"
                        "vmla.f32   q15, q5, d5[1]  \n"

                        "subs       r4, r4, #1      \n"

                        "vmla.f32   q8, q6, d6[0]   \n"
                        "vmla.f32   q9, q7, d6[0]   \n"
                        "vmla.f32   q10, q6, d6[1]  \n"
                        "vmla.f32   q11, q7, d6[1]  \n"
                        "vmla.f32   q12, q6, d7[0]  \n"
                        "vmla.f32   q13, q7, d7[0]  \n"
                        "vmla.f32   q14, q6, d7[1]  \n"
                        "vmla.f32   q15, q7, d7[1]  \n"

                        "bne        0b              \n"

                        "1:                         \n"

                        // remain loop
                        "and        r4, %12, #3     \n"// r4 = remain = tiles & 3;
                        "cmp        r4, #0          \n"
                        "beq        3f              \n"

                        "2:                         \n"

                        "pld        [%4, #256]      \n"
                        "vld1.f32   {d8-d11}, [%4 :128]! \n"

                        "pld        [%5, #128]      \n"
                        "vld1.f32   {d0-d1}, [%5 :128]!  \n"

                        "vmla.f32   q8, q4, d0[0]   \n"
                        "vmla.f32   q9, q5, d0[0]   \n"
                        "vmla.f32   q10, q4, d0[1]  \n"
                        "vmla.f32   q11, q5, d0[1]  \n"

                        "subs       r4, r4, #1      \n"

                        "vmla.f32   q12, q4, d1[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q4, d1[1]  \n"
                        "vmla.f32   q15, q5, d1[1]  \n"

                        "bne        2b              \n"

                        "3:                         \n"

                        "vst1.f32   {d16-d19}, [%0]! \n"
                        "vst1.f32   {d20-d23}, [%1]! \n"
                        "vst1.f32   {d24-d27}, [%2]! \n"
                        "vst1.f32   {d28-d31}, [%3]! \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(bb2p0),      // %4
                          "=r"(ktm0)        // %5
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(bb2p0),
                          "5"(ktm0),
                          "r"(inch)         // %12
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    float sum0_0 = 0.f;
                    float sum0_1 = 0.f;
                    float sum0_2 = 0.f;
                    float sum0_3 = 0.f;
                    float sum0_4 = 0.f;
                    float sum0_5 = 0.f;
                    float sum0_6 = 0.f;
                    float sum0_7 = 0.f;

                    float sum1_0 = 0.f;
                    float sum1_1 = 0.f;
                    float sum1_2 = 0.f;
                    float sum1_3 = 0.f;
                    float sum1_4 = 0.f;
                    float sum1_5 = 0.f;
                    float sum1_6 = 0.f;
                    float sum1_7 = 0.f;

                    float sum2_0 = 0.f;
                    float sum2_1 = 0.f;
                    float sum2_2 = 0.f;
                    float sum2_3 = 0.f;
                    float sum2_4 = 0.f;
                    float sum2_5 = 0.f;
                    float sum2_6 = 0.f;
                    float sum2_7 = 0.f;

                    float sum3_0 = 0.f;
                    float sum3_1 = 0.f;
                    float sum3_2 = 0.f;
                    float sum3_3 = 0.f;
                    float sum3_4 = 0.f;
                    float sum3_5 = 0.f;
                    float sum3_6 = 0.f;
                    float sum3_7 = 0.f;

                    for (int q=0; q<inch; q++)
                    {
                        sum0_0 += bb2p0[0] * ktm0[0];
                        sum0_1 += bb2p0[1] * ktm0[0];
                        sum0_2 += bb2p0[2] * ktm0[0];
                        sum0_3 += bb2p0[3] * ktm0[0];
                        sum0_4 += bb2p0[4] * ktm0[0];
                        sum0_5 += bb2p0[5] * ktm0[0];
                        sum0_6 += bb2p0[6] * ktm0[0];
                        sum0_7 += bb2p0[7] * ktm0[0];

                        sum1_0 += bb2p0[0] * ktm0[1];
                        sum1_1 += bb2p0[1] * ktm0[1];
                        sum1_2 += bb2p0[2] * ktm0[1];
                        sum1_3 += bb2p0[3] * ktm0[1];
                        sum1_4 += bb2p0[4] * ktm0[1];
                        sum1_5 += bb2p0[5] * ktm0[1];
                        sum1_6 += bb2p0[6] * ktm0[1];
                        sum1_7 += bb2p0[7] * ktm0[1];

                        sum2_0 += bb2p0[0] * ktm0[2];
                        sum2_1 += bb2p0[1] * ktm0[2];
                        sum2_2 += bb2p0[2] * ktm0[2];
                        sum2_3 += bb2p0[3] * ktm0[2];
                        sum2_4 += bb2p0[4] * ktm0[2];
                        sum2_5 += bb2p0[5] * ktm0[2];
                        sum2_6 += bb2p0[6] * ktm0[2];
                        sum2_7 += bb2p0[7] * ktm0[2];

                        sum3_0 += bb2p0[0] * ktm0[3];
                        sum3_1 += bb2p0[1] * ktm0[3];
                        sum3_2 += bb2p0[2] * ktm0[3];
                        sum3_3 += bb2p0[3] * ktm0[3];
                        sum3_4 += bb2p0[4] * ktm0[3];
                        sum3_5 += bb2p0[5] * ktm0[3];
                        sum3_6 += bb2p0[6] * ktm0[3];
                        sum3_7 += bb2p0[7] * ktm0[3];

                        bb2p0 += 8;
                        ktm0 += 4;
                    }

                    output0_tm[0] = sum0_0;
                    output0_tm[1] = sum0_1;
                    output0_tm[2] = sum0_2;
                    output0_tm[3] = sum0_3;
                    output0_tm[4] = sum0_4;
                    output0_tm[5] = sum0_5;
                    output0_tm[6] = sum0_6;
                    output0_tm[7] = sum0_7;

                    output1_tm[0] = sum1_0;
                    output1_tm[1] = sum1_1;
                    output1_tm[2] = sum1_2;
                    output1_tm[3] = sum1_3;
                    output1_tm[4] = sum1_4;
                    output1_tm[5] = sum1_5;
                    output1_tm[6] = sum1_6;
                    output1_tm[7] = sum1_7;

                    output2_tm[0] = sum2_0;
                    output2_tm[1] = sum2_1;
                    output2_tm[2] = sum2_2;
                    output2_tm[3] = sum2_3;
                    output2_tm[4] = sum2_4;
                    output2_tm[5] = sum2_5;
                    output2_tm[6] = sum2_6;
                    output2_tm[7] = sum2_7;

                    output3_tm[0] = sum3_0;
                    output3_tm[1] = sum3_1;
                    output3_tm[2] = sum3_2;
                    output3_tm[3] = sum3_3;
                    output3_tm[4] = sum3_4;
                    output3_tm[5] = sum3_5;
                    output3_tm[6] = sum3_6;
                    output3_tm[7] = sum3_7;

                    output0_tm += 8;
                    output1_tm += 8;
                    output2_tm += 8;
                    output3_tm += 8;
#endif // __ARM_NEON
                }
                for (; i+3<tiles; i+=4)
                {
                    const float* bb2p0 = bb2.row(i/8+(i%8)/4);

                    const float* ktm0 = kernel_tm0.row(r);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b     \n"
                        "eor    v9.16b, v9.16b, v9.16b     \n"
                        "eor    v10.16b, v10.16b, v10.16b  \n"
                        "eor    v11.16b, v11.16b, v11.16b  \n"

                        // inch loop
                        "lsr    w4, %w12, #2            \n"// w4 = nn = inch >> 2
                        "cmp    w4, #0                  \n"
                        "beq    1f                      \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%4, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

                        "prfm   pldl1keep, [%5, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v9.4s, v4.4s, v0.s[1]   \n"
                        "fmla   v10.4s, v4.4s, v0.s[2]  \n"
                        "fmla   v11.4s, v4.4s, v0.s[3]  \n"

                        "fmla   v8.4s, v5.4s, v1.s[0]   \n"
                        "fmla   v9.4s, v5.4s, v1.s[1]   \n"
                        "fmla   v10.4s, v5.4s, v1.s[2]  \n"
                        "fmla   v11.4s, v5.4s, v1.s[3]  \n"

                        "fmla   v8.4s, v6.4s, v2.s[0]   \n"
                        "fmla   v9.4s, v6.4s, v2.s[1]   \n"
                        "fmla   v10.4s, v6.4s, v2.s[2]  \n"
                        "fmla   v11.4s, v6.4s, v2.s[3]  \n"

                        "fmla   v8.4s, v7.4s, v3.s[0]   \n"
                        "fmla   v9.4s, v7.4s, v3.s[1]   \n"
                        "fmla   v10.4s, v7.4s, v3.s[2]  \n"
                        "fmla   v11.4s, v7.4s, v3.s[3]  \n"

                        "subs   w4, w4, #1              \n"
                        "bne    0b                      \n"

                        "1:                             \n"

                        // remain loop
                        "and    w4, %w12, #3            \n"// w4 = remain = tiles & 3;
                        "cmp    w4, #0                  \n"
                        "beq    3f                      \n"

                        "2:                             \n"

                        "prfm   pldl1keep, [%4, #128]   \n"
                        "ld1    {v4.4s}, [%4], #16      \n"

                        "prfm   pldl1keep, [%5, #128]   \n"
                        "ld1    {v0.4s}, [%5], #16      \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v9.4s, v4.4s, v0.s[1]   \n"
                        "fmla   v10.4s, v4.4s, v0.s[2]  \n"
                        "fmla   v11.4s, v4.4s, v0.s[3]  \n"

                        "subs   w4, w4, #1              \n"
                        "bne    2b                      \n"

                        "3:                             \n"

                        "st1    {v8.4s}, [%0], #16      \n"
                        "st1    {v9.4s}, [%1], #16      \n"
                        "st1    {v10.4s}, [%2], #16     \n"
                        "st1    {v11.4s}, [%3], #16     \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(bb2p0),      // %4
                          "=r"(ktm0)        // %5
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(bb2p0),
                          "5"(ktm0),
                          "r"(inch)         // %12
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
                    );
#else // __aarch64__
                    asm volatile(
                        "veor       q8, q8, q8      \n"
                        "veor       q9, q9, q9      \n"
                        "veor       q10, q10, q10   \n"
                        "veor       q11, q11, q11   \n"

                        // inch loop
                        "lsr        r4, %12, #2     \n"// r4 = nn = inch >> 2
                        "cmp        r4, #0          \n"
                        "beq        1f              \n"

                        "0:                         \n"

                        "pld        [%4, #512]      \n"
                        "vldm       %4!, {d8-d15}   \n"
//                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
//                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

                        "pld        [%5, #512]      \n"
                        "vldm       %5!, {d0-d7}    \n"
//                         "vld1.f32   {d0-d3}, [%5 :128]!  \n"
//                         "vld1.f32   {d4-d7}, [%5 :128]!  \n"

                        "vmla.f32   q8, q4, d0[0]   \n"
                        "vmla.f32   q9, q4, d0[1]   \n"
                        "vmla.f32   q10, q4, d1[0]  \n"
                        "vmla.f32   q11, q4, d1[1]  \n"

                        "vmla.f32   q8, q5, d2[0]   \n"
                        "vmla.f32   q9, q5, d2[1]   \n"
                        "vmla.f32   q10, q5, d3[0]  \n"
                        "vmla.f32   q11, q5, d3[1]  \n"

                        "subs       r4, r4, #1      \n"

                        "vmla.f32   q8, q6, d4[0]   \n"
                        "vmla.f32   q9, q6, d4[1]   \n"
                        "vmla.f32   q10, q6, d5[0]  \n"
                        "vmla.f32   q11, q6, d5[1]  \n"

                        "vmla.f32   q8, q7, d6[0]   \n"
                        "vmla.f32   q9, q7, d6[1]   \n"
                        "vmla.f32   q10, q7, d7[0]  \n"
                        "vmla.f32   q11, q7, d7[1]  \n"

                        "bne        0b              \n"

                        "1:                         \n"

                        // remain loop
                        "and        r4, %12, #3     \n"// r4 = remain = tiles & 3;
                        "cmp        r4, #0          \n"
                        "beq        3f              \n"

                        "2:                         \n"

                        "pld        [%4, #128]      \n"
                        "vld1.f32   {d8-d9}, [%4 :128]!  \n"

                        "pld        [%5, #128]      \n"
                        "vld1.f32   {d0-d1}, [%5 :128]!  \n"

                        "subs       r4, r4, #1      \n"

                        "vmla.f32   q8, q4, d0[0]   \n"
                        "vmla.f32   q9, q4, d0[1]   \n"
                        "vmla.f32   q10, q4, d1[0]  \n"
                        "vmla.f32   q11, q4, d1[1]  \n"

                        "bne        2b              \n"

                        "3:                         \n"

                        "vst1.f32   {d16-d17}, [%0]! \n"
                        "vst1.f32   {d18-d19}, [%1]! \n"
                        "vst1.f32   {d20-d21}, [%2]! \n"
                        "vst1.f32   {d22-d23}, [%3]! \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(bb2p0),      // %4
                          "=r"(ktm0)        // %5
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(bb2p0),
                          "5"(ktm0),
                          "r"(inch)         // %12
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
                    );
#endif // __aarch64__
#else
                    float sum0_0 = 0.f;
                    float sum0_1 = 0.f;
                    float sum0_2 = 0.f;
                    float sum0_3 = 0.f;

                    float sum1_0 = 0.f;
                    float sum1_1 = 0.f;
                    float sum1_2 = 0.f;
                    float sum1_3 = 0.f;

                    float sum2_0 = 0.f;
                    float sum2_1 = 0.f;
                    float sum2_2 = 0.f;
                    float sum2_3 = 0.f;

                    float sum3_0 = 0.f;
                    float sum3_1 = 0.f;
                    float sum3_2 = 0.f;
                    float sum3_3 = 0.f;

                    for (int q=0; q<inch; q++)
                    {
                        sum0_0 += bb2p0[0] * ktm0[0];
                        sum0_1 += bb2p0[1] * ktm0[0];
                        sum0_2 += bb2p0[2] * ktm0[0];
                        sum0_3 += bb2p0[3] * ktm0[0];

                        sum1_0 += bb2p0[0] * ktm0[1];
                        sum1_1 += bb2p0[1] * ktm0[1];
                        sum1_2 += bb2p0[2] * ktm0[1];
                        sum1_3 += bb2p0[3] * ktm0[1];

                        sum2_0 += bb2p0[0] * ktm0[2];
                        sum2_1 += bb2p0[1] * ktm0[2];
                        sum2_2 += bb2p0[2] * ktm0[2];
                        sum2_3 += bb2p0[3] * ktm0[2];

                        sum3_0 += bb2p0[0] * ktm0[3];
                        sum3_1 += bb2p0[1] * ktm0[3];
                        sum3_2 += bb2p0[2] * ktm0[3];
                        sum3_3 += bb2p0[3] * ktm0[3];

                        bb2p0 += 4;
                        ktm0 += 4;
                    }

                    output0_tm[0] = sum0_0;
                    output0_tm[1] = sum0_1;
                    output0_tm[2] = sum0_2;
                    output0_tm[3] = sum0_3;

                    output1_tm[0] = sum1_0;
                    output1_tm[1] = sum1_1;
                    output1_tm[2] = sum1_2;
                    output1_tm[3] = sum1_3;

                    output2_tm[0] = sum2_0;
                    output2_tm[1] = sum2_1;
                    output2_tm[2] = sum2_2;
                    output2_tm[3] = sum2_3;

                    output3_tm[0] = sum3_0;
                    output3_tm[1] = sum3_1;
                    output3_tm[2] = sum3_2;
                    output3_tm[3] = sum3_3;

                    output0_tm += 4;
                    output1_tm += 4;
                    output2_tm += 4;
                    output3_tm += 4;
#endif // __ARM_NEON
                }
                for (; i<tiles; i++)
                {
                    const float* bb2p0 = bb2.row(i/8+(i%8)/4+i%4);

                    const float* ktm0 = kernel_tm0.row(r);

#if __ARM_NEON
                    float32x4_t _sum0123 = vdupq_n_f32(0.f);

                    int q=0;
                    for (; q+3<inch; q+=4)
                    {
//                         asm volatile("prfm pldl1keep, [%0, #128] \n" : :"r"(bb2p0) :);
                        float32x4_t _bb2p0 = vld1q_f32(bb2p0);
                        bb2p0 += 4;

//                         asm volatile("prfm pldl1keep, [%0, #512] \n" : :"r"(ktm0) :);
                        float32x4_t _ktm0 = vld1q_f32(ktm0 + 0);
                        float32x4_t _ktm1 = vld1q_f32(ktm0 + 4);
                        float32x4_t _ktm2 = vld1q_f32(ktm0 + 8);
                        float32x4_t _ktm3 = vld1q_f32(ktm0 + 12);
                        ktm0 += 16;

#if __aarch64__
                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm0, _bb2p0, 0);
                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm1, _bb2p0, 1);
                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm2, _bb2p0, 2);
                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm3, _bb2p0, 3);
#else
                        _sum0123 = vmlaq_lane_f32(_sum0123, _ktm0, vget_low_f32(_bb2p0), 0);
                        _sum0123 = vmlaq_lane_f32(_sum0123, _ktm1, vget_low_f32(_bb2p0), 1);
                        _sum0123 = vmlaq_lane_f32(_sum0123, _ktm2, vget_high_f32(_bb2p0), 0);
                        _sum0123 = vmlaq_lane_f32(_sum0123, _ktm3, vget_high_f32(_bb2p0), 1);
#endif // __aarch64__
                    }

                    for (; q<inch; q++)
                    {
                        float32x4_t _bb2p0 = vld1q_dup_f32(bb2p0);
                        float32x4_t _ktm0 = vld1q_f32(ktm0);

                        _sum0123 = vmlaq_f32(_sum0123, _bb2p0, _ktm0);

                        bb2p0 += 1;
                        ktm0 += 4;
                    }

                    float sum0 = vgetq_lane_f32(_sum0123, 0);
                    float sum1 = vgetq_lane_f32(_sum0123, 1);
                    float sum2 = vgetq_lane_f32(_sum0123, 2);
                    float sum3 = vgetq_lane_f32(_sum0123, 3);
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum2 = 0.f;
                    float sum3 = 0.f;

                    for (int q=0; q<inch; q++)
                    {
                        sum0 += bb2p0[0] * ktm0[0];
                        sum1 += bb2p0[0] * ktm0[1];
                        sum2 += bb2p0[0] * ktm0[2];
                        sum3 += bb2p0[0] * ktm0[3];

                        bb2p0 += 1;
                        ktm0 += 4;
                    }
#endif // __ARM_NEON

                    output0_tm[0] = sum0;
                    output1_tm[0] = sum1;
                    output2_tm[0] = sum2;
                    output3_tm[0] = sum3;

                    output0_tm += 1;
                    output1_tm += 1;
                    output2_tm += 1;
                    output3_tm += 1;
                }
            }
        }

        remain_outch_start += nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=remain_outch_start; p<outch; p++)
        {
#if __ARM_NEON && __aarch64__
            const Mat kernel_tm0 = kernel_tm.channel(p/8+(p%8)/4+p%4);
#else
            const Mat kernel_tm0 = kernel_tm.channel(p/4+p%4);
#endif

            Mat out0_tm = top_blob_tm.channel(p);

            float* output0_tm = out0_tm;

            for (int r=0; r<64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                // tile
                int i=0;
                for (; i+7<tiles; i+=8)
                {
                    const float* bb2p0 = bb2.row(i/8);

                    const float* ktm0 = kernel_tm0.row(r);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b     \n"
                        "eor    v9.16b, v9.16b, v9.16b     \n"

                        // inch loop
                        "lsr    w4, %w6, #2             \n"// w4 = nn = inch >> 2
                        "cmp    w4, #0                  \n"
                        "beq    1f                      \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%1, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64     \n"

                        "prfm   pldl1keep, [%2, #128]   \n"
                        "ld1    {v0.4s}, [%2], #16      \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v9.4s, v5.4s, v0.s[0]   \n"
                        "fmla   v8.4s, v6.4s, v0.s[1]   \n"
                        "fmla   v9.4s, v7.4s, v0.s[1]   \n"

                        "prfm   pldl1keep, [%1, #512]   \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"

                        "fmla   v8.4s, v12.4s, v0.s[2]  \n"
                        "fmla   v9.4s, v13.4s, v0.s[2]  \n"
                        "fmla   v8.4s, v14.4s, v0.s[3]  \n"
                        "fmla   v9.4s, v15.4s, v0.s[3]  \n"

                        "subs   w4, w4, #1              \n"
                        "bne    0b                      \n"

                        "1:                             \n"

                        // remain loop
                        "and    w4, %w6, #3             \n"// w4 = remain = tiles & 3;
                        "cmp    w4, #0                  \n"
                        "beq    3f                      \n"

                        "2:                             \n"

                        "prfm   pldl1keep, [%1, #256]   \n"
                        "ld1    {v4.4s, v5.4s}, [%1], #32      \n"

                        "prfm   pldl1keep, [%2, #32]    \n"
                        "ld1r   {v0.4s}, [%2], #4       \n"

                        "fmla   v8.4s, v4.4s, v0.4s     \n"
                        "fmla   v9.4s, v5.4s, v0.4s     \n"

                        "subs   w4, w4, #1              \n"
                        "bne    2b                      \n"

                        "3:                             \n"

                        "st1    {v8.4s, v9.4s}, [%0], #32       \n"

                        : "=r"(output0_tm), // %0
                          "=r"(bb2p0),      // %1
                          "=r"(ktm0)        // %2
                        : "0"(output0_tm),
                          "1"(bb2p0),
                          "2"(ktm0),
                          "r"(inch)         // %6
                        : "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v12", "v13", "v14", "v15"
                    );
#else // __aarch64__
                    asm volatile(
                        "veor       q8, q8, q8          \n"
                        "veor       q9, q9, q9          \n"

                        // inch loop
                        "lsr        r4, %6, #2          \n"// r4 = nn = inch >> 2
                        "cmp        r4, #0              \n"
                        "beq        1f                  \n"

                        "0:                             \n"

                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d8-d15}       \n"
//                         "vld1.f32   {d8-d11}, [%1 :128]! \n"
//                         "vld1.f32   {d12-d15}, [%1 :128]! \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d0-d1}, [%2 :128]! \n"

                        "vmla.f32   q8, q4, d0[0]       \n"
                        "vmla.f32   q9, q5, d0[0]       \n"
                        "vmla.f32   q8, q6, d0[1]       \n"
                        "vmla.f32   q9, q7, d0[1]       \n"

                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d24-d31}      \n"
//                         "vld1.f32   {d24-d27}, [%1 :128]! \n"
//                         "vld1.f32   {d28-d31}, [%1 :128]! \n"

                        "subs       r4, r4, #1          \n"

                        "vmla.f32   q8, q12, d1[0]      \n"
                        "vmla.f32   q9, q13, d1[0]      \n"
                        "vmla.f32   q8, q14, d1[1]      \n"
                        "vmla.f32   q9, q15, d1[1]      \n"

                        "bne        0b                  \n"

                        "1:                             \n"

                        // remain loop
                        "and        r4, %6, #3          \n"// r4 = remain = tiles & 3;
                        "cmp        r4, #0              \n"
                        "beq        3f                  \n"

                        "2:                             \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d8-d11}, [%1 :128]! \n"

                        "pld        [%2, #32]           \n"
                        "vld1.f32   {d0[],d1[]}, [%2]!  \n"

                        "subs       r4, r4, #1          \n"

                        "vmla.f32   q8, q4, q0          \n"
                        "vmla.f32   q9, q5, q0          \n"

                        "bne        2b                  \n"

                        "3:                             \n"

                        "vst1.f32   {d16-d19}, [%0]!    \n"

                        : "=r"(output0_tm), // %0
                          "=r"(bb2p0),      // %1
                          "=r"(ktm0)        // %2
                        : "0"(output0_tm),
                          "1"(bb2p0),
                          "2"(ktm0),
                          "r"(inch)         // %6
                        : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum2 = 0.f;
                    float sum3 = 0.f;
                    float sum4 = 0.f;
                    float sum5 = 0.f;
                    float sum6 = 0.f;
                    float sum7 = 0.f;

                    for (int q=0; q<inch; q++)
                    {
                        sum0 += bb2p0[0] * ktm0[0];
                        sum1 += bb2p0[1] * ktm0[0];
                        sum2 += bb2p0[2] * ktm0[0];
                        sum3 += bb2p0[3] * ktm0[0];
                        sum4 += bb2p0[4] * ktm0[0];
                        sum5 += bb2p0[5] * ktm0[0];
                        sum6 += bb2p0[6] * ktm0[0];
                        sum7 += bb2p0[7] * ktm0[0];

                        bb2p0 += 8;
                        ktm0 += 1;
                    }

                    output0_tm[0] = sum0;
                    output0_tm[1] = sum1;
                    output0_tm[2] = sum2;
                    output0_tm[3] = sum3;
                    output0_tm[4] = sum4;
                    output0_tm[5] = sum5;
                    output0_tm[6] = sum6;
                    output0_tm[7] = sum7;

                    output0_tm += 8;
#endif // __ARM_NEON
                }
                for (; i+3<tiles; i+=4)
                {
                    const float* bb2p0 = bb2.row(i/8+(i%8)/4);

                    const float* ktm0 = kernel_tm0.row(r);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b     \n"

                        // inch loop
                        "lsr    w4, %w6, #2             \n"// w4 = nn = inch >> 2
                        "cmp    w4, #0                  \n"
                        "beq    1f                      \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%4, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

                        "prfm   pldl1keep, [%5, #128]   \n"
                        "ld1    {v0.4s}, [%5], #16      \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v8.4s, v5.4s, v0.s[1]   \n"
                        "fmla   v8.4s, v6.4s, v0.s[2]   \n"
                        "fmla   v8.4s, v7.4s, v0.s[3]   \n"

                        "subs   w4, w4, #1              \n"
                        "bne    0b                      \n"

                        "1:                             \n"

                        // remain loop
                        "and    w4, %w6, #3             \n"// w4 = remain = tiles & 3;
                        "cmp    w4, #0                  \n"
                        "beq    3f                      \n"

                        "2:                             \n"

                        "prfm   pldl1keep, [%4, #128]   \n"
                        "ld1    {v4.4s}, [%4], #16      \n"

                        "prfm   pldl1keep, [%5, #32]    \n"
                        "ld1r   {v0.4s}, [%5], #4       \n"

                        "fmla   v8.4s, v4.4s, v0.4s     \n"

                        "subs   w4, w4, #1              \n"
                        "bne    2b                      \n"

                        "3:                             \n"

                        "st1    {v8.4s}, [%0], #16      \n"

                        : "=r"(output0_tm), // %0
                          "=r"(bb2p0),      // %1
                          "=r"(ktm0)        // %2
                        : "0"(output0_tm),
                          "1"(bb2p0),
                          "2"(ktm0),
                          "r"(inch)         // %6
                        : "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8"
                    );
#else // __aarch64__
                    asm volatile(
                        "veor       q8, q8, q8          \n"

                        // inch loop
                        "lsr        r4, %6, #2          \n"// r4 = nn = inch >> 2
                        "cmp        r4, #0              \n"
                        "beq        1f                  \n"

                        "0:                             \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d8-d15}       \n"
//                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
//                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

                        "pld        [%5, #128]          \n"
                        "vld1.f32   {d0-d1}, [%5 :128]! \n"

                        "subs       r4, r4, #1          \n"

                        "vmla.f32   q8, q4, d0[0]       \n"
                        "vmla.f32   q8, q5, d0[1]       \n"
                        "vmla.f32   q8, q6, d1[0]       \n"
                        "vmla.f32   q8, q7, d1[1]       \n"

                        "bne        0b                  \n"

                        "1:                             \n"

                        // remain loop
                        "and        r4, %6, #3          \n"// r4 = remain = tiles & 3;
                        "cmp        r4, #0              \n"
                        "beq        3f                  \n"

                        "2:                             \n"

                        "pld        [%4, #128]          \n"
                        "vld1.f32   {d8-d9}, [%4]!      \n"

                        "pld        [%5, #32]           \n"
                        "vld1.f32   {d0[],d1[]}, [%5]!  \n"

                        "subs       r4, r4, #1          \n"

                        "vmla.f32   q8, q4, q0          \n"

                        "bne        2b                  \n"

                        "3:                             \n"

                        "vst1.f32   {d16-d17}, [%0]!    \n"

                        : "=r"(output0_tm), // %0
                          "=r"(bb2p0),      // %1
                          "=r"(ktm0)        // %2
                        : "0"(output0_tm),
                          "1"(bb2p0),
                          "2"(ktm0),
                          "r"(inch)         // %6
                        : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8"
                    );
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum2 = 0.f;
                    float sum3 = 0.f;

                    for (int q=0; q<inch; q++)
                    {
                        sum0 += bb2p0[0] * ktm0[0];
                        sum1 += bb2p0[1] * ktm0[0];
                        sum2 += bb2p0[2] * ktm0[0];
                        sum3 += bb2p0[3] * ktm0[0];

                        bb2p0 += 4;
                        ktm0 += 1;
                    }

                    output0_tm[0] = sum0;
                    output0_tm[1] = sum1;
                    output0_tm[2] = sum2;
                    output0_tm[3] = sum3;

                    output0_tm += 4;
#endif // __ARM_NEON
                }
                for (; i<tiles; i++)
                {
                    const float* bb2p0 = bb2.row(i/8+(i%8)/4+i%4);

                    const float* ktm0 = kernel_tm0.row(r);

                    int q=0;
#if __ARM_NEON
                    float32x4_t _sum0 = vdupq_n_f32(0.f);
                    for (; q+3<inch; q+=4)
                    {
//                         asm volatile("prfm pldl1keep, [%0, #128] \n" : :"r"(bb2p0) :);
                        float32x4_t _bb2p0 = vld1q_f32(bb2p0);
                        bb2p0 += 4;

                        float32x4_t _ktm0 = vld1q_f32(ktm0);
                        ktm0 += 4;

                        _sum0 = vmlaq_f32(_sum0, _bb2p0, _ktm0);
                    }

#if __aarch64__
                    float sum0 = vaddvq_f32(_sum0);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float sum0 = vget_lane_f32(vpadd_f32(_ss0, _ss0), 0);
#endif // __aarch64__
#else
                    float sum0 = 0.f;
#endif
                    for (; q<inch; q++)
                    {
                        sum0 += bb2p0[0] * ktm0[0];

                        bb2p0 += 1;
                        ktm0 += 1;
                    }

                    output0_tm[0] = sum0;

                    output0_tm += 1;
                }
            }
        }
    }
}
}
