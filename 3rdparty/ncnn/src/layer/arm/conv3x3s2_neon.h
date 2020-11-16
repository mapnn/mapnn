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
static void conv3x3s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p+1] : 0.f;

        out0.fill(bias0);
        out1.fill(bias1);

        const float* k0 = kernel + p*inch*9;
        const float* k1 = kernel + (p+1)*inch*9;

        for (int q=0; q<inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;

#if __ARM_NEON
            float32x4_t _k00 = vld1q_f32(k0);
            float32x4_t _k03 = vld1q_f32(k0+3);
            float32x4_t _k06 = vld1q_f32(k0+6);

            float32x4_t _k10 = vld1q_f32(k1);
            float32x4_t _k13 = vld1q_f32(k1+3);
            float32x4_t _k16 = vld1q_f32(k1+6);
#endif // __ARM_NEON

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld2    {v8.4s, v9.4s}, [%3], #32   \n"// v8 v9 = r0

                    "0:                                 \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v6.4s}, [%1]               \n"// v6 = _sum0

                    "fmul   v12.4s, v8.4s, %12.s[0]     \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v7.4s}, [%2]               \n"// v7 = _sum1

                    "fmul   v13.4s, v8.4s, %15.s[0]     \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld2    {v10.4s, v11.4s}, [%3]      \n"// v10

                    "fmla   v6.4s, v9.4s, %12.s[1]      \n"

                    "ext    v14.16b, v8.16b, v10.16b, #4\n"

                    "fmla   v7.4s, v9.4s, %15.s[1]      \n"

                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld2    {v8.4s, v9.4s}, [%4], #32   \n"// r1

                    "fmla   v12.4s, v14.4s, %12.s[2]    \n"
                    "fmla   v13.4s, v14.4s, %15.s[2]    \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld2    {v10.4s, v11.4s}, [%4]      \n"

                    "fmla   v6.4s, v8.4s, %13.s[0]      \n"
                    "fmla   v7.4s, v8.4s, %16.s[0]      \n"

                    "ext    v14.16b, v8.16b, v10.16b, #4\n"

                    "fmla   v12.4s, v9.4s, %13.s[1]     \n"
                    "fmla   v13.4s, v9.4s, %16.s[1]     \n"

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld2    {v8.4s, v9.4s}, [%5], #32   \n"// r2

                    "fmla   v6.4s, v14.4s, %13.s[2]     \n"
                    "fmla   v7.4s, v14.4s, %16.s[2]     \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld2    {v10.4s, v11.4s}, [%5]      \n"

                    "fmla   v12.4s, v8.4s, %14.s[0]     \n"
                    "fmla   v13.4s, v8.4s, %17.s[0]     \n"

                    "ext    v14.16b, v8.16b, v10.16b, #4\n"

                    "fmla   v6.4s, v9.4s, %14.s[1]      \n"
                    "fmla   v7.4s, v9.4s, %17.s[1]      \n"

                    "fmla   v12.4s, v14.4s, %14.s[2]    \n"
                    "fmla   v13.4s, v14.4s, %17.s[2]    \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld2    {v8.4s, v9.4s}, [%3], #32   \n"// v8 v9 = r0

                    "fadd   v6.4s, v6.4s, v12.4s        \n"
                    "fadd   v7.4s, v7.4s, v13.4s        \n"

                    "subs   %w0, %w0, #1                \n"

                    "st1    {v6.4s}, [%1], #16          \n"
                    "st1    {v7.4s}, [%2], #16          \n"

                    "bne    0b                          \n"
                    "sub    %3, %3, #32                 \n"

                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(r0),         // %3
                      "=r"(r1),         // %4
                      "=r"(r2)          // %5
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(r0),
                      "4"(r1),
                      "5"(r2),
                      "w"(_k00),      // %12
                      "w"(_k03),      // %13
                      "w"(_k06),      // %14
                      "w"(_k10),      // %15
                      "w"(_k13),      // %16
                      "w"(_k16)       // %17
                    : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%3, #256]          \n"
                    "vld2.f32   {d16-d19}, [%3]!    \n"// q8 q9 = r0

                    "0:                             \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d12-d13}, [%1]     \n"// q6 = _sum0

                    "vmul.f32   q12, q8, %e12[0]    \n"

                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d14-d15}, [%2]     \n"// q7 = _sum1

                    "vmul.f32   q13, q8, %e15[0]    \n"

                    "pld        [%3, #128]          \n"
                    "vld2.f32   {d20-d21}, [%3]     \n"// q10

                    "vmla.f32   q6, q9, %e12[1]     \n"

                    "vext.32    q11, q8, q10, #1    \n"

                    "vmla.f32   q7, q9, %e15[1]     \n"

                    "pld        [%4, #256]          \n"
                    "vld2.f32   {d16-d19}, [%4]!    \n"// r1

                    "vmla.f32   q12, q11, %f12[0]   \n"
                    "vmla.f32   q13, q11, %f15[0]   \n"

                    "pld        [%4, #128]          \n"
                    "vld2.f32   {d20-d21}, [%4]     \n"

                    "vmla.f32   q6, q8, %e13[0]     \n"
                    "vmla.f32   q7, q8, %e16[0]     \n"

                    "vext.32    q11, q8, q10, #1    \n"

                    "vmla.f32   q12, q9, %e13[1]    \n"
                    "vmla.f32   q13, q9, %e16[1]    \n"

                    "pld        [%5, #256]          \n"
                    "vld2.f32   {d16-d19}, [%5]!    \n"// r2

                    "vmla.f32   q6, q11, %f13[0]    \n"
                    "vmla.f32   q7, q11, %f16[0]    \n"

                    "pld        [%5, #128]          \n"
                    "vld2.f32   {d20-d21}, [%5]     \n"

                    "vmla.f32   q12, q8, %e14[0]    \n"
                    "vmla.f32   q13, q8, %e17[0]    \n"

                    "vext.32    q11, q8, q10, #1    \n"

                    "vmla.f32   q6, q9, %e14[1]     \n"
                    "vmla.f32   q7, q9, %e17[1]     \n"

                    "vmla.f32   q12, q11, %f14[0]   \n"
                    "vmla.f32   q13, q11, %f17[0]   \n"

                    "pld        [%3, #256]          \n"
                    "vld2.f32   {d16-d19}, [%3]!    \n"// q8 q9 = r0

                    "vadd.f32   q6, q6, q12         \n"
                    "vadd.f32   q7, q7, q13         \n"

                    "subs       %0, #1              \n"

                    "vst1.f32   {d12-d13}, [%1]!    \n"
                    "vst1.f32   {d14-d15}, [%2]!    \n"

                    "bne        0b                  \n"
                    "sub        %3, #32             \n"

                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(r0),         // %3
                      "=r"(r1),         // %4
                      "=r"(r2)          // %5
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(r0),
                      "4"(r1),
                      "5"(r2),
                      "w"(_k00),      // %12
                      "w"(_k03),      // %13
                      "w"(_k06),      // %14
                      "w"(_k10),      // %15
                      "w"(_k13),      // %16
                      "w"(_k16)       // %17
                    : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum0 = vmulq_f32(_r00, _k00);
                    float32x4_t _sum1 = vmulq_f32(_r00, _k10);
                    _sum0 = vmlaq_f32(_sum0, _r10, _k03);
                    _sum1 = vmlaq_f32(_sum1, _r10, _k13);
                    _sum0 = vmlaq_f32(_sum0, _r20, _k06);
                    _sum1 = vmlaq_f32(_sum1, _r20, _k16);

                    _sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
                    _sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
#if __aarch64__
                    *outptr0 = vaddvq_f32(_sum0);
                    *outptr1 = vaddvq_f32(_sum1);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);

                    *outptr0 = vget_lane_f32(_ss01, 0);
                    *outptr1 = vget_lane_f32(_ss01, 1);
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
#endif // __ARM_NEON

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                    outptr1++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9;
            k1 += 9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        const float* kernel0 = kernel + p*inch*9;

        for (int q=0; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

#if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(k0);
            float32x4_t _k3456 = vld1q_f32(k1);
            float32x4_t _k6789 = vld1q_f32(k2);
#endif // __ARM_NEON

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                    "0:                                        \n"

                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1]                  \n"

                    "fmla       v0.4s,  v2.4s, %10.s[0]        \n"
                    "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%2]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmul       v11.4s, v1.4s, %10.s[2]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%3], #32      \n"

                    "fmla       v0.4s,  v2.4s, %11.s[0]        \n"
                    "fmla       v10.4s, v3.4s, %11.s[1]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%3]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmla       v11.4s, v1.4s, %11.s[2]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%4], #32      \n"

                    "fmla       v0.4s,  v2.4s, %12.s[0]        \n"
                    "fmla       v10.4s, v3.4s, %12.s[1]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%4]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmla       v11.4s, v1.4s, %12.s[2]        \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                    "fadd       v0.4s, v0.4s, v10.4s           \n"
                    "fadd       v0.4s, v0.4s, v11.4s           \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s}, [%1], #16             \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #32                    \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2)      // %4
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "w"(_k0123),  // %10
                      "w"(_k3456),  // %11
                      "w"(_k6789)   // %12
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%2, #256]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"

                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1]       \n"

                    "vmla.f32   q0, q2, %e10[0]     \n"
                    "vmul.f32   q10, q3, %e10[1]    \n"

                    "pld        [%2, #128]          \n"
                    "vld2.f32   {d16-d17}, [%2]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmul.f32   q11, q1, %f10[0]    \n"

                    "pld        [%3, #256]          \n"
                    "vld2.f32   {d4-d7}, [%3]!      \n"

                    "vmla.f32   q0, q2, %e11[0]     \n"
                    "vmla.f32   q10, q3, %e11[1]    \n"

                    "pld        [%3, #128]          \n"
                    "vld2.f32   {d16-d17}, [%3]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f11[0]    \n"

                    "pld        [%4, #256]          \n"
                    "vld2.f32   {d4-d7}, [%4]!      \n"

                    "vmla.f32   q0, q2, %e12[0]     \n"
                    "vmla.f32   q10, q3, %e12[1]    \n"

                    "pld        [%4, #128]          \n"
                    "vld2.f32   {d16-d17}, [%4]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f12[0]    \n"

                    "pld        [%2, #256]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"

                    "vadd.f32   q0, q0, q10         \n"
                    "vadd.f32   q0, q0, q11         \n"

                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%1]!      \n"
                    "bne        0b                  \n"
                    "sub        %2, #32             \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2)      // %4
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "w"(_k0123),  // %10
                      "w"(_k3456),  // %11
                      "w"(_k6789)   // %12
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);

                    _sum = vsetq_lane_f32(*outptr, _sum, 3);

#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    *outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;
#endif // __ARM_NEON

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
        }
    }
}
}
