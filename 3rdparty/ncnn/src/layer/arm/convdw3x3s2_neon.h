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
static void convdw3x3s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = w - 2*outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g=0; g<group; g++)
    {
        Mat out = top_blob.channel(g);

        const float bias0 = bias ? bias[g] : 0.f;

        const float* kernel0 = kernel + g*9;

        float* outptr = out;

        const float* img0 = bottom_blob.channel(g);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w*2;

#if __ARM_NEON
        float32x4_t _k012x = vld1q_f32(kernel0);
        float32x4_t _k345x = vld1q_f32(kernel0+3);
        float32x4_t _k678x = vld1q_f32(kernel0+6);

        _k012x = vsetq_lane_f32(0.f, _k012x, 3);
        _k345x = vsetq_lane_f32(0.f, _k345x, 3);
        _k678x = vsetq_lane_f32(0.f, _k678x, 3);

        float32x4_t _bias0 = vdupq_n_f32(bias0);
#else
        const float* k0 = kernel0;
        const float* k1 = kernel0 + 3;
        const float* k2 = kernel0 + 6;
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

                "and        v11.16b, %13.16b, %13.16b      \n" // v11 = _bias0

                "0:                                        \n"
                "fmul       v0.4s,  v2.4s, %10.s[0]        \n"
                "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                "prfm       pldl1keep, [%2, #256]          \n"
                "ld2        {v8.4s, v9.4s}, [%2]           \n"
                "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                "fmla       v11.4s, v1.4s, %10.s[2]        \n"

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

                "and        v11.16b, %13.16b, %13.16b      \n" // v11 = _bias0

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
                  "w"(_k012x),  // %10
                  "w"(_k345x),  // %11
                  "w"(_k678x),  // %12
                  "w"(_bias0)   // %13
                : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "pld        [%2, #256]          \n"
                "vld2.f32   {d4-d7}, [%2]!      \n"

                "vand       q11, %q13, %q13     \n"

                "0:                             \n"
                "vmul.f32   q0, q2, %e10[0]     \n"
                "vmul.f32   q10, q3, %e10[1]    \n"

                "pld        [%2, #128]          \n"
                "vld2.f32   {d16-d17}, [%2]     \n"
                "vext.32    q1, q2, q8, #1      \n"

                "vmla.f32   q11, q1, %f10[0]    \n"

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

                "vand       q11, %q13, %q13     \n"

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
                  "w"(_k012x),  // %10
                  "w"(_k345x),  // %11
                  "w"(_k678x),  // %12
                  "w"(_bias0)   // %13
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

                float32x4_t _sum = vmulq_f32(_r00, _k012x);
                _sum = vmlaq_f32(_sum, _r10, _k345x);
                _sum = vmlaq_f32(_sum, _r20, _k678x);

                _sum = vsetq_lane_f32(bias0, _sum, 3);
#if __aarch64__
                *outptr = vaddvq_f32(_sum);
#else
                float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                _ss = vpadd_f32(_ss, _ss);

                *outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

                *outptr = sum;
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

    }
}
}
