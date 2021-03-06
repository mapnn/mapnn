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
namespace ncnn {
static void conv1x1s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 4;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);
        Mat out2 = top_blob.channel(p+2);
        Mat out3 = top_blob.channel(p+3);

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p+1] : 0.f;
        const float bias2 = bias ? bias[p+2] : 0.f;
        const float bias3 = bias ? bias[p+3] : 0.f;

        out0.fill(bias0);
        out1.fill(bias1);
        out2.fill(bias2);
        out3.fill(bias3);

        int q = 0;

        for (; q+3<inch; q+=4)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q+1);
            const float* img2 = bottom_blob.channel(q+2);
            const float* img3 = bottom_blob.channel(q+3);

            const float* kernel0 = kernel + p*inch + q;
            const float* kernel1 = kernel + (p+1)*inch + q;
            const float* kernel2 = kernel + (p+2)*inch + q;
            const float* kernel3 = kernel + (p+3)*inch + q;

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            for (int i = 0; i < outh; i++)
            {
                int size = outw;

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _k0 = vld1q_f32(kernel0);
                float32x4_t _k1 = vld1q_f32(kernel1);
                float32x4_t _k2 = vld1q_f32(kernel2);
                float32x4_t _k3 = vld1q_f32(kernel3);
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                                        \n"

                    "prfm       pldl1keep, [%5, #512]          \n"
                    "ld2        {v4.4s, v5.4s}, [%5], #32      \n"
                    "ld2        {v6.4s, v7.4s}, [%5], #32      \n"
                    "and        v5.16b, v6.16b, v6.16b         \n"// v4 v5

                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v8.4s, v9.4s}, [%1]           \n"

                    "fmla       v8.4s, v4.4s, %18.s[0]         \n"
                    "fmla       v9.4s, v5.4s, %18.s[0]         \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v10.4s, v11.4s}, [%2]         \n"

                    "fmla       v10.4s, v4.4s, %19.s[0]        \n"
                    "fmla       v11.4s, v5.4s, %19.s[0]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld1        {v12.4s, v13.4s}, [%3]         \n"

                    "fmla       v12.4s, v4.4s, %20.s[0]        \n"
                    "fmla       v13.4s, v5.4s, %20.s[0]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld1        {v14.4s, v15.4s}, [%4]         \n"

                    "prfm       pldl1keep, [%6, #512]          \n"
                    "ld2        {v6.4s, v7.4s}, [%6], #32      \n"

                    "fmla       v14.4s, v4.4s, %21.s[0]        \n"
                    "fmla       v15.4s, v5.4s, %21.s[0]        \n"

                    "ld2        {v4.4s, v5.4s}, [%6], #32      \n"
                    "and        v7.16b, v4.16b, v4.16b         \n"// v6 v7

                    "fmla       v8.4s, v6.4s, %18.s[1]         \n"
                    "fmla       v9.4s, v7.4s, %18.s[1]         \n"

                    "fmla       v10.4s, v6.4s, %19.s[1]        \n"
                    "fmla       v11.4s, v7.4s, %19.s[1]        \n"

                    "fmla       v12.4s, v6.4s, %20.s[1]        \n"
                    "fmla       v13.4s, v7.4s, %20.s[1]        \n"

                    "prfm       pldl1keep, [%7, #512]          \n"
                    "ld2        {v4.4s, v5.4s}, [%7], #32      \n"

                    "fmla       v14.4s, v6.4s, %21.s[1]        \n"
                    "fmla       v15.4s, v7.4s, %21.s[1]        \n"

                    "ld2        {v6.4s, v7.4s}, [%7], #32      \n"
                    "and        v5.16b, v6.16b, v6.16b         \n"// v4 v5

                    "fmla       v8.4s, v4.4s, %18.s[2]         \n"
                    "fmla       v9.4s, v5.4s, %18.s[2]         \n"

                    "fmla       v10.4s, v4.4s, %19.s[2]        \n"
                    "fmla       v11.4s, v5.4s, %19.s[2]        \n"

                    "fmla       v12.4s, v4.4s, %20.s[2]        \n"
                    "fmla       v13.4s, v5.4s, %20.s[2]        \n"

                    "prfm       pldl1keep, [%8, #512]          \n"
                    "ld2        {v6.4s, v7.4s}, [%8], #32      \n"

                    "fmla       v14.4s, v4.4s, %21.s[2]        \n"
                    "fmla       v15.4s, v5.4s, %21.s[2]        \n"

                    "ld2        {v4.4s, v5.4s}, [%8], #32      \n"
                    "and        v7.16b, v4.16b, v4.16b         \n"// v6 v7

                    "fmla       v8.4s, v6.4s, %18.s[3]         \n"
                    "fmla       v9.4s, v7.4s, %18.s[3]         \n"

                    "fmla       v10.4s, v6.4s, %19.s[3]        \n"
                    "fmla       v11.4s, v7.4s, %19.s[3]        \n"

                    "st1        {v8.4s, v9.4s}, [%1], #32      \n"

                    "fmla       v12.4s, v6.4s, %20.s[3]        \n"
                    "fmla       v13.4s, v7.4s, %20.s[3]        \n"

                    "st1        {v10.4s, v11.4s}, [%2], #32    \n"

                    "fmla       v14.4s, v6.4s, %21.s[3]        \n"
                    "fmla       v15.4s, v7.4s, %21.s[3]        \n"

                    "st1        {v12.4s, v13.4s}, [%3], #32    \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v14.4s, v15.4s}, [%4], #32    \n"

                    "bne        0b                             \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr0),// %1
                      "=r"(outptr1),// %2
                      "=r"(outptr2),// %3
                      "=r"(outptr3),// %4
                      "=r"(r0),     // %5
                      "=r"(r1),     // %6
                      "=r"(r2),     // %7
                      "=r"(r3)      // %8
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(r3),
                      "w"(_k0),     // %18
                      "w"(_k1),     // %19
                      "w"(_k2),     // %20
                      "w"(_k3)      // %21
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"

                    "pld        [%5, #512]          \n"
                    "vld2.f32   {d8-d11}, [%5]!     \n"
                    "vld2.f32   {d12-d15}, [%5]!    \n"
                    "vand       q5, q6, q6          \n"// q4 q5

                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d16-d19}, [%1]     \n"

                    "vmla.f32   q8, q4, %e18[0]     \n"
                    "vmla.f32   q9, q5, %e18[0]     \n"

                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d20-d23}, [%2]     \n"

                    "vmla.f32   q10, q4, %e19[0]    \n"
                    "vmla.f32   q11, q5, %e19[0]    \n"

                    "pld        [%3, #256]          \n"
                    "vld1.f32   {d24-d27}, [%3]     \n"

                    "vmla.f32   q12, q4, %e20[0]    \n"
                    "vmla.f32   q13, q5, %e20[0]    \n"

                    "pld        [%4, #256]          \n"
                    "vld1.f32   {d28-d31}, [%4]     \n"

                    "pld        [%6, #512]          \n"
                    "vld2.f32   {d12-d15}, [%6]!    \n"

                    "vmla.f32   q14, q4, %e21[0]    \n"
                    "vmla.f32   q15, q5, %e21[0]    \n"

                    "vld2.f32   {d8-d11}, [%6]!     \n"
                    "vand       q7, q4, q4          \n"// q6 q7

                    "vmla.f32   q8, q6, %e18[1]     \n"
                    "vmla.f32   q9, q7, %e18[1]     \n"

                    "vmla.f32   q10, q6, %e19[1]    \n"
                    "vmla.f32   q11, q7, %e19[1]    \n"

                    "vmla.f32   q12, q6, %e20[1]    \n"
                    "vmla.f32   q13, q7, %e20[1]    \n"

                    "pld        [%7, #512]          \n"
                    "vld2.f32   {d8-d11}, [%7]!     \n"

                    "vmla.f32   q14, q6, %e21[1]    \n"
                    "vmla.f32   q15, q7, %e21[1]    \n"

                    "vld2.f32   {d12-d15}, [%7]!    \n"
                    "vand       q5, q6, q6          \n"// q4 q5

                    "vmla.f32   q8, q4, %f18[0]     \n"
                    "vmla.f32   q9, q5, %f18[0]     \n"

                    "vmla.f32   q10, q4, %f19[0]    \n"
                    "vmla.f32   q11, q5, %f19[0]    \n"

                    "vmla.f32   q12, q4, %f20[0]    \n"
                    "vmla.f32   q13, q5, %f20[0]    \n"

                    "pld        [%8, #512]          \n"
                    "vld2.f32   {d12-d15}, [%8]!    \n"

                    "vmla.f32   q14, q4, %f21[0]    \n"
                    "vmla.f32   q15, q5, %f21[0]    \n"

                    "vld2.f32   {d8-d11}, [%8]!     \n"
                    "vand       q7, q4, q4          \n"// q6 q7

                    "vmla.f32   q8, q6, %f18[1]     \n"
                    "vmla.f32   q9, q7, %f18[1]     \n"

                    "vmla.f32   q10, q6, %f19[1]    \n"
                    "vmla.f32   q11, q7, %f19[1]    \n"

                    "vst1.f32   {d16-d19}, [%1]!    \n"

                    "vmla.f32   q12, q6, %f20[1]    \n"
                    "vmla.f32   q13, q7, %f20[1]    \n"

                    "vst1.f32   {d20-d23}, [%2]!    \n"

                    "vmla.f32   q14, q6, %f21[1]    \n"
                    "vmla.f32   q15, q7, %f21[1]    \n"

                    "vst1.f32   {d24-d27}, [%3]!    \n"

                    "subs       %0, #1              \n"
                    "vst1.f32   {d28-d31}, [%4]!    \n"

                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr0),// %1
                      "=r"(outptr1),// %2
                      "=r"(outptr2),// %3
                      "=r"(outptr3),// %4
                      "=r"(r0),     // %5
                      "=r"(r1),     // %6
                      "=r"(r2),     // %7
                      "=r"(r3)      // %8
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(r3),
                      "w"(_k0),     // %18
                      "w"(_k1),     // %19
                      "w"(_k2),     // %20
                      "w"(_k3)      // %21
                    : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    // TODO neon optimize
                    float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                    float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                    float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                    float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
            }
        }

        for (; q<inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch + q;
            const float* kernel1 = kernel + (p+1)*inch + q;
            const float* kernel2 = kernel + (p+2)*inch + q;
            const float* kernel3 = kernel + (p+3)*inch + q;

            const float k0 = kernel0[0];
            const float k1 = kernel1[0];
            const float k2 = kernel2[0];
            const float k3 = kernel3[0];

            const float* r0 = img0;

            for (int i = 0; i < outh; i++)
            {
                int size = outw;

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _k0 = vdupq_n_f32(k0);
                float32x4_t _k1 = vdupq_n_f32(k1);
                float32x4_t _k2 = vdupq_n_f32(k2);
                float32x4_t _k3 = vdupq_n_f32(k3);
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                                        \n"

                    "prfm       pldl1keep, [%5, #512]          \n"
                    "ld2        {v4.4s, v5.4s}, [%5], #32      \n"
                    "ld2        {v6.4s, v7.4s}, [%5], #32      \n"
                    "and        v5.16b, v6.16b, v6.16b         \n"
                    
                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v8.4s, v9.4s}, [%1]           \n"

                    "fmla       v8.4s, v4.4s, %12.4s           \n"
                    "fmla       v9.4s, v5.4s, %12.4s           \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v10.4s, v11.4s}, [%2]         \n"

                    "fmla       v10.4s, v4.4s, %13.4s          \n"
                    "fmla       v11.4s, v5.4s, %13.4s          \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld1        {v12.4s, v13.4s}, [%3]         \n"

                    "st1        {v8.4s, v9.4s}, [%1], #32      \n"

                    "fmla       v12.4s, v4.4s, %14.4s          \n"
                    "fmla       v13.4s, v5.4s, %14.4s          \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld1        {v14.4s, v15.4s}, [%4]         \n"

                    "st1        {v10.4s, v11.4s}, [%2], #32    \n"
                    
                    "fmla       v14.4s, v4.4s, %15.4s          \n"
                    "fmla       v15.4s, v5.4s, %15.4s          \n"

                    "st1        {v12.4s, v13.4s}, [%3], #32    \n"
                    "subs       %w0, %w0, #1                   \n"
                    
                    "st1        {v14.4s, v15.4s}, [%4], #32    \n"
                    "bne        0b                             \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr0),// %1
                      "=r"(outptr1),// %2
                      "=r"(outptr2),// %3
                      "=r"(outptr3),// %4
                      "=r"(r0)      // %5
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "w"(_k0),     // %12
                      "w"(_k1),     // %13
                      "w"(_k2),     // %14
                      "w"(_k3)      // %15
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"

                    "pld        [%5, #512]          \n"
                    "vld2.f32   {d8-d11}, [%5]!     \n"
                    "vld2.f32   {d12-d15}, [%5]!    \n"
                    "vand       q5, q6, q6          \n"// q4 q5

                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d16-d19}, [%1]     \n"

                    "vmla.f32   q8, q4, %q12        \n"
                    "vmla.f32   q9, q5, %q12        \n"

                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d20-d23}, [%2]     \n"

                    "vmla.f32   q10, q4, %q13       \n"
                    "vmla.f32   q11, q5, %q13       \n"

                    "pld        [%3, #256]          \n"
                    "vld1.f32   {d24-d27}, [%3]     \n"

                    "vst1.f32   {d16-d19}, [%1]!    \n"

                    "vmla.f32   q12, q4, %q14       \n"
                    "vmla.f32   q13, q5, %q14       \n"

                    "pld        [%4, #256]          \n"
                    "vld1.f32   {d28-d31}, [%4]     \n"

                    "vst1.f32   {d20-d23}, [%2]!    \n"

                    "vmla.f32   q14, q4, %q15       \n"
                    "vmla.f32   q15, q5, %q15       \n"

                    "vst1.f32   {d24-d27}, [%3]!    \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d28-d31}, [%4]!    \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr0),// %1
                      "=r"(outptr1),// %2
                      "=r"(outptr2),// %3
                      "=r"(outptr3),// %4
                      "=r"(r0)      // %5
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "w"(_k0),     // %12
                      "w"(_k1),     // %13
                      "w"(_k2),     // %14
                      "w"(_k3)      // %15
                    : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    // TODO neon optimize
                    float sum0 = *r0 * k0;
                    float sum1 = *r0 * k1;
                    float sum2 = *r0 * k2;
                    float sum3 = *r0 * k3;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;

                    r0 += 2;
                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                }

                r0 += tailstep;
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        int q = 0;

        for (; q+3<inch; q+=4)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q+1);
            const float* img2 = bottom_blob.channel(q+2);
            const float* img3 = bottom_blob.channel(q+3);

            const float* kernel0 = kernel + p*inch + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            for (int i = 0; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _k0 = vdupq_n_f32(k0);
                float32x4_t _k1 = vdupq_n_f32(k1);
                float32x4_t _k2 = vdupq_n_f32(k2);
                float32x4_t _k3 = vdupq_n_f32(k3);
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "prfm       pldl1keep, [%2, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%2], #32      \n"
                    "0:                                        \n"

                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v0.4s, v1.4s}, [%1]           \n"
                    "fmla       v0.4s, v2.4s, %12.4s           \n"
                    "fmla       v1.4s, v8.4s, %12.4s           \n"

                    "prfm       pldl1keep, [%3, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%3], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%3], #32      \n"
                    "fmla       v0.4s, v2.4s, %13.4s           \n"
                    "fmla       v1.4s, v8.4s, %13.4s           \n"

                    "prfm       pldl1keep, [%4, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%4], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%4], #32      \n"
                    "fmla       v0.4s, v2.4s, %14.4s           \n"
                    "fmla       v1.4s, v8.4s, %14.4s           \n"

                    "prfm       pldl1keep, [%5, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%5], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%5], #32      \n"
                    "fmla       v0.4s, v2.4s, %15.4s           \n"
                    "fmla       v1.4s, v8.4s, %15.4s           \n"

                    "prfm       pldl1keep, [%2, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%2], #32      \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #64                    \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2),     // %4
                      "=r"(r3)      // %5
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "w"(_k0),     // %12
                      "w"(_k1),     // %13
                      "w"(_k2),     // %14
                      "w"(_k3)      // %15
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%2, #512]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"
                    "vld2.f32   {d16-d19}, [%2]!    \n"
                    "0:                             \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1]       \n"
                    "vmla.f32   q0, q2, %q12        \n"
                    "vmla.f32   q1, q8, %q12        \n"
                    "pld        [%3, #512]          \n"
                    "vld2.f32   {d4-d7}, [%3]!      \n"
                    "vld2.f32   {d16-d19}, [%3]!    \n"
                    "vmla.f32   q0, q2, %q13        \n"
                    "vmla.f32   q1, q8, %q13        \n"
                    "pld        [%4, #512]          \n"
                    "vld2.f32   {d4-d7}, [%4]!      \n"
                    "vld2.f32   {d16-d19}, [%4]!    \n"
                    "vmla.f32   q0, q2, %q14        \n"
                    "vmla.f32   q1, q8, %q14        \n"
                    "pld        [%5, #512]          \n"
                    "vld2.f32   {d4-d7}, [%5]!      \n"
                    "vld2.f32   {d16-d19}, [%5]!    \n"
                    "vmla.f32   q0, q2, %q15        \n"
                    "vmla.f32   q1, q8, %q15        \n"
                    "pld        [%2, #512]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"
                    "vld2.f32   {d16-d19}, [%2]!    \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d3}, [%1]!      \n"
                    "bne        0b                  \n"
                    "sub        %2, #64             \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2),     // %4
                      "=r"(r3)      // %5
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "w"(_k0),     // %12
                      "w"(_k1),     // %13
                      "w"(_k2),     // %14
                      "w"(_k3)      // %15
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    float sum = *r0 * k0;
                    float sum1 = *r1 * k1;
                    float sum2 = *r2 * k2;
                    float sum3 = *r3 * k3;

                    *outptr += sum + sum1 + sum2 + sum3;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
            }

        }

        for (; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch + q;
            const float k0 = kernel0[0];

            const float* r0 = img0;

            for (int i = 0; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _k0 = vdupq_n_f32(k0);
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "prfm       pldl1keep, [%2, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%2], #32      \n"

                    "0:                                        \n"

                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v0.4s, v1.4s}, [%1]           \n"
                    "fmla       v0.4s, v2.4s, %6.4s            \n"
                    "fmla       v1.4s, v8.4s, %6.4s            \n"

                    "prfm       pldl1keep, [%2, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%2], #32      \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #64                    \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0)      // %2
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "w"(_k0)      // %6
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%2, #512]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"
                    "vld2.f32   {d16-d19}, [%2]!    \n"
                    "0:                             \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1]       \n"
                    "vmla.f32   q0, q2, %q6         \n"
                    "vmla.f32   q1, q8, %q6         \n"
                    "pld        [%2, #512]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"
                    "vld2.f32   {d16-d19}, [%2]!    \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d3}, [%1]!      \n"
                    "bne        0b                  \n"
                    "sub        %2, #64             \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0)      // %2
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "w"(_k0)      // %6
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    float sum = *r0 * k0;

                    *outptr += sum;

                    r0 += 2;
                    outptr++;
                }

                r0 += tailstep;
            }

        }
    }

}
}
