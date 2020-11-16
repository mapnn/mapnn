// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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
static void convdw5x5s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    //int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    //int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    const int group = bottom_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g=0; g<group; g++)
    {
        Mat out = top_blob.channel(g);

        const float bias0 = bias ? bias[g] : 0.f;

        const float* kernel0 = kernel + g*25;

        float* outptr = out;

        const float* img0 = bottom_blob.channel(g);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w*2;
        const float* r3 = img0 + w*3;
        const float* r4 = img0 + w*4;

        const float* k0 = kernel0;
        const float* k1 = kernel0 + 5;
        const float* k2 = kernel0 + 10;
        const float* k3 = kernel0 + 15;
        const float* k4 = kernel0 + 20;

#if __ARM_NEON
        float32x4_t _k0123 = vld1q_f32(kernel0);
        float32x4_t _k4567 = vld1q_f32(kernel0+4);
        float32x4_t _k891011 = vld1q_f32(kernel0+8);
        float32x4_t _k12131415 = vld1q_f32(kernel0+12);
        float32x4_t _k16171819 = vld1q_f32(kernel0+16);
        float32x4_t _k20212223 = vld1q_f32(kernel0+20);
        float32x4_t _k24242424 = vdupq_n_f32(kernel0[24]);

        float32x4_t _bias0 = vdupq_n_f32(bias0);
#endif // __ARM_NEON

        int i = 0;

        // NOTE unroll outh 2 results somewhat speed drop :| (about -4%)
        // so we do not implement it here

        for (; i < outh; i++)
        {
#if __ARM_NEON
#if __aarch64__
            int nn = outw >> 3;
            int remain = outw & 7;
#else
            int nn = outw >> 2;
            int remain = outw & 3;
#endif // __aarch64__
#else
            int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                // r0
                "prfm   pldl1keep, [%2, #256]           \n"
                "ld2    {v16.4s, v17.4s}, [%2], #32     \n"// v16 v17 = r00 r01

                "mov    v8.16b, %21.16b                 \n"// v8 = _bias0
                "mov    v9.16b, %21.16b                 \n"// v9 = _bias0

                "prfm   pldl1keep, [%2, #256]           \n"
                "ld2    {v18.4s, v19.4s}, [%2], #32     \n"// v18 v19 = r08 r09

                "0:                                     \n"

                "fmul   v10.4s, v16.4s, %14.s[0]        \n"

                "prfm   pldl1keep, [%2, #256]           \n"
                "ld2    {v20.4s, v21.4s}, [%2]          \n"// v20 v21 = r016 r017

                "fmul   v11.4s, v18.4s, %14.s[0]        \n"

                "ext    v22.16b, v16.16b, v18.16b, #4   \n"// v22 = r02

                "fmla   v8.4s, v17.4s, %14.s[1]         \n"

                "ext    v25.16b, v18.16b, v20.16b, #4   \n"// v25 = r010

                "fmla   v9.4s, v19.4s, %14.s[1]         \n"

                "ext    v23.16b, v17.16b, v19.16b, #4   \n"// v23 = r03

                "fmla   v10.4s, v22.4s, %14.s[2]        \n"

                "ext    v26.16b, v19.16b, v21.16b, #4   \n"// v26 = r011

                "fmla   v11.4s, v25.4s, %14.s[2]        \n"

                "ext    v24.16b, v16.16b, v18.16b, #8   \n"// v24 = r04

                "fmla   v8.4s, v23.4s, %14.s[3]         \n"

                "ext    v27.16b, v18.16b, v20.16b, #8   \n"// v27 = r012

                "fmla   v9.4s, v26.4s, %14.s[3]         \n"

                // r1
                "prfm   pldl1keep, [%3, #256]           \n"
                "ld2    {v12.4s, v13.4s}, [%3], #32     \n"// v12 v13 = r10 r11

                "fmla   v10.4s, v24.4s, %15.s[0]        \n"

                "prfm   pldl1keep, [%3, #256]           \n"
                "ld2    {v14.4s, v15.4s}, [%3], #32     \n"// v14 v15 = r18 r19

                "fmla   v11.4s, v27.4s, %15.s[0]        \n"

                "fmla   v8.4s, v12.4s, %15.s[1]         \n"

                "prfm   pldl1keep, [%3, #256]           \n"
                "ld2    {v20.4s, v21.4s}, [%3]          \n"// v20 v21 = r116 r117

                "fmla   v9.4s, v14.4s, %15.s[1]         \n"

                "ext    v22.16b, v12.16b, v14.16b, #4   \n"// v22 = r12

                "fmla   v10.4s, v13.4s, %15.s[2]        \n"

                "ext    v25.16b, v14.16b, v20.16b, #4   \n"// v25 = r110

                "fmla   v11.4s, v15.4s, %15.s[2]        \n"

                "ext    v23.16b, v13.16b, v15.16b, #4   \n"// v23 = r13

                "fmla   v8.4s, v22.4s, %15.s[3]         \n"

                "ext    v26.16b, v15.16b, v21.16b, #4   \n"// v26 = r111

                "fmla   v9.4s, v25.4s, %15.s[3]         \n"

                "ext    v24.16b, v12.16b, v14.16b, #8   \n"// v24 = r14

                "fmla   v10.4s, v23.4s, %16.s[0]        \n"

                "ext    v27.16b, v14.16b, v20.16b, #8   \n"// v27 = r112

                "fmla   v11.4s, v26.4s, %16.s[0]        \n"

                // r2
                "prfm   pldl1keep, [%4, #256]           \n"
                "ld2    {v16.4s, v17.4s}, [%4], #32     \n"// v16 v17 = r20 r21

                "fmla   v8.4s, v24.4s, %16.s[1]         \n"

                "prfm   pldl1keep, [%4, #256]           \n"
                "ld2    {v18.4s, v19.4s}, [%4], #32     \n"// v18 v19 = r28 r29

                "fmla   v9.4s, v27.4s, %16.s[1]         \n"

                "fmla   v10.4s, v16.4s, %16.s[2]        \n"

                "prfm   pldl1keep, [%4, #256]           \n"
                "ld2    {v20.4s, v21.4s}, [%4]          \n"// v20 v21 = r216 r217

                "fmla   v11.4s, v18.4s, %16.s[2]        \n"

                "ext    v22.16b, v16.16b, v18.16b, #4   \n"// v22 = r22

                "fmla   v8.4s, v17.4s, %16.s[3]         \n"

                "ext    v25.16b, v18.16b, v20.16b, #4   \n"// v25 = r210

                "fmla   v9.4s, v19.4s, %16.s[3]         \n"

                "ext    v23.16b, v17.16b, v19.16b, #4   \n"// v23 = r23

                "fmla   v10.4s, v22.4s, %17.s[0]        \n"

                "ext    v26.16b, v19.16b, v21.16b, #4   \n"// v26 = r211

                "fmla   v11.4s, v25.4s, %17.s[0]        \n"

                "ext    v24.16b, v16.16b, v18.16b, #8   \n"// v24 = r24

                "fmla   v8.4s, v23.4s, %17.s[1]         \n"

                "ext    v27.16b, v18.16b, v20.16b, #8   \n"// v27 = r212

                "fmla   v9.4s, v26.4s, %17.s[1]         \n"

                // r3
                "prfm   pldl1keep, [%5, #256]           \n"
                "ld2    {v12.4s, v13.4s}, [%5], #32     \n"// v12 v13 = r30 r31

                "fmla   v10.4s, v24.4s, %17.s[2]        \n"

                "prfm   pldl1keep, [%5, #256]           \n"
                "ld2    {v14.4s, v15.4s}, [%5], #32     \n"// v14 v15 = r38 r39

                "fmla   v11.4s, v27.4s, %17.s[2]        \n"

                "fmla   v8.4s, v12.4s, %17.s[3]         \n"

                "prfm   pldl1keep, [%5, #256]           \n"
                "ld2    {v20.4s, v21.4s}, [%5]          \n"// v20 v21 = r316 r317

                "fmla   v9.4s, v14.4s, %17.s[3]         \n"

                "ext    v22.16b, v12.16b, v14.16b, #4   \n"// v22 = r32

                "fmla   v10.4s, v13.4s, %18.s[0]        \n"

                "ext    v25.16b, v14.16b, v20.16b, #4   \n"// v25 = r310

                "fmla   v11.4s, v15.4s, %18.s[0]        \n"

                "ext    v23.16b, v13.16b, v15.16b, #4   \n"// v23 = r33

                "fmla   v8.4s, v22.4s, %18.s[1]         \n"

                "ext    v26.16b, v15.16b, v21.16b, #4   \n"// v26 = r311

                "fmla   v9.4s, v25.4s, %18.s[1]         \n"

                "ext    v24.16b, v12.16b, v14.16b, #8   \n"// v24 = r34

                "fmla   v10.4s, v23.4s, %18.s[2]        \n"

                "ext    v27.16b, v14.16b, v20.16b, #8   \n"// v27 = r312

                "fmla   v11.4s, v26.4s, %18.s[2]        \n"

                // r4
                "prfm   pldl1keep, [%6, #256]           \n"
                "ld2    {v16.4s, v17.4s}, [%6], #32     \n"// v16 v17 = r40 r41

                "fmla   v8.4s, v24.4s, %18.s[3]         \n"

                "prfm   pldl1keep, [%6, #256]           \n"
                "ld2    {v18.4s, v19.4s}, [%6], #32     \n"// v18 v19 = r48 r49

                "fmla   v9.4s, v27.4s, %18.s[3]         \n"

                "fmla   v10.4s, v16.4s, %19.s[0]        \n"

                "prfm   pldl1keep, [%6, #256]           \n"
                "ld2    {v20.4s, v21.4s}, [%6]          \n"// v20 v21 = r416 r417

                "fmla   v11.4s, v18.4s, %19.s[0]        \n"

                "ext    v22.16b, v16.16b, v18.16b, #4   \n"// v22 = r42

                "fmla   v8.4s, v17.4s, %19.s[1]         \n"

                "ext    v25.16b, v18.16b, v20.16b, #4   \n"// v25 = r410

                "fmla   v9.4s, v19.4s, %19.s[1]         \n"

                "ext    v23.16b, v17.16b, v19.16b, #4   \n"// v23 = r43

                "fmla   v10.4s, v22.4s, %19.s[2]        \n"

                "ext    v26.16b, v19.16b, v21.16b, #4   \n"// v26 = r411

                "fmla   v11.4s, v25.4s, %19.s[2]        \n"

                "ext    v24.16b, v16.16b, v18.16b, #8   \n"// v24 = r44

                "fmla   v8.4s, v23.4s, %19.s[3]         \n"

                "ext    v27.16b, v18.16b, v20.16b, #8   \n"// v27 = r412

                "fmla   v9.4s, v26.4s, %19.s[3]         \n"
                "fmla   v10.4s, v24.4s, %20.s[0]        \n"

                // r0
                "prfm   pldl1keep, [%2, #256]           \n"
                "ld2    {v16.4s, v17.4s}, [%2], #32     \n"// v16 v17 = r00 r01

                "fmla   v11.4s, v27.4s, %20.s[0]        \n"

                "prfm   pldl1keep, [%2, #256]           \n"
                "ld2    {v18.4s, v19.4s}, [%2], #32     \n"// v18 v19 = r08 r09

                "fadd   v10.4s, v8.4s, v10.4s           \n"
                "fadd   v11.4s, v9.4s, v11.4s           \n"

                "subs   %w0, %w0, #1                    \n"

                "mov    v8.16b, %21.16b                 \n"// v8 = _bias0
                "mov    v9.16b, %21.16b                 \n"// v9 = _bias0

                "st1    {v10.4s, v11.4s}, [%1], #32     \n"

                "bne    0b                              \n"
                "sub    %2, %2, #64                     \n"
                : "=r"(nn),         // %0
                  "=r"(outptr),     // %1
                  "=r"(r0),         // %2
                  "=r"(r1),         // %3
                  "=r"(r2),         // %4
                  "=r"(r3),         // %5
                  "=r"(r4)          // %6
                : "0"(nn),
                  "1"(outptr),
                  "2"(r0),
                  "3"(r1),
                  "4"(r2),
                  "5"(r3),
                  "6"(r4),
                  "w"(_k0123),      // %14
                  "w"(_k4567),      // %15
                  "w"(_k891011),    // %16
                  "w"(_k12131415),  // %17
                  "w"(_k16171819),  // %18
                  "w"(_k20212223),  // %19
                  "w"(_k24242424),  // %20
                  "w"(_bias0)       // %21
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27"
            );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                // r0
                "pld        [%2, #256]          \n"
                "vld2.f32   {d20-d23}, [%2]!    \n"// q10 q11 = r00 r01

                "vmov       q8, %q21            \n"

                "pld        [%2, #128]          \n"
                "vld2.f32   {d24-d25}, [%2]     \n"// q12 = r08 x x

                "0:                             \n"

                "vmul.f32   q9, q10, %e14[0]    \n"

                "vmov       d26, d25            \n"// q13 = r09 x x

                "vext.32    q14, q10, q12, #1   \n"// q14 = r02

                "vmla.f32   q8, q11, %e14[1]    \n"

                "vext.32    q15, q11, q13, #1   \n"// q15 = r03

                "vmla.f32   q9, q14, %f14[0]    \n"

                "vext.32    q14, q10, q12, #2   \n"// q14 = r04

                "vmla.f32   q8, q15, %f14[1]    \n"

                // r1
                "pld        [%3, #256]          \n"
                "vld2.f32   {d20-d23}, [%3]!    \n"// q10 q11 = r10 r11

                "vmla.f32   q9, q14, %e15[0]    \n"

                "pld        [%3, #128]          \n"
                "vld2.f32   {d24-d25}, [%3]     \n"// q12 = r18 x x

                "vmla.f32   q8, q10, %e15[1]    \n"

                "vmov       d26, d25            \n"// q13 = r19 x x

                "vext.32    q14, q10, q12, #1   \n"// q14 = r12

                "vmla.f32   q9, q11, %f15[0]    \n"

                "vext.32    q15, q11, q13, #1   \n"// q15 = r13

                "vmla.f32   q8, q14, %f15[1]    \n"

                "vext.32    q14, q10, q12, #2   \n"// q14 = r14

                "vmla.f32   q9, q15, %e16[0]    \n"

                // r2
                "pld        [%4, #256]          \n"
                "vld2.f32   {d20-d23}, [%4]!    \n"// q10 q11 = r20 r21

                "vmla.f32   q8, q14, %e16[1]    \n"

                "pld        [%4, #128]          \n"
                "vld2.f32   {d24-d25}, [%4]     \n"// q12 = r28 x x

                "vmla.f32   q9, q10, %f16[0]    \n"

                "vmov       d26, d25            \n"// q13 = r29 x x

                "vext.32    q14, q10, q12, #1   \n"// q14 = r22

                "vmla.f32   q8, q11, %f16[1]    \n"

                "vext.32    q15, q11, q13, #1   \n"// q15 = r23

                "vmla.f32   q9, q14, %e17[0]    \n"

                "vext.32    q14, q10, q12, #2   \n"// q14 = r24

                "vmla.f32   q8, q15, %e17[1]    \n"

                // r3
                "pld        [%5, #256]          \n"
                "vld2.f32   {d20-d23}, [%5]!    \n"// q10 q11 = r30 r31

                "vmla.f32   q9, q14, %f17[0]    \n"

                "pld        [%5, #128]          \n"
                "vld2.f32   {d24-d25}, [%5]     \n"// q12 = r38 x x

                "vmla.f32   q8, q10, %f17[1]    \n"

                "vmov       d26, d25            \n"// q13 = r39 x x

                "vext.32    q14, q10, q12, #1   \n"// q14 = r32

                "vmla.f32   q9, q11, %e18[0]    \n"

                "vext.32    q15, q11, q13, #1   \n"// q15 = r33

                "vmla.f32   q8, q14, %e18[1]    \n"

                "vext.32    q14, q10, q12, #2   \n"// q14 = r34

                "vmla.f32   q9, q15, %f18[0]    \n"

                // r4
                "pld        [%6, #256]          \n"
                "vld2.f32   {d20-d23}, [%6]!    \n"// q10 q11 = r40 r41

                "vmla.f32   q8, q14, %f18[1]    \n"

                "pld        [%6, #128]          \n"
                "vld2.f32   {d24-d25}, [%6]     \n"// q12 = r48 x x

                "vmla.f32   q9, q10, %e19[0]    \n"

                "vmov       d26, d25            \n"// q13 = r49 x x

                "vext.32    q14, q10, q12, #1   \n"// q14 = r42

                "vmla.f32   q8, q11, %e19[1]    \n"

                "vext.32    q15, q11, q13, #1   \n"// q15 = r43

                "vmla.f32   q9, q14, %f19[0]    \n"

                "vext.32    q14, q10, q12, #2   \n"// q14 = r44

                "vmla.f32   q8, q15, %f19[1]    \n"

                // r0
                "pld        [%2, #256]          \n"
                "vld2.f32   {d20-d23}, [%2]!    \n"// q10 q11 = r00 r01

                "vmla.f32   q9, q14, %e20[0]    \n"

                "pld        [%2, #128]          \n"
                "vld2.f32   {d24-d25}, [%2]     \n"// q12 = r08 x x

                "vadd.f32   q9, q8, q9          \n"

                "vmov       q8, %q21            \n"

                "subs       %0, #1              \n"

                "vst1.f32   {d18-d19}, [%1]!    \n"

                "bne        0b                  \n"
                "sub        %2, #32             \n"

                : "=r"(nn),         // %0
                  "=r"(outptr),     // %1
                  "=r"(r0),         // %2
                  "=r"(r1),         // %3
                  "=r"(r2),         // %4
                  "=r"(r3),         // %5
                  "=r"(r4)          // %6
                : "0"(nn),
                  "1"(outptr),
                  "2"(r0),
                  "3"(r1),
                  "4"(r2),
                  "5"(r3),
                  "6"(r4),
                  "w"(_k0123),      // %14
                  "w"(_k4567),      // %15
                  "w"(_k891011),    // %16
                  "w"(_k12131415),  // %17
                  "w"(_k16171819),  // %18
                  "w"(_k20212223),  // %19
                  "w"(_k24242424),  // %20
                  "w"(_bias0)       // %21
                : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                float sum = bias0;
#if __ARM_NEON
                // TODO neon assembly optimize
                float32x4_t _r0 = vld1q_f32(r0);
                float32x4_t _sum = vmulq_f32(_r0, _k0123);

                float32x4_t _r1 = vld1q_f32(r1);
                _sum = vmlaq_f32(_sum, _r1, vld1q_f32(k1));

                float32x4_t _r2 = vld1q_f32(r2);
                _sum = vmlaq_f32(_sum, _r2, vld1q_f32(k2));

                float32x4_t _r3 = vld1q_f32(r3);
                _sum = vmlaq_f32(_sum, _r3, vld1q_f32(k3));

                float32x4_t _r4 = vld1q_f32(r4);
                _sum = vmlaq_f32(_sum, _r4, _k20212223);

                sum += r0[4] * k0[4];
                sum += r1[4] * k1[4];
                sum += r2[4] * k2[4];
                sum += r3[4] * k3[4];
                sum += r4[4] * k4[4];

                float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                _ss = vpadd_f32(_ss, _ss);

                sum += vget_lane_f32(_ss, 0);
#else
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r0[3] * k0[3];
                sum += r0[4] * k0[4];

                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r1[3] * k1[3];
                sum += r1[4] * k1[4];

                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];
                sum += r2[3] * k2[3];
                sum += r2[4] * k2[4];

                sum += r3[0] * k3[0];
                sum += r3[1] * k3[1];
                sum += r3[2] * k3[2];
                sum += r3[3] * k3[3];
                sum += r3[4] * k3[4];

                sum += r4[0] * k4[0];
                sum += r4[1] * k4[1];
                sum += r4[2] * k4[2];
                sum += r4[3] * k4[3];
                sum += r4[4] * k4[4];
#endif
                *outptr = sum;

                r0 += 2;
                r1 += 2;
                r2 += 2;
                r3 += 2;
                r4 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
            r3 += tailstep;
            r4 += tailstep;
        }
    }

}
}
