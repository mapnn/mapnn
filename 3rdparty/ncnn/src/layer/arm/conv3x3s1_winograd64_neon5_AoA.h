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
static void conv3x3s1_winograd64_neon5_AoA(const Mat& bottom_blob, Mat& top_blob, const Mat& _bias, const Option& opt,
        int inch, int outw, int outh, int outch)
{
    const float* bias = _bias;
    Mat top_blob_tm = bottom_blob;
    Mat top_blob_bordered = top_blob;
    {
//         const float otm[6][8] = {
//             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
//         };

        // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
        // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
        // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
        // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
        // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
        // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

#if __ARM_NEON
        const float coeff[4] = { 4.f, 8.f, 16.f, 32.f };
        float32x4_t _coeff = vld1q_f32(coeff);
#endif // __ARM_NEON

        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm/8 * h_tm/8;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p<outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            const float bias0 = bias ? bias[p] : 0.f;
#if __ARM_NEON
            float32x2_t _bias0 = vdup_n_f32(bias0);
#endif // __ARM_NEON

            float tmp[6][8];

            // tile
            for (int i=0; i<outh/6; i++)
            {
                for (int j=0; j<outw/6; j++)
                {
#if __ARM_NEON
#if __aarch64__
                    const float* output0_tm0 = out0_tm.row(i * w_tm/8 + j);
                    const float* output0_tm1 = out0_tm.row(i * w_tm/8 + j + tiles*8);
                    const float* output0_tm2 = out0_tm.row(i * w_tm/8 + j + tiles*16);
                    const float* output0_tm3 = out0_tm.row(i * w_tm/8 + j + tiles*24);

                    for (int m=0; m+3<8; m+=4)
                    {
                        float32x4_t _output0_tm_00;
                        float32x4_t _output0_tm_11;
                        float32x4_t _output0_tm_22;
                        float32x4_t _output0_tm_33;
                        float32x4_t _output0_tm_44;
                        float32x4_t _output0_tm_55;
                        float32x4_t _output0_tm_66;
                        float32x4_t _output0_tm_77;

                        _output0_tm_00 = vsetq_lane_f32(output0_tm0[0], _output0_tm_00, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_00 = vsetq_lane_f32(output0_tm1[0], _output0_tm_00, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_00 = vsetq_lane_f32(output0_tm2[0], _output0_tm_00, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_00 = vsetq_lane_f32(output0_tm3[0], _output0_tm_00, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_11 = vsetq_lane_f32(output0_tm0[0], _output0_tm_11, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_11 = vsetq_lane_f32(output0_tm1[0], _output0_tm_11, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_11 = vsetq_lane_f32(output0_tm2[0], _output0_tm_11, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_11 = vsetq_lane_f32(output0_tm3[0], _output0_tm_11, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_22 = vsetq_lane_f32(output0_tm0[0], _output0_tm_22, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_22 = vsetq_lane_f32(output0_tm1[0], _output0_tm_22, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_22 = vsetq_lane_f32(output0_tm2[0], _output0_tm_22, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_22 = vsetq_lane_f32(output0_tm3[0], _output0_tm_22, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_33 = vsetq_lane_f32(output0_tm0[0], _output0_tm_33, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_33 = vsetq_lane_f32(output0_tm1[0], _output0_tm_33, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_33 = vsetq_lane_f32(output0_tm2[0], _output0_tm_33, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_33 = vsetq_lane_f32(output0_tm3[0], _output0_tm_33, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_44 = vsetq_lane_f32(output0_tm0[0], _output0_tm_44, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_44 = vsetq_lane_f32(output0_tm1[0], _output0_tm_44, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_44 = vsetq_lane_f32(output0_tm2[0], _output0_tm_44, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_44 = vsetq_lane_f32(output0_tm3[0], _output0_tm_44, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_55 = vsetq_lane_f32(output0_tm0[0], _output0_tm_55, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_55 = vsetq_lane_f32(output0_tm1[0], _output0_tm_55, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_55 = vsetq_lane_f32(output0_tm2[0], _output0_tm_55, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_55 = vsetq_lane_f32(output0_tm3[0], _output0_tm_55, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_66 = vsetq_lane_f32(output0_tm0[0], _output0_tm_66, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_66 = vsetq_lane_f32(output0_tm1[0], _output0_tm_66, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_66 = vsetq_lane_f32(output0_tm2[0], _output0_tm_66, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_66 = vsetq_lane_f32(output0_tm3[0], _output0_tm_66, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_77 = vsetq_lane_f32(output0_tm0[0], _output0_tm_77, 0);
                        _output0_tm_77 = vsetq_lane_f32(output0_tm1[0], _output0_tm_77, 1);
                        _output0_tm_77 = vsetq_lane_f32(output0_tm2[0], _output0_tm_77, 2);
                        _output0_tm_77 = vsetq_lane_f32(output0_tm3[0], _output0_tm_77, 3);

                        float32x4_t _tmp024a = vaddq_f32(_output0_tm_11, _output0_tm_22);
                        float32x4_t _tmp135a = vsubq_f32(_output0_tm_11, _output0_tm_22);

                        float32x4_t _tmp024b = vaddq_f32(_output0_tm_33, _output0_tm_44);
                        float32x4_t _tmp135b = vsubq_f32(_output0_tm_33, _output0_tm_44);

                        float32x4_t _tmp024c = vaddq_f32(_output0_tm_55, _output0_tm_66);
                        float32x4_t _tmp135c = vsubq_f32(_output0_tm_55, _output0_tm_66);

                        float32x4_t _tmp0 = vaddq_f32(_output0_tm_00, _tmp024a);
                        _tmp0 = vmlaq_lane_f32(_tmp0, _tmp024c, vget_high_f32(_coeff), 1);
                        _tmp0 = vaddq_f32(_tmp0, _tmp024b);

                        float32x4_t _tmp2 = vmlaq_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                        _tmp2 = vmlaq_lane_f32(_tmp2, _tmp024c, vget_low_f32(_coeff), 1);

                        float32x4_t _tmp4 = vmlaq_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                        _tmp4 = vaddq_f32(_tmp4, _tmp024c);
                        _tmp4 = vaddq_f32(_tmp4, _tmp024c);

                        vst1q_f32(&tmp[0][m], _tmp0);
                        vst1q_f32(&tmp[2][m], _tmp2);
                        vst1q_f32(&tmp[4][m], _tmp4);

                        float32x4_t _tmp1 = vmlaq_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                        _tmp1 = vaddq_f32(_tmp1, _tmp135b);
                        _tmp1 = vaddq_f32(_tmp1, _tmp135b);

                        float32x4_t _tmp3 = vmlaq_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                        _tmp3 = vmlaq_lane_f32(_tmp3, _tmp135c, vget_low_f32(_coeff), 0);

                        float32x4_t _tmp5 = vaddq_f32(_output0_tm_77, _tmp135a);
                        _tmp5 = vmlaq_lane_f32(_tmp5, _tmp135b, vget_high_f32(_coeff), 1);
                        _tmp5 = vaddq_f32(_tmp5, _tmp135c);

                        vst1q_f32(&tmp[1][m], _tmp1);
                        vst1q_f32(&tmp[3][m], _tmp3);
                        vst1q_f32(&tmp[5][m], _tmp5);

                        output0_tm0 += out0_tm.w*tiles*25;
                        output0_tm1 += out0_tm.w*tiles*25;
                        output0_tm2 += out0_tm.w*tiles*25;
                        output0_tm3 += out0_tm.w*tiles*25;
                    }

                    const float* t0 = tmp[0];
                    const float* t1 = tmp[1];

                    float* output0 = out0.row(i * 6) + j * 6;
                    float* output1 = output0 + outw;

                    for (int m=0; m+1<6; m+=2)
                    {
                        float32x4_t _t0_0123 = vld1q_f32(t0);
                        float32x4_t _t0_4567 = vld1q_f32(t0+4);
                        float32x4_t _t1_0123 = vld1q_f32(t1);
                        float32x4_t _t1_4567 = vld1q_f32(t1+4);

                        float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
                        float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);

                        float32x2_t _t_00 = vget_low_f32(_t01_00221133.val[0]);
                        float32x2_t _t_11 = vget_low_f32(_t01_00221133.val[1]);
                        float32x2_t _t_22 = vget_high_f32(_t01_00221133.val[0]);
                        float32x2_t _t_33 = vget_high_f32(_t01_00221133.val[1]);
                        float32x2_t _t_44 = vget_low_f32(_t01_44665577.val[0]);
                        float32x2_t _t_55 = vget_low_f32(_t01_44665577.val[1]);
                        float32x2_t _t_66 = vget_high_f32(_t01_44665577.val[0]);
                        float32x2_t _t_77 = vget_high_f32(_t01_44665577.val[1]);

                        float32x2_t _tmp024a = vadd_f32(_t_11, _t_22);
                        float32x2_t _tmp135a = vsub_f32(_t_11, _t_22);

                        float32x2_t _tmp024b = vadd_f32(_t_33, _t_44);
                        float32x2_t _tmp135b = vsub_f32(_t_33, _t_44);

                        float32x2_t _tmp024c = vadd_f32(_t_55, _t_66);
                        float32x2_t _tmp135c = vsub_f32(_t_55, _t_66);

                        float32x2_t _output_0 = vadd_f32(_t_00, _tmp024a);
                        _output_0 = vmla_lane_f32(_output_0, _tmp024c, vget_high_f32(_coeff), 1);
                        _output_0 = vadd_f32(_output_0, _tmp024b);
                        _output_0 = vadd_f32(_output_0, _bias0);

                        float32x2_t _output_2 = vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                        _output_2 = vmla_lane_f32(_output_2, _tmp024c, vget_low_f32(_coeff), 1);
                        _output_2 = vadd_f32(_output_2, _bias0);

                        float32x2_t _output_4 = vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                        _output_4 = vadd_f32(_output_4, _tmp024c);
                        _output_4 = vadd_f32(_output_4, _tmp024c);
                        _output_4 = vadd_f32(_output_4, _bias0);

                        output0[0] = vget_lane_f32(_output_0, 0);
                        output1[0] = vget_lane_f32(_output_0, 1);
                        output0[2] = vget_lane_f32(_output_2, 0);
                        output1[2] = vget_lane_f32(_output_2, 1);
                        output0[4] = vget_lane_f32(_output_4, 0);
                        output1[4] = vget_lane_f32(_output_4, 1);

                        float32x2_t _output_1 = vmla_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                        _output_1 = vadd_f32(_output_1, _tmp135b);
                        _output_1 = vadd_f32(_output_1, _tmp135b);
                        _output_1 = vadd_f32(_output_1, _bias0);

                        float32x2_t _output_3 = vmla_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                        _output_3 = vmla_lane_f32(_output_3, _tmp135c, vget_low_f32(_coeff), 0);
                        _output_3 = vadd_f32(_output_3, _bias0);

                        float32x2_t _output_5 = vadd_f32(_t_77, _tmp135a);
                        _output_5 = vmla_lane_f32(_output_5, _tmp135b, vget_high_f32(_coeff), 1);
                        _output_5 = vadd_f32(_output_5, _tmp135c);
                        _output_5 = vadd_f32(_output_5, _bias0);

                        output0[1] = vget_lane_f32(_output_1, 0);
                        output1[1] = vget_lane_f32(_output_1, 1);
                        output0[3] = vget_lane_f32(_output_3, 0);
                        output1[3] = vget_lane_f32(_output_3, 1);
                        output0[5] = vget_lane_f32(_output_5, 0);
                        output1[5] = vget_lane_f32(_output_5, 1);

                        t0 += 8*2;
                        t1 += 8*2;
                        output0 += outw*2;
                        output1 += outw*2;
                    }
#else // __aarch64__
                    const float* output0_tm0_0 = out0_tm.row(i * w_tm/8 + j);
                    const float* output0_tm1_0 = out0_tm.row(i * w_tm/8 + j + tiles*8);
                    const float* output0_tm2_0 = out0_tm.row(i * w_tm/8 + j + tiles*16);
                    const float* output0_tm3_0 = out0_tm.row(i * w_tm/8 + j + tiles*24);
                    const float* output0_tm0_4 = out0_tm.row(i * w_tm/8 + j + tiles*32);
                    const float* output0_tm1_4 = out0_tm.row(i * w_tm/8 + j + tiles*40);
                    const float* output0_tm2_4 = out0_tm.row(i * w_tm/8 + j + tiles*48);
                    const float* output0_tm3_4 = out0_tm.row(i * w_tm/8 + j + tiles*56);

                    float* t0 = tmp[0];
                    float* t1 = tmp[1];

//                     int step = out0_tm.w * tiles * 2*4 *4;
                    int step = out0_tm.w * tiles *4;

                    asm volatile(

                        // loop0
//                         "vld1.f32   {d16-d17}, [%2], %21 \n"
//                         "vld1.f32   {d18-d19}, [%3], %21 \n"
//                         "vld1.f32   {d20-d21}, [%4], %21 \n"
//                         "vld1.f32   {d22-d23}, [%5], %21 \n"
//                         "vld1.f32   {d24-d25}, [%6], %21 \n"
//                         "vld1.f32   {d26-d27}, [%7], %21 \n"
//                         "vld1.f32   {d28-d29}, [%8], %21 \n"
//                         "vld1.f32   {d30-d31}, [%9], %21 \n"

//                         "vtrn.32    q8, q10             \n"
//                         "vtrn.32    q9, q11             \n"
//                         "vtrn.32    q12, q14            \n"
//                         "vtrn.32    q13, q15            \n"

//                         "vswp       d17, d24            \n"
//                         "vswp       d19, d26            \n"
//                         "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
//                         "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77
                        "vld1.f32   {d16[0]}, [%2], %21 \n"
                        "vld1.f32   {d16[1]}, [%3], %21 \n"
                        "vld1.f32   {d17[0]}, [%4], %21 \n"
                        "vld1.f32   {d17[1]}, [%5], %21 \n"

                        "vld1.f32   {d20[0]}, [%2], %21 \n"
                        "vld1.f32   {d20[1]}, [%3], %21 \n"
                        "vld1.f32   {d21[0]}, [%4], %21 \n"
                        "vld1.f32   {d21[1]}, [%5], %21 \n"

                        "vld1.f32   {d24[0]}, [%2], %21 \n"
                        "vld1.f32   {d24[1]}, [%3], %21 \n"
                        "vld1.f32   {d25[0]}, [%4], %21 \n"
                        "vld1.f32   {d25[1]}, [%5], %21 \n"

                        "vadd.f32   q2, q10, q12        \n"
                        "vsub.f32   q3, q10, q12        \n"

                        "vld1.f32   {d28[0]}, [%2], %21 \n"
                        "vld1.f32   {d28[1]}, [%3], %21 \n"
                        "vld1.f32   {d29[0]}, [%4], %21 \n"
                        "vld1.f32   {d29[1]}, [%5], %21 \n"

                        "vld1.f32   {d18[0]}, [%2], %21 \n"
                        "vld1.f32   {d18[1]}, [%3], %21 \n"
                        "vld1.f32   {d19[0]}, [%4], %21 \n"
                        "vld1.f32   {d19[1]}, [%5], %21 \n"

                        "vadd.f32   q4, q14, q9         \n"
                        "vsub.f32   q5, q14, q9         \n"

                        "vld1.f32   {d22[0]}, [%2], %21 \n"
                        "vld1.f32   {d22[1]}, [%3], %21 \n"
                        "vld1.f32   {d23[0]}, [%4], %21 \n"
                        "vld1.f32   {d23[1]}, [%5], %21 \n"

                        "vld1.f32   {d26[0]}, [%2], %21 \n"
                        "vld1.f32   {d26[1]}, [%3], %21 \n"
                        "vld1.f32   {d27[0]}, [%4], %21 \n"
                        "vld1.f32   {d27[1]}, [%5], %21 \n"

                        "vadd.f32   q6, q11, q13        \n"
                        "vsub.f32   q7, q11, q13        \n"// spare q9 q10 q11 q12 q13 q14

                        "vld1.f32   {d30[0]}, [%2]      \n"
                        "vld1.f32   {d30[1]}, [%3]      \n"
                        "vld1.f32   {d31[0]}, [%4]      \n"
                        "vld1.f32   {d31[1]}, [%5]      \n"

                        "vmov       q9, q3              \n"
                        "vadd.f32   q8, q8, q2          \n"
                        "vmla.f32   q9, q7, %f20[0]     \n"
                        "vmov       q12, q2             \n"
                        "vmov       q10, q2             \n"
                        "vmov       q11, q3             \n"
                        "vmla.f32   q12, q4, %f20[0]    \n"
                        "vadd.f32   q15, q15, q3        \n"
                        "vmla.f32   q8, q6, %f20[1]     \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vmla.f32   q10, q4, %e20[0]    \n"
                        "vmla.f32   q11, q5, %e20[1]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vmla.f32   q15, q5, %f20[1]    \n"
                        "vadd.f32   q8, q8, q4          \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vmla.f32   q10, q6, %e20[1]    \n"
                        "vmla.f32   q11, q7, %e20[0]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vadd.f32   q15, q15, q7        \n"

                        "vst1.f32   {d16-d17}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d18-d19}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d20-d21}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d22-d23}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d24-d25}, [%0]     \n"
                        "sub        %0, %0, #112        \n"

                        "vst1.f32   {d30-d31}, [%1]     \n"
                        "sub        %1, %1, #112        \n"

                        // loop1
//                         "vld1.f32   {d16-d17}, [%2]     \n"
//                         "vld1.f32   {d18-d19}, [%3]     \n"
//                         "vld1.f32   {d20-d21}, [%4]     \n"
//                         "vld1.f32   {d22-d23}, [%5]     \n"
//                         "vld1.f32   {d24-d25}, [%6]     \n"
//                         "vld1.f32   {d26-d27}, [%7]     \n"
//                         "vld1.f32   {d28-d29}, [%8]     \n"
//                         "vld1.f32   {d30-d31}, [%9]     \n"

//                         "vtrn.32    q8, q10             \n"
//                         "vtrn.32    q9, q11             \n"
//                         "vtrn.32    q12, q14            \n"
//                         "vtrn.32    q13, q15            \n"

//                         "vswp       d17, d24            \n"
//                         "vswp       d19, d26            \n"
//                         "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
//                         "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77
                        "vld1.f32   {d16[0]}, [%6], %21 \n"
                        "vld1.f32   {d16[1]}, [%7], %21 \n"
                        "vld1.f32   {d17[0]}, [%8], %21 \n"
                        "vld1.f32   {d17[1]}, [%9], %21 \n"

                        "vld1.f32   {d20[0]}, [%6], %21 \n"
                        "vld1.f32   {d20[1]}, [%7], %21 \n"
                        "vld1.f32   {d21[0]}, [%8], %21 \n"
                        "vld1.f32   {d21[1]}, [%9], %21 \n"

                        "vld1.f32   {d24[0]}, [%6], %21 \n"
                        "vld1.f32   {d24[1]}, [%7], %21 \n"
                        "vld1.f32   {d25[0]}, [%8], %21 \n"
                        "vld1.f32   {d25[1]}, [%9], %21 \n"

                        "vadd.f32   q2, q10, q12        \n"
                        "vsub.f32   q3, q10, q12        \n"

                        "vld1.f32   {d28[0]}, [%6], %21 \n"
                        "vld1.f32   {d28[1]}, [%7], %21 \n"
                        "vld1.f32   {d29[0]}, [%8], %21 \n"
                        "vld1.f32   {d29[1]}, [%9], %21 \n"

                        "vld1.f32   {d18[0]}, [%6], %21 \n"
                        "vld1.f32   {d18[1]}, [%7], %21 \n"
                        "vld1.f32   {d19[0]}, [%8], %21 \n"
                        "vld1.f32   {d19[1]}, [%9], %21 \n"

                        "vadd.f32   q4, q14, q9         \n"
                        "vsub.f32   q5, q14, q9         \n"

                        "vld1.f32   {d22[0]}, [%6], %21 \n"
                        "vld1.f32   {d22[1]}, [%7], %21 \n"
                        "vld1.f32   {d23[0]}, [%8], %21 \n"
                        "vld1.f32   {d23[1]}, [%9], %21 \n"

                        "vld1.f32   {d26[0]}, [%6], %21 \n"
                        "vld1.f32   {d26[1]}, [%7], %21 \n"
                        "vld1.f32   {d27[0]}, [%8], %21 \n"
                        "vld1.f32   {d27[1]}, [%9], %21 \n"

                        "vadd.f32   q6, q11, q13        \n"
                        "vsub.f32   q7, q11, q13        \n"// spare q9 q10 q11 q12 q13 q14

                        "vld1.f32   {d30[0]}, [%6]      \n"
                        "vld1.f32   {d30[1]}, [%7]      \n"
                        "vld1.f32   {d31[0]}, [%8]      \n"
                        "vld1.f32   {d31[1]}, [%9]      \n"

                        "vmov       q9, q3              \n"
                        "vadd.f32   q8, q8, q2          \n"
                        "vmla.f32   q9, q7, %f20[0]     \n"
                        "vmov       q12, q2             \n"
                        "vmov       q10, q2             \n"
                        "vmov       q11, q3             \n"
                        "vmla.f32   q12, q4, %f20[0]    \n"
                        "vadd.f32   q15, q15, q3        \n"
                        "vmla.f32   q8, q6, %f20[1]     \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vmla.f32   q10, q4, %e20[0]    \n"
                        "vmla.f32   q11, q5, %e20[1]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vmla.f32   q15, q5, %f20[1]    \n"
                        "vadd.f32   q8, q8, q4          \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vmla.f32   q10, q6, %e20[1]    \n"
                        "vmla.f32   q11, q7, %e20[0]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vadd.f32   q15, q15, q7        \n"

                        "vst1.f32   {d16-d17}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d18-d19}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d20-d21}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d22-d23}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d24-d25}, [%0]     \n"

                        "vst1.f32   {d30-d31}, [%1]     \n"

                        : "=r"(t0),             // %0
                          "=r"(t1),             // %1
                          "=r"(output0_tm0_0),  // %2
                          "=r"(output0_tm1_0),  // %3
                          "=r"(output0_tm2_0),  // %4
                          "=r"(output0_tm3_0),  // %5
                          "=r"(output0_tm0_4),  // %6
                          "=r"(output0_tm1_4),  // %7
                          "=r"(output0_tm2_4),  // %8
                          "=r"(output0_tm3_4)   // %9
                        : "0"(t0),
                          "1"(t1),
                          "2"(output0_tm0_0),
                          "3"(output0_tm1_0),
                          "4"(output0_tm2_0),
                          "5"(output0_tm3_0),
                          "6"(output0_tm0_4),
                          "7"(output0_tm1_4),
                          "8"(output0_tm2_4),
                          "9"(output0_tm3_4),
                          "w"(_coeff),          // %20
                          "r"(step)             // %21
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                    t0 = tmp[0];
                    t1 = tmp[1];

                    float* output0 = out0.row(i * 6) + j * 6;
                    float* output1 = output0 + outw;

                    int stepw = outw*2 * 4;

                    asm volatile(

                        // loop0
                        "vld1.f32   {d16-d19}, [%2]     \n"
                        "vld1.f32   {d20-d23}, [%3]     \n"

                        "add        %2, %2, #64         \n"
                        "add        %3, %3, #64         \n"

                        "vtrn.32    q8, q10             \n"// q8 = 0 2  q10 = 1 3
                        "vtrn.32    q9, q11             \n"// q9 = 4 6  q11 = 5 7

                        "vadd.f32   d4, d20, d17        \n"
                        "vsub.f32   d5, d20, d17        \n"

                        "vadd.f32   d6, d21, d18        \n"
                        "vsub.f32   d7, d21, d18        \n"

                        "vadd.f32   d8, d22, d19        \n"
                        "vsub.f32   d9, d22, d19        \n"// spare d17 ~ d22

                        "vmov       d20, d5             \n"
                        "vmov       d18, d4             \n"

                        "vadd.f32   d16, d16, d4        \n"
                        "vmla.f32   d20, d9, %f8[0]     \n"
                        "vmov       d17, d4             \n"
                        "vmov       d21, d5             \n"
                        "vmla.f32   d18, d6, %f8[0]     \n"
                        "vadd.f32   d22, d23, d5        \n"

                        "vmla.f32   d16, d8, %f8[1]     \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d6, %e8[0]     \n"
                        "vmla.f32   d21, d7, %e8[1]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vmla.f32   d22, d7, %f8[1]     \n"

                        "vadd.f32   d16, d16, d6        \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d8, %e8[1]     \n"
                        "vmla.f32   d21, d9, %e8[0]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vadd.f32   d22, d22, d9        \n"

                        "vadd.f32   d16, d16, %P9       \n"// _bias0
                        "vadd.f32   d20, d20, %P9       \n"// _bias0
                        "vadd.f32   d17, d17, %P9       \n"// _bias0
                        "vadd.f32   d21, d21, %P9       \n"// _bias0
                        "vadd.f32   d18, d18, %P9       \n"// _bias0
                        "vadd.f32   d22, d22, %P9       \n"// _bias0

                        "vtrn.f32   q8, q10             \n"
                        "vtrn.f32   d18, d22            \n"

                        "vst1.f32   {d16-d18}, [%0], %10 \n"
                        "vst1.f32   {d20-d22}, [%1], %10 \n"

                        // loop1
                        "vld1.f32   {d16-d19}, [%2]     \n"
                        "vld1.f32   {d20-d23}, [%3]     \n"

                        "add        %2, %2, #64         \n"
                        "add        %3, %3, #64         \n"

                        "vtrn.32    q8, q10             \n"// q8 = 0 2  q10 = 1 3
                        "vtrn.32    q9, q11             \n"// q9 = 4 6  q11 = 5 7

                        "vadd.f32   d4, d20, d17        \n"
                        "vsub.f32   d5, d20, d17        \n"

                        "vadd.f32   d6, d21, d18        \n"
                        "vsub.f32   d7, d21, d18        \n"

                        "vadd.f32   d8, d22, d19        \n"
                        "vsub.f32   d9, d22, d19        \n"// spare d17 ~ d22

                        "vmov       d20, d5             \n"
                        "vmov       d18, d4             \n"

                        "vadd.f32   d16, d16, d4        \n"
                        "vmla.f32   d20, d9, %f8[0]     \n"
                        "vmov       d17, d4             \n"
                        "vmov       d21, d5             \n"
                        "vmla.f32   d18, d6, %f8[0]     \n"
                        "vadd.f32   d22, d23, d5        \n"

                        "vmla.f32   d16, d8, %f8[1]     \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d6, %e8[0]     \n"
                        "vmla.f32   d21, d7, %e8[1]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vmla.f32   d22, d7, %f8[1]     \n"

                        "vadd.f32   d16, d16, d6        \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d8, %e8[1]     \n"
                        "vmla.f32   d21, d9, %e8[0]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vadd.f32   d22, d22, d9        \n"

                        "vadd.f32   d16, d16, %P9       \n"// _bias0
                        "vadd.f32   d20, d20, %P9       \n"// _bias0
                        "vadd.f32   d17, d17, %P9       \n"// _bias0
                        "vadd.f32   d21, d21, %P9       \n"// _bias0
                        "vadd.f32   d18, d18, %P9       \n"// _bias0
                        "vadd.f32   d22, d22, %P9       \n"// _bias0

                        "vtrn.f32   q8, q10             \n"
                        "vtrn.f32   d18, d22            \n"

                        "vst1.f32   {d16-d18}, [%0], %10 \n"
                        "vst1.f32   {d20-d22}, [%1], %10 \n"

                        // loop2
                        "vld1.f32   {d16-d19}, [%2]     \n"
                        "vld1.f32   {d20-d23}, [%3]     \n"

                        "add        %2, %2, #64         \n"
                        "add        %3, %3, #64         \n"

                        "vtrn.32    q8, q10             \n"// q8 = 0 2  q10 = 1 3
                        "vtrn.32    q9, q11             \n"// q9 = 4 6  q11 = 5 7

                        "vadd.f32   d4, d20, d17        \n"
                        "vsub.f32   d5, d20, d17        \n"

                        "vadd.f32   d6, d21, d18        \n"
                        "vsub.f32   d7, d21, d18        \n"

                        "vadd.f32   d8, d22, d19        \n"
                        "vsub.f32   d9, d22, d19        \n"// spare d17 ~ d22

                        "vmov       d20, d5             \n"
                        "vmov       d18, d4             \n"

                        "vadd.f32   d16, d16, d4        \n"
                        "vmla.f32   d20, d9, %f8[0]     \n"
                        "vmov       d17, d4             \n"
                        "vmov       d21, d5             \n"
                        "vmla.f32   d18, d6, %f8[0]     \n"
                        "vadd.f32   d22, d23, d5        \n"

                        "vmla.f32   d16, d8, %f8[1]     \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d6, %e8[0]     \n"
                        "vmla.f32   d21, d7, %e8[1]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vmla.f32   d22, d7, %f8[1]     \n"

                        "vadd.f32   d16, d16, d6        \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d8, %e8[1]     \n"
                        "vmla.f32   d21, d9, %e8[0]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vadd.f32   d22, d22, d9        \n"

                        "vadd.f32   d16, d16, %P9       \n"// _bias0
                        "vadd.f32   d20, d20, %P9       \n"// _bias0
                        "vadd.f32   d17, d17, %P9       \n"// _bias0
                        "vadd.f32   d21, d21, %P9       \n"// _bias0
                        "vadd.f32   d18, d18, %P9       \n"// _bias0
                        "vadd.f32   d22, d22, %P9       \n"// _bias0

                        "vtrn.f32   q8, q10             \n"
                        "vtrn.f32   d18, d22            \n"

                        "vst1.f32   {d16-d18}, [%0], %10 \n"
                        "vst1.f32   {d20-d22}, [%1], %10 \n"

                        : "=r"(output0),    // %0
                          "=r"(output1),    // %1
                          "=r"(t0),         // %2
                          "=r"(t1)          // %3
                        : "0"(output0),
                          "1"(output1),
                          "2"(t0),
                          "3"(t1),
                          "w"(_coeff),      // %8
                          "w"(_bias0),      // %9
                          "r"(stepw)        // %10
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    const float* output0_tm_0 = out0_tm.row(i * w_tm/8 + j);
                    const float* output0_tm_1 = out0_tm.row(i * w_tm/8 + j + tiles);
                    const float* output0_tm_2 = out0_tm.row(i * w_tm/8 + j + tiles*2);
                    const float* output0_tm_3 = out0_tm.row(i * w_tm/8 + j + tiles*3);
                    const float* output0_tm_4 = out0_tm.row(i * w_tm/8 + j + tiles*4);
                    const float* output0_tm_5 = out0_tm.row(i * w_tm/8 + j + tiles*5);
                    const float* output0_tm_6 = out0_tm.row(i * w_tm/8 + j + tiles*6);
                    const float* output0_tm_7 = out0_tm.row(i * w_tm/8 + j + tiles*7);

                    for (int m=0; m<8; m++)
                    {
                        float tmp024a = output0_tm_1[0] + output0_tm_2[0];
                        float tmp135a = output0_tm_1[0] - output0_tm_2[0];

                        float tmp024b = output0_tm_3[0] + output0_tm_4[0];
                        float tmp135b = output0_tm_3[0] - output0_tm_4[0];

                        float tmp024c = output0_tm_5[0] + output0_tm_6[0];
                        float tmp135c = output0_tm_5[0] - output0_tm_6[0];

                        tmp[0][m] = output0_tm_0[0] + tmp024a + tmp024b + tmp024c * 32;
                        tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                        tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                        tmp[5][m] = output0_tm_7[0] + tmp135a + tmp135b * 32 + tmp135c;

                        output0_tm_0 += out0_tm.w * tiles * 8;
                        output0_tm_1 += out0_tm.w * tiles * 8;
                        output0_tm_2 += out0_tm.w * tiles * 8;
                        output0_tm_3 += out0_tm.w * tiles * 8;
                        output0_tm_4 += out0_tm.w * tiles * 8;
                        output0_tm_5 += out0_tm.w * tiles * 8;
                        output0_tm_6 += out0_tm.w * tiles * 8;
                        output0_tm_7 += out0_tm.w * tiles * 8;
                    }

                    float* output0 = out0.row(i * 6) + j * 6;

                    for (int m=0; m<6; m++)
                    {
                        const float* tmp0 = tmp[m];

                        float tmp024a = tmp0[1] + tmp0[2];
                        float tmp135a = tmp0[1] - tmp0[2];

                        float tmp024b = tmp0[3] + tmp0[4];
                        float tmp135b = tmp0[3] - tmp0[4];

                        float tmp024c = tmp0[5] + tmp0[6];
                        float tmp135c = tmp0[5] - tmp0[6];

                        output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                        output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                        output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                        output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0 += outw;
                    }
#endif // __ARM_NEON
                }
            }
        }
    }

}
}
