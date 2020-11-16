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
static void conv3x3s1_winograd64_neon4_BdB(const Mat& bottom_blob, Mat& top_blob, const Option& opt,
        int outch, int inch, int outh, int outw)
{
    // BEGIN transform input
    int w = bottom_blob.w;
    //int h = bottom_blob.h;
    Mat bottom_blob_bordered = bottom_blob;
    Mat bottom_blob_tm = top_blob;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        bottom_blob_tm.create(4, 16 * w_tm/8 * h_tm/8, inch, 4u, opt.workspace_allocator);
        const int tiles = w_tm/8 * h_tm/8;

//         const float itm[8][8] = {
//             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
//
//             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
//             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
//
//             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
//             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
//
//             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
//             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
//
//             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
//         };

        // 0 = r00 - r06 + (r04 - r02) * 5.25
        // 7 = r07 - r01 + (r03 - r05) * 5.25

        // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
        // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

        // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
        // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

        // reuse r04 * 1.25
        // reuse r03 * 2.5
        // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
        // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

#if __ARM_NEON
        const float coeff[8] = {
            0.25f, 0.5f, -1.25f,   2.f,
            -2.5f,  4.f,  4.25f, 5.25f
        };
        float32x4_t _coeff0 = vld1q_f32(coeff);
        float32x4_t _coeff1 = vld1q_f32(coeff+4);
#endif // __ARM_NEON

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q<inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            float tmp[8][8];

            // tile
            for (int i=0; i<h_tm/8; i++)
            {
                for (int j=0; j<w_tm/8; j++)
                {
#if __ARM_NEON
                    const float* r0 = img0.row(i * 6) + j * 6;
                    const float* r1 = r0 + w;
                    const float* r2 = r0 + w*2;
                    const float* r3 = r0 + w*3;

                    // the assembly block for armv7 input transform requires 13 general registers
                    // old gcc may fail to allocate register on debug build without -fomit-frame-pointer
                    // so, fallback to intrinsic version for armv7 debug build     --- nihui
#if __aarch64__ || !defined(NDEBUG)
                    for (int m=0; m+3<8; m+=4)
                    {
                        float32x4_t _r0_0123 = vld1q_f32(r0);
                        float32x4_t _r0_4567 = vld1q_f32(r0+4);
                        float32x4_t _r1_0123 = vld1q_f32(r1);
                        float32x4_t _r1_4567 = vld1q_f32(r1+4);
                        float32x4_t _r2_0123 = vld1q_f32(r2);
                        float32x4_t _r2_4567 = vld1q_f32(r2+4);
                        float32x4_t _r3_0123 = vld1q_f32(r3);
                        float32x4_t _r3_4567 = vld1q_f32(r3+4);

                        float32x4x2_t _r01_00221133 = vtrnq_f32(_r0_0123, _r1_0123);
                        float32x4x2_t _r01_44665577 = vtrnq_f32(_r0_4567, _r1_4567);
                        float32x4x2_t _r23_00221133 = vtrnq_f32(_r2_0123, _r3_0123);
                        float32x4x2_t _r23_44665577 = vtrnq_f32(_r2_4567, _r3_4567);

                        // no vswp intrinsic  :(
                        float32x4_t _r_00 = vcombine_f32(vget_low_f32(_r01_00221133.val[0]), vget_low_f32(_r23_00221133.val[0]));
                        float32x4_t _r_11 = vcombine_f32(vget_low_f32(_r01_00221133.val[1]), vget_low_f32(_r23_00221133.val[1]));
                        float32x4_t _r_22 = vcombine_f32(vget_high_f32(_r01_00221133.val[0]), vget_high_f32(_r23_00221133.val[0]));
                        float32x4_t _r_33 = vcombine_f32(vget_high_f32(_r01_00221133.val[1]), vget_high_f32(_r23_00221133.val[1]));
                        float32x4_t _r_44 = vcombine_f32(vget_low_f32(_r01_44665577.val[0]), vget_low_f32(_r23_44665577.val[0]));
                        float32x4_t _r_55 = vcombine_f32(vget_low_f32(_r01_44665577.val[1]), vget_low_f32(_r23_44665577.val[1]));
                        float32x4_t _r_66 = vcombine_f32(vget_high_f32(_r01_44665577.val[0]), vget_high_f32(_r23_44665577.val[0]));
                        float32x4_t _r_77 = vcombine_f32(vget_high_f32(_r01_44665577.val[1]), vget_high_f32(_r23_44665577.val[1]));

                        float32x4_t _r_0_m_6 = vsubq_f32(_r_00, _r_66);
                        float32x4_t _r_7_m_1 = vsubq_f32(_r_77, _r_11);

                        float32x4_t _r_4_m_2 = vsubq_f32(_r_44, _r_22);
                        float32x4_t _r_3_m_5 = vsubq_f32(_r_33, _r_55);

                        float32x4_t _tmp0 = vmlaq_lane_f32(_r_0_m_6, _r_4_m_2, vget_high_f32(_coeff1), 1);
                        float32x4_t _tmp7 = vmlaq_lane_f32(_r_7_m_1, _r_3_m_5, vget_high_f32(_coeff1), 1);

                        vst1q_f32(&tmp[0][m], _tmp0);
                        vst1q_f32(&tmp[7][m], _tmp7);

                        float32x4_t _r_2_a_6 = vaddq_f32(_r_22, _r_66);
                        float32x4_t _r_1_a_5 = vaddq_f32(_r_11, _r_55);

                        float32x4_t _tmp12a = vmlsq_lane_f32(_r_2_a_6, _r_44, vget_high_f32(_coeff1), 0);
                        float32x4_t _tmp12b = vmlsq_lane_f32(_r_1_a_5, _r_33, vget_high_f32(_coeff1), 0);

                        float32x4_t _tmp1 = vaddq_f32(_tmp12a, _tmp12b);
                        float32x4_t _tmp2 = vsubq_f32(_tmp12a, _tmp12b);

                        vst1q_f32(&tmp[1][m], _tmp1);
                        vst1q_f32(&tmp[2][m], _tmp2);

                        float32x4_t _r_4_x_c = vmulq_lane_f32(_r_44, vget_high_f32(_coeff0), 0);
                        float32x4_t _r_3_x_c = vmulq_lane_f32(_r_33, vget_low_f32(_coeff1), 0);

                        float32x4_t _tmp34a = vaddq_f32(_r_66, _r_4_x_c);
                        _tmp34a = vmlaq_lane_f32(_tmp34a, _r_22, vget_low_f32(_coeff0), 0);

                        float32x4_t _tmp34b = vmlaq_lane_f32(_r_3_x_c, _r_11, vget_low_f32(_coeff0), 1);
                        _tmp34b = vmlaq_lane_f32(_tmp34b, _r_55, vget_high_f32(_coeff0), 1);

                        float32x4_t _tmp3 = vaddq_f32(_tmp34a, _tmp34b);
                        float32x4_t _tmp4 = vsubq_f32(_tmp34a, _tmp34b);

                        vst1q_f32(&tmp[3][m], _tmp3);
                        vst1q_f32(&tmp[4][m], _tmp4);

                        // reuse r04 * 1.25
                        // reuse r03 * 2.5
                        float32x4_t _r_2_a_4c = vaddq_f32(_r_22, _r_4_x_c);
                        float32x4_t _tmp56a = vmlaq_lane_f32(_r_66, _r_2_a_4c, vget_low_f32(_coeff1), 1);
                        float32x4_t _tmp56b = vmlaq_lane_f32(_r_3_x_c, _r_11, vget_high_f32(_coeff0), 1);
                        _tmp56b = vmlaq_lane_f32(_tmp56b, _r_55, vget_low_f32(_coeff0), 1);

                        float32x4_t _tmp5 = vaddq_f32(_tmp56a, _tmp56b);
                        float32x4_t _tmp6 = vsubq_f32(_tmp56a, _tmp56b);

                        vst1q_f32(&tmp[5][m], _tmp5);
                        vst1q_f32(&tmp[6][m], _tmp6);

                        r0 += w*4;
                        r1 += w*4;
                        r2 += w*4;
                        r3 += w*4;
                    }

                    const float* t0 = tmp[0];
                    const float* t1 = tmp[1];
                    const float* t2 = tmp[2];
                    const float* t3 = tmp[3];

                    float* r0_tm0_0 = img0_tm.row(i * w_tm/8 + j);
                    float* r0_tm0_4 = img0_tm.row(i * w_tm/8 + j + tiles);
                    float* r0_tm1_0 = img0_tm.row(i * w_tm/8 + j + tiles*2);
                    float* r0_tm1_4 = img0_tm.row(i * w_tm/8 + j + tiles*3);
                    float* r0_tm2_0 = img0_tm.row(i * w_tm/8 + j + tiles*4);
                    float* r0_tm2_4 = img0_tm.row(i * w_tm/8 + j + tiles*5);
                    float* r0_tm3_0 = img0_tm.row(i * w_tm/8 + j + tiles*6);
                    float* r0_tm3_4 = img0_tm.row(i * w_tm/8 + j + tiles*7);

                    for (int m=0; m+3<8; m+=4)
                    {
                        float32x4_t _t0_0123 = vld1q_f32(t0);
                        float32x4_t _t0_4567 = vld1q_f32(t0+4);
                        float32x4_t _t1_0123 = vld1q_f32(t1);
                        float32x4_t _t1_4567 = vld1q_f32(t1+4);
                        float32x4_t _t2_0123 = vld1q_f32(t2);
                        float32x4_t _t2_4567 = vld1q_f32(t2+4);
                        float32x4_t _t3_0123 = vld1q_f32(t3);
                        float32x4_t _t3_4567 = vld1q_f32(t3+4);

                        float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
                        float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);
                        float32x4x2_t _t23_00221133 = vtrnq_f32(_t2_0123, _t3_0123);
                        float32x4x2_t _t23_44665577 = vtrnq_f32(_t2_4567, _t3_4567);

                        // no vswp intrinsic  :(
                        float32x4_t _t_00 = vcombine_f32(vget_low_f32(_t01_00221133.val[0]), vget_low_f32(_t23_00221133.val[0]));
                        float32x4_t _t_11 = vcombine_f32(vget_low_f32(_t01_00221133.val[1]), vget_low_f32(_t23_00221133.val[1]));
                        float32x4_t _t_22 = vcombine_f32(vget_high_f32(_t01_00221133.val[0]), vget_high_f32(_t23_00221133.val[0]));
                        float32x4_t _t_33 = vcombine_f32(vget_high_f32(_t01_00221133.val[1]), vget_high_f32(_t23_00221133.val[1]));
                        float32x4_t _t_44 = vcombine_f32(vget_low_f32(_t01_44665577.val[0]), vget_low_f32(_t23_44665577.val[0]));
                        float32x4_t _t_55 = vcombine_f32(vget_low_f32(_t01_44665577.val[1]), vget_low_f32(_t23_44665577.val[1]));
                        float32x4_t _t_66 = vcombine_f32(vget_high_f32(_t01_44665577.val[0]), vget_high_f32(_t23_44665577.val[0]));
                        float32x4_t _t_77 = vcombine_f32(vget_high_f32(_t01_44665577.val[1]), vget_high_f32(_t23_44665577.val[1]));

                        float32x4_t _t_0_m_6 = vsubq_f32(_t_00, _t_66);
                        float32x4_t _t_7_m_1 = vsubq_f32(_t_77, _t_11);

                        float32x4_t _t_4_m_2 = vsubq_f32(_t_44, _t_22);
                        float32x4_t _t_3_m_5 = vsubq_f32(_t_33, _t_55);

                        float32x4_t _r0_tm_0_0 = vmlaq_lane_f32(_t_0_m_6, _t_4_m_2, vget_high_f32(_coeff1), 1);
                        float32x4_t _r0_tm_4_3 = vmlaq_lane_f32(_t_7_m_1, _t_3_m_5, vget_high_f32(_coeff1), 1);

                        r0_tm0_0[0] = vgetq_lane_f32(_r0_tm_0_0, 0);
                        r0_tm1_0[0] = vgetq_lane_f32(_r0_tm_0_0, 1);
                        r0_tm2_0[0] = vgetq_lane_f32(_r0_tm_0_0, 2);
                        r0_tm3_0[0] = vgetq_lane_f32(_r0_tm_0_0, 3);

                        r0_tm0_4[3] = vgetq_lane_f32(_r0_tm_4_3, 0);
                        r0_tm1_4[3] = vgetq_lane_f32(_r0_tm_4_3, 1);
                        r0_tm2_4[3] = vgetq_lane_f32(_r0_tm_4_3, 2);
                        r0_tm3_4[3] = vgetq_lane_f32(_r0_tm_4_3, 3);

                        float32x4_t _t_2_m_6 = vaddq_f32(_t_22, _t_66);
                        float32x4_t _t_1_m_5 = vaddq_f32(_t_11, _t_55);

                        float32x4_t _tmp12a = vmlsq_lane_f32(_t_2_m_6, _t_44, vget_high_f32(_coeff1), 0);
                        float32x4_t _tmp12b = vmlsq_lane_f32(_t_1_m_5, _t_33, vget_high_f32(_coeff1), 0);

                        float32x4_t _r0_tm_0_1 = vaddq_f32(_tmp12a, _tmp12b);
                        float32x4_t _r0_tm_0_2 = vsubq_f32(_tmp12a, _tmp12b);

                        r0_tm0_0[1] = vgetq_lane_f32(_r0_tm_0_1, 0);
                        r0_tm1_0[1] = vgetq_lane_f32(_r0_tm_0_1, 1);
                        r0_tm2_0[1] = vgetq_lane_f32(_r0_tm_0_1, 2);
                        r0_tm3_0[1] = vgetq_lane_f32(_r0_tm_0_1, 3);

                        r0_tm0_0[2] = vgetq_lane_f32(_r0_tm_0_2, 0);
                        r0_tm1_0[2] = vgetq_lane_f32(_r0_tm_0_2, 1);
                        r0_tm2_0[2] = vgetq_lane_f32(_r0_tm_0_2, 2);
                        r0_tm3_0[2] = vgetq_lane_f32(_r0_tm_0_2, 3);

                        float32x4_t _t_4_x_c = vmulq_lane_f32(_t_44, vget_high_f32(_coeff0), 0);
                        float32x4_t _t_3_x_c = vmulq_lane_f32(_t_33, vget_low_f32(_coeff1), 0);

                        float32x4_t _tmp34a = vaddq_f32(_t_66, _t_4_x_c);
                        _tmp34a = vmlaq_lane_f32(_tmp34a, _t_22, vget_low_f32(_coeff0), 0);

                        float32x4_t _tmp34b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_low_f32(_coeff0), 1);
                        _tmp34b = vmlaq_lane_f32(_tmp34b, _t_55, vget_high_f32(_coeff0), 1);

                        float32x4_t _r0_tm_0_3 = vaddq_f32(_tmp34a, _tmp34b);
                        float32x4_t _r0_tm_4_0 = vsubq_f32(_tmp34a, _tmp34b);

                        r0_tm0_0[3] = vgetq_lane_f32(_r0_tm_0_3, 0);
                        r0_tm1_0[3] = vgetq_lane_f32(_r0_tm_0_3, 1);
                        r0_tm2_0[3] = vgetq_lane_f32(_r0_tm_0_3, 2);
                        r0_tm3_0[3] = vgetq_lane_f32(_r0_tm_0_3, 3);

                        r0_tm0_4[0] = vgetq_lane_f32(_r0_tm_4_0, 0);
                        r0_tm1_4[0] = vgetq_lane_f32(_r0_tm_4_0, 1);
                        r0_tm2_4[0] = vgetq_lane_f32(_r0_tm_4_0, 2);
                        r0_tm3_4[0] = vgetq_lane_f32(_r0_tm_4_0, 3);

                        float32x4_t _t_2_a_4c = vaddq_f32(_t_22, _t_4_x_c);
                        float32x4_t _tmp56a = vmlaq_lane_f32(_t_66, _t_2_a_4c, vget_low_f32(_coeff1), 1);
                        float32x4_t _tmp56b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_high_f32(_coeff0), 1);
                        _tmp56b = vmlaq_lane_f32(_tmp56b, _t_55, vget_low_f32(_coeff0), 1);

                        float32x4_t _r0_tm_4_1 = vaddq_f32(_tmp56a, _tmp56b);
                        float32x4_t _r0_tm_4_2 = vsubq_f32(_tmp56a, _tmp56b);

                        r0_tm0_4[1] = vgetq_lane_f32(_r0_tm_4_1, 0);
                        r0_tm1_4[1] = vgetq_lane_f32(_r0_tm_4_1, 1);
                        r0_tm2_4[1] = vgetq_lane_f32(_r0_tm_4_1, 2);
                        r0_tm3_4[1] = vgetq_lane_f32(_r0_tm_4_1, 3);

                        r0_tm0_4[2] = vgetq_lane_f32(_r0_tm_4_2, 0);
                        r0_tm1_4[2] = vgetq_lane_f32(_r0_tm_4_2, 1);
                        r0_tm2_4[2] = vgetq_lane_f32(_r0_tm_4_2, 2);
                        r0_tm3_4[2] = vgetq_lane_f32(_r0_tm_4_2, 3);

                        t0 += 8*4;
                        t1 += 8*4;
                        t2 += 8*4;
                        t3 += 8*4;

                        r0_tm0_0 += img0_tm.w*tiles*2*4;
                        r0_tm0_4 += img0_tm.w*tiles*2*4;
                        r0_tm1_0 += img0_tm.w*tiles*2*4;
                        r0_tm1_4 += img0_tm.w*tiles*2*4;
                        r0_tm2_0 += img0_tm.w*tiles*2*4;
                        r0_tm2_4 += img0_tm.w*tiles*2*4;
                        r0_tm3_0 += img0_tm.w*tiles*2*4;
                        r0_tm3_4 += img0_tm.w*tiles*2*4;
                    }
#else // __aarch64__
                    float* t0 = tmp[0];
                    float* t1 = tmp[1];
                    float* t2 = tmp[2];
                    float* t3 = tmp[3];
                    float* t4 = tmp[4];
                    float* t5 = tmp[5];
                    float* t6 = tmp[6];
                    float* t7 = tmp[7];

                    int stepw = w*4*4;

                    asm volatile(

                        // loop0
                        "vld1.f32   {d16-d19}, [%8], %26    \n"
                        "vld1.f32   {d20-d23}, [%9], %26    \n"
                        "vld1.f32   {d24-d27}, [%10], %26   \n"

                        "vtrn.32    q8, q10             \n"

                        "vld1.f32   {d28-d31}, [%11], %26   \n"

                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vsub.f32   q2, q8, q13         \n"
                        "vsub.f32   q3, q9, q12         \n"

                        "vadd.f32   q4, q12, q13        \n"
                        "vadd.f32   q5, q10, q11        \n"

                        "vmla.f32   q2, q3, %f25[1]     \n"

                        "vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
                        "vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

                        "vmls.f32   q4, q9, %f25[0]     \n"
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        "vst1.f32   {d4-d5}, [%0]!      \n"// tmp[0][m]

                        "vmov       q3, q7              \n"// use q7

                        "vadd.f32   q2, q13, q6         \n"// use q6
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        "vadd.f32   q8, q4, q5          \n"
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n"// use q7

                        "vadd.f32   q6, q12, q6         \n"// use q6
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n"

                        "vmla.f32   q2, q12, %e24[0]    \n"
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        "vst1.f32   {d16-d17}, [%1]!    \n"// tmp[1][m]

                        "vmla.f32   q4, q6, %e25[1]     \n"
                        "vmla.f32   q5, q11, %e24[1]    \n"

                        "vst1.f32   {d18-d19}, [%2]!    \n"// tmp[2][m]

                        "vadd.f32   q8, q2, q3          \n"
                        "vsub.f32   q9, q2, q3          \n"

                        "vsub.f32   q6, q15, q10        \n"
                        "vsub.f32   q7, q14, q11        \n"

                        "vadd.f32   q2, q4, q5          \n"
                        "vsub.f32   q3, q4, q5          \n"

                        "vst1.f32   {d16-d17}, [%3]!    \n"// tmp[3][m]
                        "vst1.f32   {d18-d19}, [%4]!    \n"// tmp[4][m]

                        "vmla.f32   q6, q7, %f25[1]     \n"

                        "vst1.f32   {d4-d5}, [%5]!      \n"// tmp[5][m]
                        "vst1.f32   {d6-d7}, [%6]!      \n"// tmp[6][m]

                        "vst1.f32   {d12-d13}, [%7]!    \n"// tmp[7][m]

                        // loop1
                        "vld1.f32   {d16-d19}, [%8]     \n"
                        "vld1.f32   {d20-d23}, [%9]     \n"
                        "vld1.f32   {d24-d27}, [%10]    \n"

                        "vtrn.32    q8, q10             \n"

                        "vld1.f32   {d28-d31}, [%11]    \n"

                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vsub.f32   q2, q8, q13         \n"
                        "vsub.f32   q3, q9, q12         \n"

                        "vadd.f32   q4, q12, q13        \n"
                        "vadd.f32   q5, q10, q11        \n"

                        "vmla.f32   q2, q3, %f25[1]     \n"

                        "vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
                        "vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

                        "vmls.f32   q4, q9, %f25[0]     \n"
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        "vst1.f32   {d4-d5}, [%0]!      \n"// tmp[0][m]

                        "vmov       q3, q7              \n"// use q7

                        "vadd.f32   q2, q13, q6         \n"// use q6
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        "vadd.f32   q8, q4, q5          \n"
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n"// use q7

                        "vadd.f32   q6, q12, q6         \n"// use q6
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n"

                        "vmla.f32   q2, q12, %e24[0]    \n"
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        "vst1.f32   {d16-d17}, [%1]!    \n"// tmp[1][m]

                        "vmla.f32   q4, q6, %e25[1]     \n"
                        "vmla.f32   q5, q11, %e24[1]    \n"

                        "vst1.f32   {d18-d19}, [%2]!    \n"// tmp[2][m]

                        "vadd.f32   q8, q2, q3          \n"
                        "vsub.f32   q9, q2, q3          \n"

                        "vsub.f32   q6, q15, q10        \n"
                        "vsub.f32   q7, q14, q11        \n"

                        "vadd.f32   q2, q4, q5          \n"
                        "vsub.f32   q3, q4, q5          \n"

                        "vst1.f32   {d16-d17}, [%3]!    \n"// tmp[3][m]
                        "vst1.f32   {d18-d19}, [%4]!    \n"// tmp[4][m]

                        "vmla.f32   q6, q7, %f25[1]     \n"

                        "vst1.f32   {d4-d5}, [%5]!      \n"// tmp[5][m]
                        "vst1.f32   {d6-d7}, [%6]!      \n"// tmp[6][m]

                        "vst1.f32   {d12-d13}, [%7]!    \n"// tmp[7][m]

                        : "=r"(t0),     // %0
                          "=r"(t1),     // %1
                          "=r"(t2),     // %2
                          "=r"(t3),     // %3
                          "=r"(t4),     // %4
                          "=r"(t5),     // %5
                          "=r"(t6),     // %6
                          "=r"(t7),     // %7
                          "=r"(r0),     // %8
                          "=r"(r1),     // %9
                          "=r"(r2),     // %10
                          "=r"(r3)      // %11
                        : "0"(t0),
                          "1"(t1),
                          "2"(t2),
                          "3"(t3),
                          "4"(t4),
                          "5"(t5),
                          "6"(t6),
                          "7"(t7),
                          "8"(r0),
                          "9"(r1),
                          "10"(r2),
                          "11"(r3),
                          "w"(_coeff0), // %24
                          "w"(_coeff1), // %25
                          "r"(stepw)        // %26
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                    t0 = tmp[0];
                    t1 = tmp[1];
                    t2 = tmp[2];
                    t3 = tmp[3];

                    float* r0_tm0_0 = img0_tm.row(i * w_tm/8 + j);
                    float* r0_tm0_4 = img0_tm.row(i * w_tm/8 + j + tiles);
                    float* r0_tm1_0 = img0_tm.row(i * w_tm/8 + j + tiles*2);
                    float* r0_tm1_4 = img0_tm.row(i * w_tm/8 + j + tiles*3);
                    float* r0_tm2_0 = img0_tm.row(i * w_tm/8 + j + tiles*4);
                    float* r0_tm2_4 = img0_tm.row(i * w_tm/8 + j + tiles*5);
                    float* r0_tm3_0 = img0_tm.row(i * w_tm/8 + j + tiles*6);
                    float* r0_tm3_4 = img0_tm.row(i * w_tm/8 + j + tiles*7);

                    int step = img0_tm.w*tiles*2*4*4;

                    asm volatile(

                        // loop0
                        "vld1.f32   {d16-d19}, [%8]     \n"
                        "add        %8, %8, #128        \n"
                        "vld1.f32   {d20-d23}, [%9]     \n"
                        "add        %9, %9, #128        \n"
                        "vld1.f32   {d24-d27}, [%10]    \n"
                        "add        %10, %10, #128      \n"

                        "vtrn.32    q8, q10             \n"

                        "vld1.f32   {d28-d31}, [%11]    \n"
                        "add        %11, %11, #128      \n"

                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vsub.f32   q2, q8, q13         \n"
                        "vsub.f32   q3, q9, q12         \n"

                        "vadd.f32   q4, q12, q13        \n"
                        "vadd.f32   q5, q10, q11        \n"

                        "vmla.f32   q2, q3, %f25[1]     \n"

                        "vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
                        "vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

                        "vmls.f32   q4, q9, %f25[0]     \n"
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        "vst1.f32   {d4[0]}, [%0]!      \n"
                        "vst1.f32   {d4[1]}, [%2]!      \n"

                        "vmov       q3, q7              \n"// use q7

                        "vst1.f32   {d5[0]}, [%4]!      \n"
                        "vst1.f32   {d5[1]}, [%6]!      \n"

                        "vadd.f32   q2, q13, q6         \n"// use q6
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        "vadd.f32   q8, q4, q5          \n"
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n"// use q7

                        "vadd.f32   q6, q12, q6         \n"// use q6
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n"

                        "vmla.f32   q2, q12, %e24[0]    \n"
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        "vst1.f32   {d16[0]}, [%0]!     \n"
                        "vst1.f32   {d16[1]}, [%2]!     \n"

                        "vmla.f32   q4, q6, %e25[1]     \n"

                        "vst1.f32   {d17[0]}, [%4]!     \n"
                        "vst1.f32   {d17[1]}, [%6]!     \n"

                        "vmla.f32   q5, q11, %e24[1]    \n"

                        "vst1.f32   {d18[0]}, [%0]!     \n"
                        "vst1.f32   {d18[1]}, [%2]!     \n"

                        "vadd.f32   q8, q2, q3          \n"

                        "vst1.f32   {d19[0]}, [%4]!     \n"
                        "vst1.f32   {d19[1]}, [%6]!     \n"

                        "vsub.f32   q9, q2, q3          \n"

                        "vsub.f32   q6, q15, q10        \n"
                        "vsub.f32   q7, q14, q11        \n"

                        "vadd.f32   q2, q4, q5          \n"
                        "vsub.f32   q3, q4, q5          \n"

                        "vst1.f32   {d16[0]}, [%0], %26 \n"
                        "vst1.f32   {d16[1]}, [%2], %26 \n"

                        "vmla.f32   q6, q7, %f25[1]     \n"

                        "vst1.f32   {d17[0]}, [%4], %26 \n"
                        "vst1.f32   {d17[1]}, [%6], %26 \n"

                        "vtrn.32    q9, q2              \n"
                        "vtrn.32    q3, q6              \n"

                        "sub        %0, %0, #12         \n"
                        "sub        %2, %2, #12         \n"
                        "sub        %4, %4, #12         \n"
                        "sub        %6, %6, #12         \n"

                        "vswp       d19, d6             \n"
                        "vswp       d5, d12             \n"

                        "vst1.f32   {d18-d19}, [%1], %26 \n"
                        "vst1.f32   {d4-d5}, [%3], %26  \n"
                        "vst1.f32   {d6-d7}, [%5], %26  \n"
                        "vst1.f32   {d12-d13}, [%7], %26 \n"

                        // loop1
                        "vld1.f32   {d16-d19}, [%8]     \n"
                        "vld1.f32   {d20-d23}, [%9]     \n"
                        "vld1.f32   {d24-d27}, [%10]    \n"

                        "vtrn.32    q8, q10             \n"

                        "vld1.f32   {d28-d31}, [%11]    \n"

                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vsub.f32   q2, q8, q13         \n"
                        "vsub.f32   q3, q9, q12         \n"

                        "vadd.f32   q4, q12, q13        \n"
                        "vadd.f32   q5, q10, q11        \n"

                        "vmla.f32   q2, q3, %f25[1]     \n"

                        "vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
                        "vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

                        "vmls.f32   q4, q9, %f25[0]     \n"
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        "vst1.f32   {d4[0]}, [%0]!      \n"
                        "vst1.f32   {d4[1]}, [%2]!      \n"

                        "vmov       q3, q7              \n"// use q7

                        "vst1.f32   {d5[0]}, [%4]!      \n"
                        "vst1.f32   {d5[1]}, [%6]!      \n"

                        "vadd.f32   q2, q13, q6         \n"// use q6
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        "vadd.f32   q8, q4, q5          \n"
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n"// use q7

                        "vadd.f32   q6, q12, q6         \n"// use q6
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n"

                        "vmla.f32   q2, q12, %e24[0]    \n"
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        "vst1.f32   {d16[0]}, [%0]!     \n"
                        "vst1.f32   {d16[1]}, [%2]!     \n"

                        "vmla.f32   q4, q6, %e25[1]     \n"

                        "vst1.f32   {d17[0]}, [%4]!     \n"
                        "vst1.f32   {d17[1]}, [%6]!     \n"

                        "vmla.f32   q5, q11, %e24[1]    \n"

                        "vst1.f32   {d18[0]}, [%0]!     \n"
                        "vst1.f32   {d18[1]}, [%2]!     \n"

                        "vadd.f32   q8, q2, q3          \n"

                        "vst1.f32   {d19[0]}, [%4]!     \n"
                        "vst1.f32   {d19[1]}, [%6]!     \n"

                        "vsub.f32   q9, q2, q3          \n"

                        "vsub.f32   q6, q15, q10        \n"
                        "vsub.f32   q7, q14, q11        \n"

                        "vadd.f32   q2, q4, q5          \n"
                        "vsub.f32   q3, q4, q5          \n"

                        "vst1.f32   {d16[0]}, [%0]      \n"
                        "vst1.f32   {d16[1]}, [%2]      \n"

                        "vmla.f32   q6, q7, %f25[1]     \n"

                        "vst1.f32   {d17[0]}, [%4]      \n"
                        "vst1.f32   {d17[1]}, [%6]      \n"

                        "vtrn.32    q9, q2              \n"
                        "vtrn.32    q3, q6              \n"

                        "vswp       d19, d6             \n"
                        "vswp       d5, d12             \n"

                        "vst1.f32   {d18-d19}, [%1]     \n"
                        "vst1.f32   {d4-d5}, [%3]       \n"
                        "vst1.f32   {d6-d7}, [%5]       \n"
                        "vst1.f32   {d12-d13}, [%7]     \n"

                        : "=r"(r0_tm0_0),     // %0
                          "=r"(r0_tm0_4),     // %1
                          "=r"(r0_tm1_0),     // %2
                          "=r"(r0_tm1_4),     // %3
                          "=r"(r0_tm2_0),     // %4
                          "=r"(r0_tm2_4),     // %5
                          "=r"(r0_tm3_0),     // %6
                          "=r"(r0_tm3_4),     // %7
                          "=r"(t0),     // %8
                          "=r"(t1),     // %9
                          "=r"(t2),     // %10
                          "=r"(t3)      // %11
                        : "0"(r0_tm0_0),
                          "1"(r0_tm0_4),
                          "2"(r0_tm1_0),
                          "3"(r0_tm1_4),
                          "4"(r0_tm2_0),
                          "5"(r0_tm2_4),
                          "6"(r0_tm3_0),
                          "7"(r0_tm3_4),
                          "8"(t0),
                          "9"(t1),
                          "10"(t2),
                          "11"(t3),
                          "w"(_coeff0), // %24
                          "w"(_coeff1), // %25
                          "r"(step)        // %26
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    const float* r0 = img0.row(i * 6) + j * 6;

                    for (int m=0; m<8; m++)
                    {
                        tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25f;
                        tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25f;

                        float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25f);
                        float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25f);

                        tmp[1][m] = tmp12a + tmp12b;
                        tmp[2][m] = tmp12a - tmp12b;

                        float tmp34a = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                        float tmp34b = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);

                        tmp[3][m] = tmp34a + tmp34b;
                        tmp[4][m] = tmp34a - tmp34b;

                        float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
                        float tmp56b = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);

                        tmp[5][m] = tmp56a + tmp56b;
                        tmp[6][m] = tmp56a - tmp56b;

                        r0 += w;
                    }

                    float* r0_tm_0 = img0_tm.row(i * w_tm/8 + j);
                    float* r0_tm_4 = img0_tm.row(i * w_tm/8 + j + tiles);

                    for (int m=0; m<8; m++)
                    {
                        const float* tmp0 = tmp[m];

                        r0_tm_0[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25f;
                        r0_tm_4[3] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25f;

                        float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25f);
                        float tmp12b = (tmp0[1] - tmp0[3] * 4.25f + tmp0[5]);

                        r0_tm_0[1] = tmp12a + tmp12b;
                        r0_tm_0[2] = tmp12a - tmp12b;

                        float tmp34a = (tmp0[6] + tmp0[2] * 0.25f - tmp0[4] * 1.25f);
                        float tmp34b = (tmp0[1] * 0.5f - tmp0[3] * 2.5f + tmp0[5] * 2.f);

                        r0_tm_0[3] = tmp34a + tmp34b;
                        r0_tm_4[0] = tmp34a - tmp34b;

                        float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25f) * 4.f);
                        float tmp56b = (tmp0[1] * 2.f - tmp0[3] * 2.5f + tmp0[5] * 0.5f);

                        r0_tm_4[1] = tmp56a + tmp56b;
                        r0_tm_4[2] = tmp56a - tmp56b;

                        r0_tm_0 += img0_tm.w * tiles * 2;
                        r0_tm_4 += img0_tm.w * tiles * 2;
                    }
#endif // __ARM_NEON
                }
            }
        }
    }
}
}
