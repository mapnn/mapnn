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
static void conv3x3s1_winograd64_pack4_neon_AoA(const Mat& bottom_blob, Mat& top_blob, const Mat& _bias, const Option& opt,
        int outch, int inch, int outh, int outw)
{
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    const float* bias = _bias;
    Mat top_blob_tm = bottom_blob;
    Mat top_blob_bordered = top_blob;
    top_blob_bordered.create(outw, outh, outch, elemsize, elempack, opt.workspace_allocator);
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

        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm/8 * h_tm/8;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p<outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

//             const float bias0 = bias ? bias[p] : 0.f;
            float32x4_t _bias0 = bias ? vld1q_f32( (const float*)bias + p * 4) : vdupq_n_f32(0.f);

            float tmp[6][8][4];

            // tile
            for (int i=0; i<outh/6; i++)
            {
                for (int j=0; j<outw/6; j++)
                {
//                     top_blob_tm.create(tiles, 64, outch, elemsize, elempack);

                    const float* output0_tm_0 = (const float*)out0_tm + (i * w_tm/8 + j) * 4;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 4;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 8;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 12;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 16;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 20;
                    const float* output0_tm_6 = output0_tm_0 + tiles * 24;
                    const float* output0_tm_7 = output0_tm_0 + tiles * 28;

                    float* output0 = out0.row(i * 6) + (j * 6) * 4;

                    // TODO neon optimize
                    for (int m=0; m<8; m++)
                    {
                        float32x4_t _out0tm0 = vld1q_f32(output0_tm_0);
                        float32x4_t _out0tm1 = vld1q_f32(output0_tm_1);
                        float32x4_t _out0tm2 = vld1q_f32(output0_tm_2);
                        float32x4_t _out0tm3 = vld1q_f32(output0_tm_3);
                        float32x4_t _out0tm4 = vld1q_f32(output0_tm_4);
                        float32x4_t _out0tm5 = vld1q_f32(output0_tm_5);
                        float32x4_t _out0tm6 = vld1q_f32(output0_tm_6);
                        float32x4_t _out0tm7 = vld1q_f32(output0_tm_7);

                        float32x4_t _tmp024a = vaddq_f32(_out0tm1, _out0tm2);
                        float32x4_t _tmp135a = vsubq_f32(_out0tm1, _out0tm2);

//                         float tmp024a = output0_tm[1] + output0_tm[2];
//                         float tmp135a = output0_tm[1] - output0_tm[2];

                        float32x4_t _tmp024b = vaddq_f32(_out0tm3, _out0tm4);
                        float32x4_t _tmp135b = vsubq_f32(_out0tm3, _out0tm4);

//                         float tmp024b = output0_tm[3] + output0_tm[4];
//                         float tmp135b = output0_tm[3] - output0_tm[4];

                        float32x4_t _tmp024c = vaddq_f32(_out0tm5, _out0tm6);
                        float32x4_t _tmp135c = vsubq_f32(_out0tm5, _out0tm6);

//                         float tmp024c = output0_tm[5] + output0_tm[6];
//                         float tmp135c = output0_tm[5] - output0_tm[6];

                        float32x4_t _tmp0m = vaddq_f32(vaddq_f32(_out0tm0, _tmp024a), vmlaq_n_f32(_tmp024b, _tmp024c, 32.f));
                        float32x4_t _tmp2m = vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f);
                        float32x4_t _tmp4m = vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f);
                        vst1q_f32(tmp[0][m], _tmp0m);
                        vst1q_f32(tmp[2][m], _tmp2m);
                        vst1q_f32(tmp[4][m], _tmp4m);

//                         tmp[0][m] = output0_tm[0] + tmp024a + tmp024b + tmp024c * 32;
//                         tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
//                         tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        float32x4_t _tmp1m = vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f);
                        float32x4_t _tmp3m = vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f);
                        float32x4_t _tmp5m = vaddq_f32(vaddq_f32(_out0tm7, _tmp135a), vmlaq_n_f32(_tmp135c, _tmp135b, 32.f));
                        vst1q_f32(tmp[1][m], _tmp1m);
                        vst1q_f32(tmp[3][m], _tmp3m);
                        vst1q_f32(tmp[5][m], _tmp5m);

//                         tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
//                         tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
//                         tmp[5][m] = output0_tm[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0_tm_0 += tiles * 32;
                        output0_tm_1 += tiles * 32;
                        output0_tm_2 += tiles * 32;
                        output0_tm_3 += tiles * 32;
                        output0_tm_4 += tiles * 32;
                        output0_tm_5 += tiles * 32;
                        output0_tm_6 += tiles * 32;
                        output0_tm_7 += tiles * 32;
                    }

                    for (int m=0; m<6; m++)
                    {
                        float32x4_t _tmp00 = vld1q_f32(tmp[m][0]);
                        float32x4_t _tmp01 = vld1q_f32(tmp[m][1]);
                        float32x4_t _tmp02 = vld1q_f32(tmp[m][2]);
                        float32x4_t _tmp03 = vld1q_f32(tmp[m][3]);
                        float32x4_t _tmp04 = vld1q_f32(tmp[m][4]);
                        float32x4_t _tmp05 = vld1q_f32(tmp[m][5]);
                        float32x4_t _tmp06 = vld1q_f32(tmp[m][6]);
                        float32x4_t _tmp07 = vld1q_f32(tmp[m][7]);

                        float32x4_t _tmp024a = vaddq_f32(_tmp01, _tmp02);
                        float32x4_t _tmp135a = vsubq_f32(_tmp01, _tmp02);

//                         float tmp024a = tmp0[1] + tmp0[2];
//                         float tmp135a = tmp0[1] - tmp0[2];

                        float32x4_t _tmp024b = vaddq_f32(_tmp03, _tmp04);
                        float32x4_t _tmp135b = vsubq_f32(_tmp03, _tmp04);

//                         float tmp024b = tmp0[3] + tmp0[4];
//                         float tmp135b = tmp0[3] - tmp0[4];

                        float32x4_t _tmp024c = vaddq_f32(_tmp05, _tmp06);
                        float32x4_t _tmp135c = vsubq_f32(_tmp05, _tmp06);

//                         float tmp024c = tmp0[5] + tmp0[6];
//                         float tmp135c = tmp0[5] - tmp0[6];

                        float32x4_t _out00 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_tmp00, _tmp024a), vmlaq_n_f32(_tmp024b, _tmp024c, 32.f)));
                        float32x4_t _out02 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f));
                        float32x4_t _out04 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f));
                        vst1q_f32(output0, _out00);
                        vst1q_f32(output0 + 8, _out02);
                        vst1q_f32(output0 + 16, _out04);

//                         output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
//                         output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
//                         output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        float32x4_t _out01 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f));
                        float32x4_t _out03 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f));
                        float32x4_t _out05 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_tmp07, _tmp135a), vmlaq_n_f32(_tmp135c, _tmp135b, 32.f)));
                        vst1q_f32(output0 + 4, _out01);
                        vst1q_f32(output0 + 12, _out03);
                        vst1q_f32(output0 + 20, _out05);

//                         output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
//                         output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
//                         output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0 += outw * 4;
                    }
                }
            }
        }
    }
}
}
