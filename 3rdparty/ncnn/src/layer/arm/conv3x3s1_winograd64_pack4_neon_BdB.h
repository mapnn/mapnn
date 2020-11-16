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
static void conv3x3s1_winograd64_pack4_neon_BdB(const Mat& bottom_blob, Mat& top_blob, const Option& opt,
        int inch, int outh, int outw)
{
    int w = bottom_blob.w;
    //int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    Mat bottom_blob_bordered = bottom_blob;

    // BEGIN transform input
    Mat bottom_blob_tm = top_blob;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;

        const int tiles = w_tm/8 * h_tm/8;

        bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);

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

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q<inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            float tmp[8][8][4];

            // tile
            for (int i=0; i<h_tm/8; i++)
            {
                for (int j=0; j<w_tm/8; j++)
                {
                    const float* r0 = img0.row(i * 6) + (j * 6) * 4;

                    for (int m=0; m<8; m++)
                    {
                        float32x4_t _r00 = vld1q_f32(r0);
                        float32x4_t _r01 = vld1q_f32(r0 + 4);
                        float32x4_t _r02 = vld1q_f32(r0 + 8);
                        float32x4_t _r03 = vld1q_f32(r0 + 12);
                        float32x4_t _r04 = vld1q_f32(r0 + 16);
                        float32x4_t _r05 = vld1q_f32(r0 + 20);
                        float32x4_t _r06 = vld1q_f32(r0 + 24);
                        float32x4_t _r07 = vld1q_f32(r0 + 28);

                        float32x4_t _tmp0m = vmlaq_n_f32(vsubq_f32(_r00, _r06), vsubq_f32(_r04, _r02), 5.25f);
                        float32x4_t _tmp7m = vmlaq_n_f32(vsubq_f32(_r07, _r01), vsubq_f32(_r03, _r05), 5.25f);
                        vst1q_f32(tmp[0][m], _tmp0m);
                        vst1q_f32(tmp[7][m], _tmp7m);

//                         tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25;
//                         tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25;

                        float32x4_t _tmp12a = vmlsq_n_f32(vaddq_f32(_r02, _r06), _r04, 4.25f);
                        float32x4_t _tmp12b = vmlsq_n_f32(vaddq_f32(_r01, _r05), _r03, 4.25f);

//                         float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25);
//                         float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25);

                        float32x4_t _tmp1m = vaddq_f32(_tmp12a, _tmp12b);
                        float32x4_t _tmp2m = vsubq_f32(_tmp12a, _tmp12b);
                        vst1q_f32(tmp[1][m], _tmp1m);
                        vst1q_f32(tmp[2][m], _tmp2m);

//                         tmp[1][m] = tmp12a + tmp12b;
//                         tmp[2][m] = tmp12a - tmp12b;

                        float32x4_t _tmp34a = vmlsq_n_f32(vmlaq_n_f32(_r06, _r02, 0.25f), _r04, 1.25f);
                        float32x4_t _tmp34b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_r01, 0.5f), _r03, 2.5f), _r05, 2.f);

//                         float tmp34a = (r0[6] + r0[2] * 0.25 - r0[4] * 1.25);
//                         float tmp34b = (r0[1] * 0.5 - r0[3] * 2.5 + r0[5] * 2);

                        float32x4_t _tmp3m = vaddq_f32(_tmp34a, _tmp34b);
                        float32x4_t _tmp4m = vsubq_f32(_tmp34a, _tmp34b);
                        vst1q_f32(tmp[3][m], _tmp3m);
                        vst1q_f32(tmp[4][m], _tmp4m);

//                         tmp[3][m] = tmp34a + tmp34b;
//                         tmp[4][m] = tmp34a - tmp34b;

                        float32x4_t _tmp56a = vmlaq_n_f32(_r06, vmlsq_n_f32(_r02, _r04, 1.25f), 4.f);
                        float32x4_t _tmp56b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_r01, 2.f), _r03, 2.5f), _r05, 0.5f);

//                         float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25) * 4);
//                         float tmp56b = (r0[1] * 2 - r0[3] * 2.5 + r0[5] * 0.5);

                        float32x4_t _tmp5m = vaddq_f32(_tmp56a, _tmp56b);
                        float32x4_t _tmp6m = vsubq_f32(_tmp56a, _tmp56b);
                        vst1q_f32(tmp[5][m], _tmp5m);
                        vst1q_f32(tmp[6][m], _tmp6m);

//                         tmp[5][m] = tmp56a + tmp56b;
//                         tmp[6][m] = tmp56a - tmp56b;

                        r0 += w * 4;
                    }

                    float* r0_tm_0 = (float*)img0_tm + (i * w_tm/8 + j) * 4;
                    float* r0_tm_1 = r0_tm_0 + tiles * 4;
                    float* r0_tm_2 = r0_tm_0 + tiles * 8;
                    float* r0_tm_3 = r0_tm_0 + tiles * 12;
                    float* r0_tm_4 = r0_tm_0 + tiles * 16;
                    float* r0_tm_5 = r0_tm_0 + tiles * 20;
                    float* r0_tm_6 = r0_tm_0 + tiles * 24;
                    float* r0_tm_7 = r0_tm_0 + tiles * 28;

                    for (int m=0; m<8; m++)
                    {
                        float32x4_t _tmp00 = vld1q_f32(tmp[m][0]);
                        float32x4_t _tmp01 = vld1q_f32(tmp[m][1]);
                        float32x4_t _tmp02 = vld1q_f32(tmp[m][2]);
                        float32x4_t _tmp03 = vld1q_f32(tmp[m][3]);
                        float32x4_t _tmp04 = vld1q_f32(tmp[m][4]);
                        float32x4_t _tmp05 = vld1q_f32(tmp[m][5]);
                        float32x4_t _tmp06 = vld1q_f32(tmp[m][6]);
                        float32x4_t _tmp07 = vld1q_f32(tmp[m][7]);

                        float32x4_t _r0tm0 = vmlaq_n_f32(vsubq_f32(_tmp00, _tmp06), vsubq_f32(_tmp04, _tmp02), 5.25f);
                        float32x4_t _r0tm7 = vmlaq_n_f32(vsubq_f32(_tmp07, _tmp01), vsubq_f32(_tmp03, _tmp05), 5.25f);

//                         r0_tm[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25;
//                         r0_tm[7] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25;

                        float32x4_t _tmp12a = vmlsq_n_f32(vaddq_f32(_tmp02, _tmp06), _tmp04, 4.25f);
                        float32x4_t _tmp12b = vmlsq_n_f32(vaddq_f32(_tmp01, _tmp05), _tmp03, 4.25f);

//                         float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25);
//                         float tmp12b = (tmp0[1] + tmp0[5] - tmp0[3] * 4.25);

                        float32x4_t _r0tm1 = vaddq_f32(_tmp12a, _tmp12b);
                        float32x4_t _r0tm2 = vsubq_f32(_tmp12a, _tmp12b);

//                         r0_tm[1] = tmp12a + tmp12b;
//                         r0_tm[2] = tmp12a - tmp12b;

                        float32x4_t _tmp34a = vmlsq_n_f32(vmlaq_n_f32(_tmp06, _tmp02, 0.25f), _tmp04, 1.25f);
                        float32x4_t _tmp34b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_tmp01, 0.5f), _tmp03, 2.5f), _tmp05, 2.f);

//                         float tmp34a = (tmp0[6] + tmp0[2] * 0.25 - tmp0[4] * 1.25);
//                         float tmp34b = (tmp0[1] * 0.5 - tmp0[3] * 2.5 + tmp0[5] * 2);

                        float32x4_t _r0tm3 = vaddq_f32(_tmp34a, _tmp34b);
                        float32x4_t _r0tm4 = vsubq_f32(_tmp34a, _tmp34b);

//                         r0_tm[3] = tmp34a + tmp34b;
//                         r0_tm[4] = tmp34a - tmp34b;

                        float32x4_t _tmp56a = vmlaq_n_f32(_tmp06, vmlsq_n_f32(_tmp02, _tmp04, 1.25f), 4.f);
                        float32x4_t _tmp56b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_tmp01, 2.f), _tmp03, 2.5f), _tmp05, 0.5f);

//                         float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25) * 4);
//                         float tmp56b = (tmp0[1] * 2 - tmp0[3] * 2.5 + tmp0[5] * 0.5);

                        float32x4_t _r0tm5 = vaddq_f32(_tmp56a, _tmp56b);
                        float32x4_t _r0tm6 = vsubq_f32(_tmp56a, _tmp56b);

//                         r0_tm[5] = tmp56a + tmp56b;
//                         r0_tm[6] = tmp56a - tmp56b;

                        vst1q_f32(r0_tm_0, _r0tm0);
                        vst1q_f32(r0_tm_1, _r0tm1);
                        vst1q_f32(r0_tm_2, _r0tm2);
                        vst1q_f32(r0_tm_3, _r0tm3);
                        vst1q_f32(r0_tm_4, _r0tm4);
                        vst1q_f32(r0_tm_5, _r0tm5);
                        vst1q_f32(r0_tm_6, _r0tm6);
                        vst1q_f32(r0_tm_7, _r0tm7);

                        r0_tm_0 += tiles * 32;
                        r0_tm_1 += tiles * 32;
                        r0_tm_2 += tiles * 32;
                        r0_tm_3 += tiles * 32;
                        r0_tm_4 += tiles * 32;
                        r0_tm_5 += tiles * 32;
                        r0_tm_6 += tiles * 32;
                        r0_tm_7 += tiles * 32;
                    }
                }
            }
        }

    }
}
}
