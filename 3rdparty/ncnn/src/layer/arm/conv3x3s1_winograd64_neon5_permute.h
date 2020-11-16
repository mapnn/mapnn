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
static void conv3x3s1_winograd64_neon5_permute(const Mat& bottom_blob, Mat& top_blob, const Option& opt,
        int inch, int outw, int outh, int outch)
{
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm/8 * h_tm/8;

        // permute
        // bottom_blob_tm.create(1, 64 * tiles, inch);
//         Mat bottom_blob_tm2(inch, tiles, 64);
        Mat bottom_blob_tm2 = top_blob;
        Mat bottom_blob_tm = bottom_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r=0; r<64; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i=0;
            for (; i+7<tiles; i+=8)
            {
                float* tm2p = tm2.row(i/8);

                const float* r0 = bottom_blob_tm;

                r0 += r*tiles + i;

                for (int q=0; q<inch; q++)
                {
#if __ARM_NEON
                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r0n = vld1q_f32(r0+4);
                    vst1q_f32(tm2p, _r0);
                    vst1q_f32(tm2p+4, _r0n);
#else
                    tm2p[0] = r0[0];
                    tm2p[1] = r0[1];
                    tm2p[2] = r0[2];
                    tm2p[3] = r0[3];
                    tm2p[4] = r0[4];
                    tm2p[5] = r0[5];
                    tm2p[6] = r0[6];
                    tm2p[7] = r0[7];
#endif // __ARM_NEON

                    r0 += bottom_blob_tm.cstep;
                    tm2p += 8;
                }
            }
            for (; i+3<tiles; i+=4)
            {
                float* tm2p = tm2.row(i/8+(i%8)/4);

                const float* r0 = bottom_blob_tm;

                r0 += r*tiles + i;

                for (int q=0; q<inch; q++)
                {
#if __ARM_NEON
                    float32x4_t _r0 = vld1q_f32(r0);
                    vst1q_f32(tm2p, _r0);
#else
                    tm2p[0] = r0[0];
                    tm2p[1] = r0[1];
                    tm2p[2] = r0[2];
                    tm2p[3] = r0[3];
#endif // __ARM_NEON

                    r0 += bottom_blob_tm.cstep;
                    tm2p += 4;
                }
            }
            for (; i<tiles; i++)
            {
                float* tm2p = tm2.row(i/8+(i%8)/4+i%4);

                const float* r0 = bottom_blob_tm;

                r0 += r*tiles + i;

                for (int q=0; q<inch; q++)
                {
                    tm2p[0] = r0[0];

                    r0 += bottom_blob_tm.cstep;
                    tm2p += 1;
                }
            }
        }
    }
}
}
