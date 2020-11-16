// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
// Copyright (C) 2019 BUG1989. All rights reserved.
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
//
#include "option.h"
#include "mat.h"
namespace ncnn{
static void conv3x3s1_winograd23_sse_BdB(const Mat& bottom_blob, Mat& top_blob, const Option& opt,
        int outch, int outh, int outw)
{
    int w = bottom_blob.w;
    //int h = bottom_blob.h;
    int inch = bottom_blob.c;

    // BEGIN transform input
    Mat bottom_blob_tm = top_blob;
    Mat bottom_blob_bordered= bottom_blob;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm/4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm/4;

        const int tiles = nColBlocks * nRowBlocks;

        bottom_blob_tm.create(4*4, tiles, inch, 4u, opt.workspace_allocator);

        // BT
        // const float itm[4][4] = {
        //     {1.0f,  0.0f, -1.0f,  0.0f},
        //     {0.0f,  1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  0.00f, 1.0f}
        // };        
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<inch; q++)
        {
            const float* img = bottom_blob_bordered.channel(q);
            float* out_tm0 = bottom_blob_tm.channel(q);

            for (int j = 0; j < nColBlocks; j++)
            {
                const float* r0 = img + w * j * 2;
                const float* r1 = r0 + w;
                const float* r2 = r1 + w;
                const float* r3 = r2 + w;

                for (int i = 0; i < nRowBlocks; i++)
                {
#if __AVX__
                    __m128 _d0, _d1, _d2, _d3;
                    __m128 _w0, _w1, _w2, _w3;

                    // load
                    _d0 = _mm_loadu_ps(r0);
                    _d1 = _mm_loadu_ps(r1);
                    _d2 = _mm_loadu_ps(r2);
                    _d3 = _mm_loadu_ps(r3);

                    // w = B_t * d
                    _w0 = _mm_sub_ps(_d0, _d2);
                    _w1 = _mm_add_ps(_d1, _d2);
                    _w2 = _mm_sub_ps(_d2, _d1);
                    _w3 = _mm_sub_ps(_d3, _d1);
                                
                    // transpose d to d_t
                    _MM_TRANSPOSE4_PS(_w0, _w1, _w2, _w3);

                    // d = B_t * d_t
                    _d0 = _mm_sub_ps(_w0, _w2);
                    _d1 = _mm_add_ps(_w1, _w2);
                    _d2 = _mm_sub_ps(_w2, _w1);
                    _d3 = _mm_sub_ps(_w3, _w1);

                    // save to out_tm
                    _mm_storeu_ps(out_tm0, _d0);
                    _mm_storeu_ps(out_tm0+4, _d1);
                    _mm_storeu_ps(out_tm0+8, _d2);
                    _mm_storeu_ps(out_tm0+12, _d3);
#else
                    float d0[4],d1[4],d2[4],d3[4];
                    float w0[4],w1[4],w2[4],w3[4];
                    float t0[4],t1[4],t2[4],t3[4];
                    // load
                    for (int n = 0; n < 4; n++)
                    {
                        d0[n] = r0[n];
                        d1[n] = r1[n];
                        d2[n] = r2[n];
                        d3[n] = r3[n];
                    }
                    // w = B_t * d
                    for (int n = 0; n < 4; n++)
                    {   
                        w0[n] = d0[n] - d2[n];
                        w1[n] = d1[n] + d2[n];
                        w2[n] = d2[n] - d1[n];
                        w3[n] = d3[n] - d1[n];
                    }                                
                    // transpose d to d_t
                    {
                        t0[0]=w0[0]; t1[0]=w0[1]; t2[0]=w0[2]; t3[0]=w0[3];
                        t0[1]=w1[0]; t1[1]=w1[1]; t2[1]=w1[2]; t3[1]=w1[3];
                        t0[2]=w2[0]; t1[2]=w2[1]; t2[2]=w2[2]; t3[2]=w2[3];
                        t0[3]=w3[0]; t1[3]=w3[1]; t2[3]=w3[2]; t3[3]=w3[3];
                    }
                    // d = B_t * d_t
                    for (int n = 0; n < 4; n++)
                    {   
                        d0[n] = t0[n] - t2[n];
                        d1[n] = t1[n] + t2[n];
                        d2[n] = t2[n] - t1[n];
                        d3[n] = t3[n] - t1[n];
                    }
                    // save to out_tm
                    for (int n = 0; n < 4; n++)
                    {
                        out_tm0[n   ] = d0[n];
                        out_tm0[n+ 4] = d1[n];
                        out_tm0[n+ 8] = d2[n];
                        out_tm0[n+12] = d3[n];
                    }                  
#endif
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;

                    out_tm0 += 16;
                }
            }
        }
    }
}
}
