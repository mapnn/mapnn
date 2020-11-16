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
static void conv3x3s1_winograd23_sse_AoA(const Mat& bottom_blob, Mat& top_blob, const Mat& _bias, const Option& opt,
        int inch, int outch, int outh, int outw)
{
    Mat top_blob_tm = bottom_blob;
    Mat top_blob_bordered = top_blob;
    top_blob_bordered.create(outw, outh, outch, 4u, opt.workspace_allocator);
    const float* bias = _bias;
    {
        // AT
        // const float itm[2][4] = {
        //     {1.0f,  1.0f,  1.0f,  0.0f},
        //     {0.0f,  1.0f, -1.0f,  1.0f}
        // }; 

        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm/4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm/4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<outch; p++)
        {
            Mat out_tm = top_blob_tm.channel(p);
            Mat out = top_blob_bordered.channel(p);

            const float bias0 = bias ? bias[p] : 0.f;

            for (int j=0; j<nColBlocks; j++)
            {
                float* outRow0 = out.row(j*2);
                float* outRow1 = out.row(j*2+1);

                for(int i=0; i<nRowBlocks; i++)
                {
                    float* out_tile = out_tm.row(j*nRowBlocks + i);

                    float s0[4],s1[4],s2[4],s3[4];
                    float w0[4],w1[4];
                    float d0[2],d1[2],d2[2],d3[2];
                    float o0[2],o1[2];
                    // load
                    for (int n = 0; n < 4; n++)
                    {
                        s0[n] = out_tile[n];
                        s1[n] = out_tile[n+ 4];
                        s2[n] = out_tile[n+ 8];
                        s3[n] = out_tile[n+12];
                    }
                    // w = A_T * W
                    for (int n = 0; n < 4; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n];
                        w1[n] = s1[n] - s2[n] + s3[n];
                    }
                    // transpose w to w_t
                    {
                        d0[0] = w0[0]; d0[1] = w1[0];
                        d1[0] = w0[1]; d1[1] = w1[1];
                        d2[0] = w0[2]; d2[1] = w1[2];
                        d3[0] = w0[3]; d3[1] = w1[3];
                    }
                    // Y = A_T * w_t
                    for (int n = 0; n < 2; n++)
                    {
                        o0[n] = d0[n] + d1[n] + d2[n] + bias0;
                        o1[n] = d1[n] - d2[n] + d3[n] + bias0;
                    }
                    // save to top blob tm
                    outRow0[0] = o0[0];
                    outRow0[1] = o0[1];
                    outRow1[0] = o1[0];
                    outRow1[1] = o1[1];

                    outRow0 += 2;
                    outRow1 += 2;
                }
            }
        }        
    }
    // END transform output 
}
}
