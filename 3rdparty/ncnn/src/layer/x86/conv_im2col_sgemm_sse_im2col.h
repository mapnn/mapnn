// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
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
static void conv_im2col_sgemm_sse_im2col(const Mat &bottom_blob, Mat &top_blob,
            const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Option& opt,
            int outch, int outh, int outw)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    // im2col
    Mat bottom_im2col = top_blob;
    bottom_im2col.create(outw*outh, kernel_h*kernel_w*inch, elemsize, opt.workspace_allocator);
    {
        const int stride = kernel_h*kernel_w*outw*outh;
        float* ret = (float*)bottom_im2col;
    
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<inch; p++)
        {
            const float* input = bottom_blob.channel(p);
            int retID = stride * p;
            for (int u=0; u<kernel_h; u++)
            {
                for (int v=0; v<kernel_w; v++)
                {
                    for (int i=0; i<outh; i++)
                    {
                        for (int j=0; j<outw; j++)
                        {
                            int row = u + i * stride_h;
                            int col = v + j * stride_w;
                            int index = row * w + col;
                            ret[retID] = input[index];
                            retID++;
                        }
                    }
                }
            }
        }
    }
}
}
