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
static void conv_im2col_sgemm_sse_pack(const Mat &bottom_blob, Mat &top_blob,
            const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Option& opt,
            int inch, int outch, int outh, int outw)
{
    size_t elemsize = bottom_blob.elemsize;

    int kernel_size = kernel_w * kernel_h;
    int out_size = outw * outh;

    // bottom_im2col memory packed 4 x 4
    Mat bottom_im2col = bottom_blob;
    Mat bottom_tm = top_blob;
    bottom_tm.create(4*kernel_size, inch, out_size/4 + out_size%4, elemsize, opt.workspace_allocator);
    {
        int nn_size = out_size >> 2;
        int remain_size_start = nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = ii * 4;

            const float* img0 = bottom_im2col.channel(0);
            img0 += i;

            float* tmpptr = bottom_tm.channel(i/4);

            for (int q=0; q<inch*kernel_size; q++)
            {
#if __SSE__
                _mm_storeu_ps(tmpptr, _mm_loadu_ps(img0));
#else                
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];
#endif // __SSE__              
                tmpptr += 4;
                img0 += out_size;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=remain_size_start; i<out_size; i++)
        {
            const float* img0 = bottom_im2col.channel(0);
            img0 += i;

            float* tmpptr = bottom_tm.channel(i/4 + i%4);

            for (int q=0; q<inch*kernel_size; q++)
            {
                tmpptr[0] = img0[0];

                tmpptr += 1;
                img0 += out_size;
            }
        }
    }
}
}
