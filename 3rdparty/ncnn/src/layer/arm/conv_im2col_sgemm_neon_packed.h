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

#include "option.h"
#include "mat.h"
namespace ncnn{
static void conv_im2col_sgemm_neon_packed(const Mat &bottom_blob, Mat &top_blob,
            const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Option& opt,
            int inch, int outw, int outh, int outch)
{
    //size_t elemsize = bottom_blob.elemsize;
    int kernel_size = kernel_w * kernel_h;
    int out_size = outw * outh;
    const Mat bottom_im2col = bottom_blob;
    Mat bottom_tm = top_blob;
    {
        int nn_size = out_size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = ii * 8;

            const float* img0 = bottom_im2col.channel(0);
            img0 += i;

            float* tmpptr = bottom_tm.channel(i/8);

            for (int q=0; q<inch*kernel_size; q++)
            {
#if __ARM_NEON
#if __aarch64__
                asm volatile(
                    "prfm    pldl1keep, [%0, #256]   \n"
                    "ld1     {v0.4s, v1.4s}, [%0]    \n"
                    "st1     {v0.4s, v1.4s}, [%1]    \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "cc", "memory", "v0", "v1"
                );                
#else
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld1.f32   {d0-d3}, [%0]       \n"
                    "vst1.f32   {d0-d3}, [%1]       \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "memory", "q0", "q1"
                );
#endif // __aarch64__
#else                
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];
                tmpptr[4] = img0[4];
                tmpptr[5] = img0[5];
                tmpptr[6] = img0[6];
                tmpptr[7] = img0[7];
#endif // __ARM_NEON              
                tmpptr += 8;
                img0 += out_size;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=remain_size_start; i<out_size; i++)
        {
            const float* img0 = bottom_im2col.channel(0);
            img0 += i;

            float* tmpptr = bottom_tm.channel(i/8 + i%8);

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
