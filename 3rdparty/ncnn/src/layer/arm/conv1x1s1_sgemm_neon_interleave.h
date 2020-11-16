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
static void conv1x1s1_sgemm_neon_interleave(const Mat& bottom_blob, Mat& top_blob, const Option& opt,
        int w, int h, int inch, int outch)
{
    // interleave
    Mat& tmp = top_blob;
    {
        const int size = w * h;
        int nn_size = size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = ii * 8;

            const float* img0 = bottom_blob.channel(0);
            img0 += i;

            float* tmpptr = tmp.channel(i/8);

            for (int q=0; q<inch; q++)
            {
#if __ARM_NEON
#if __aarch64__
                vst1q_f32(tmpptr, vld1q_f32(img0));
                vst1q_f32(tmpptr+4, vld1q_f32(img0+4));

                tmpptr += 8;
                img0 += bottom_blob.cstep;
#else
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld1.f32   {d0-d3}, [%0 :128]  \n"
                    "vst1.f32   {d0-d3}, [%1 :128]! \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "memory", "q0", "q1"
                );

                img0 += bottom_blob.cstep;
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

                tmpptr += 8;
                img0 += bottom_blob.cstep;
#endif // __ARM_NEON
            }
        }

        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            const float* img0 = bottom_blob.channel(0);
            img0 += i;

            float* tmpptr = tmp.channel(i/8 + (i%8)/4);

            for (int q=0; q<inch; q++)
            {
#if __ARM_NEON
#if __aarch64__
                vst1q_f32(tmpptr, vld1q_f32(img0));

                tmpptr += 4;
                img0 += bottom_blob.cstep;
#else
                asm volatile(
                    "pld        [%0, #128]          \n"
                    "vld1.f32   {d0-d1}, [%0 :128]  \n"
                    "vst1.f32   {d0-d1}, [%1 :128]! \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "memory", "q0"
                );

                img0 += bottom_blob.cstep;
#endif // __aarch64__
#else
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];

                tmpptr += 4;
                img0 += bottom_blob.cstep;
#endif // __ARM_NEON
            }
        }

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=remain_size_start; i<size; i++)
        {
            const float* img0 = bottom_blob.channel(0);
            img0 += i;

            float* tmpptr = tmp.channel(i/8 + (i%8)/4 + i%4);

            for (int q=0; q<inch; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr++;
                img0 += bottom_blob.cstep;
            }
        }
    }
}
}
