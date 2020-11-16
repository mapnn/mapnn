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
static void conv1x1s1_sgemm_pack4_neon_interleave(const Mat& bottom_blob, Mat& top_blob, const Option& opt,
        int outch, int inch)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int size = w * h;

    // interleave
    Mat tmp = top_blob;
#if __aarch64__
    tmp.create(12, inch, size/12 + (size%12)/8 + (size%12%8)/4 + (size%12%4)/2 + size%12%2, elemsize, elempack);
#else
    tmp.create(8, inch, size/8 + (size%8)/4 + (size%4)/2 + size%2, elemsize, elempack);
#endif
    {
        int nn_size;
        int remain_size_start;

#if __aarch64__
        nn_size = size / 12;
        remain_size_start = nn_size * 12;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = ii * 12;

            const float* img0 = bottom_blob.channel(0);
            img0 += i*4;

            float* tmpptr = tmp.channel(i/12);

            for (int q=0; q<inch; q++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0] \n"
                    "st1    {v0.4s}, [%1], #16          \n"
                    "st1    {v4.4s}, [%1], #16          \n"
                    "st1    {v8.4s}, [%1], #16          \n"
                    "sub    %0, %0, #128                \n"
                    "st1    {v1.4s}, [%1], #16          \n"
                    "st1    {v5.4s}, [%1], #16          \n"
                    "st1    {v9.4s}, [%1], #16          \n"
                    "st1    {v2.4s}, [%1], #16          \n"
                    "st1    {v6.4s}, [%1], #16          \n"
                    "st1    {v10.4s}, [%1], #16         \n"
                    "st1    {v3.4s}, [%1], #16          \n"
                    "st1    {v7.4s}, [%1], #16          \n"
                    "st1    {v11.4s}, [%1], #16         \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
                );
                img0 += bottom_blob.cstep * 4;
            }
        }
#else
        remain_size_start = 0;
#endif
        nn_size = (size - remain_size_start) >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            const float* img0 = bottom_blob.channel(0);
            img0 += i*4;

#if __aarch64__
            float* tmpptr = tmp.channel(i/12+(i%12)/8);
#else
            float* tmpptr = tmp.channel(i/8);
#endif

            for (int q=0; q<inch; q++)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0] \n"
                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                    "sub    %0, %0, #64                 \n"
                    "st1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                );
#else
                asm volatile(
                    "pld        [%0, #512]          \n"
                    "vldm       %0!, {d0-d7}        \n"
                    "pld        [%0, #512]          \n"
                    "vldm       %0, {d16-d23}       \n"

                    // transpose 8x4
                    "vtrn.32    q0, q1              \n"
                    "vtrn.32    q2, q3              \n"
                    "vtrn.32    q8, q9              \n"
                    "vtrn.32    q10, q11            \n"
                    "vswp       d1, d4              \n"
                    "vswp       d3, d6              \n"
                    "vswp       d17, d20            \n"
                    "vswp       d19, d22            \n"
                    "vswp       q1, q8              \n"
                    "vswp       q3, q10             \n"

                    "vst1.f32   {d0-d3}, [%1 :128]! \n"
                    "vst1.f32   {d16-d19}, [%1 :128]! \n"
                    "sub        %0, %0, #64         \n"
                    "vst1.f32   {d4-d7}, [%1 :128]! \n"
                    "vst1.f32   {d20-d23}, [%1 :128]! \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
                );
#endif // __aarch64__
                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            const float* img0 = bottom_blob.channel(0);
            img0 += i*4;

#if __aarch64__
            float* tmpptr = tmp.channel(i/12 + (i%12)/8 + (i%12%8)/4);
#else
            float* tmpptr = tmp.channel(i/8 + (i%8)/4);
#endif

            for (int q=0; q<inch; q++)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "memory", "v0", "v1", "v2", "v3"
                );
#else
                asm volatile(
                    "pld        [%0, #512]          \n"
                    "vldm       %0, {d0-d7}         \n"
                    "vstm       %1!, {d0-d7}        \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "memory", "q0", "q1", "q2", "q3"
                );
#endif // __aarch64__
                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            const float* img0 = bottom_blob.channel(0);
            img0 += i*4;

#if __aarch64__
            float* tmpptr = tmp.channel(i/12 + (i%12)/8 + (i%12%8)/4 + (i%12%4)/2);
#else
            float* tmpptr = tmp.channel(i/8 + (i%8)/4 + (i%4)/2);
#endif

            for (int q=0; q<inch; q++)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%0]        \n"
                    "st1    {v0.4s, v1.4s}, [%1], #32   \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "memory", "v0", "v1"
                );
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
#endif // __aarch64__
                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=remain_size_start; i<size; i++)
        {
            const float* img0 = bottom_blob.channel(0);
            img0 += i*4;

#if __aarch64__
            float* tmpptr = tmp.channel(i/12 + (i%12)/8 + (i%12%8)/4 + (i%12%4)/2 + i%12%2);
#else
            float* tmpptr = tmp.channel(i/8 + (i%8)/4 + (i%4)/2 + i%2);
#endif

            for (int q=0; q<inch; q++)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v0.4s}, [%0]               \n"
                    "st1    {v0.4s}, [%1], #16          \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "memory", "v0"
                );
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
#endif // __aarch64__
                img0 += bottom_blob.cstep * 4;
            }
        }
    }

}
}
