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
static void eltwise_add_arm(const Mat& bottom_blob, const Mat& bottom_blob1, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;
    //#pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        const float* ptr1 = bottom_blob1.channel(q);
        float* outptr = top_blob.channel(q);

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        if (nn > 0)
        {
            asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2], #16    \n"
                    "fadd       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%3], #16    \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                    "=r"(ptr),    // %1
                    "=r"(ptr1),   // %2
                    "=r"(outptr)  // %3
                    : "0"(nn),
                    "1"(ptr),
                    "2"(ptr1),
                    "3"(outptr)
                    : "cc", "memory", "v0", "v1"
                    );
        }
#else
        if (nn > 0)
        {
            asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1]! \n"
                    "vld1.f32   {d2-d3}, [%2]! \n"
                    "vadd.f32   q0, q0, q1          \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%3]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                    "=r"(ptr),    // %1
                    "=r"(ptr1),   // %2
                    "=r"(outptr)  // %3
                    : "0"(nn),
                    "1"(ptr),
                    "2"(ptr1),
                    "3"(outptr)
                    : "cc", "memory", "q0", "q1"
                    );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *outptr = *ptr + *ptr1;

            ptr++;
            ptr1++;
            outptr++;
        }
    }
}
}
