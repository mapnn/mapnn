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
static void conv3x3s1_winograd64_pack4_neon_permute(const Mat& bottom_blob, Mat& top_blob, const Option& opt,
        int outch, int inch, int outh, int outw)
{
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    Mat bottom_blob_tm = bottom_blob;
    //Mat bottom_blob_tm2 = top_blob;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;

        const int tiles = h_tm/8 * w_tm/8;

        // permute
//         bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);
        Mat bottom_blob_tm2 = top_blob;
#if __aarch64__
        if (tiles >= 12)
            bottom_blob_tm2.create(12 * inch, tiles/12 + (tiles%12)/8 + (tiles%12%8)/4 + (tiles%12%4)/2 + tiles%12%2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles/8 + (tiles%8)/4 + (tiles%4)/2 + tiles%2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles/4 + (tiles%4)/2 + tiles%2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles/2 + tiles%2, 64, elemsize, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, elemsize, elempack, opt.workspace_allocator);
#else
        if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles/8 + (tiles%8)/4 + (tiles%4)/2 + tiles%2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles/4 + (tiles%4)/2 + tiles%2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles/2 + tiles%2, 64, elemsize, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, elemsize, elempack, opt.workspace_allocator);
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r=0; r<64; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i=0;
#if __aarch64__
            for (; i+11<tiles; i+=12)
            {
                float* tm2p = tm2.row(i/12);

                const float* r0 = bottom_blob_tm;

                r0 += (r*tiles + i) * 4;

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
                        : "=r"(r0),     // %0
                          "=r"(tm2p)    // %1
                        : "0"(r0),
                          "1"(tm2p)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
                    );
                    r0 += bottom_blob_tm.cstep * 4;
                }
            }
#endif
            for (; i+7<tiles; i+=8)
            {
#if __aarch64__
                float* tm2p = tm2.row(i/12 + (i%12)/8);
#else
                float* tm2p = tm2.row(i/8);
#endif

                const float* r0 = bottom_blob_tm;

                r0 += (r*tiles + i) * 4;

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
                        : "=r"(r0),     // %0
                          "=r"(tm2p)    // %1
                        : "0"(r0),
                          "1"(tm2p)
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
                        : "=r"(r0),     // %0
                          "=r"(tm2p)    // %1
                        : "0"(r0),
                          "1"(tm2p)
                        : "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
                    );
#endif
                    r0 += bottom_blob_tm.cstep * 4;
                }
            }
            for (; i+3<tiles; i+=4)
            {
#if __aarch64__
                float* tm2p = tm2.row(i/12 + (i%12)/8 + (i%12%8)/4);
#else
                float* tm2p = tm2.row(i/8 + (i%8)/4);
#endif

                const float* r0 = bottom_blob_tm;

                r0 += (r*tiles + i) * 4;

                for (int q=0; q<inch; q++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                        : "=r"(r0),     // %0
                          "=r"(tm2p)    // %1
                        : "0"(r0),
                          "1"(tm2p)
                        : "memory", "v0", "v1", "v2", "v3"
                    );
#else
                    asm volatile(
                        "pld        [%0, #512]          \n"
                        "vldm       %0, {d0-d7}         \n"
                        "vstm       %1!, {d0-d7}        \n"
                        : "=r"(r0),     // %0
                          "=r"(tm2p)    // %1
                        : "0"(r0),
                          "1"(tm2p)
                        : "memory", "q0", "q1", "q2", "q3"
                    );
#endif // __aarch64__
                    r0 += bottom_blob_tm.cstep * 4;
                }
            }
            for (; i+1<tiles; i+=2)
            {
#if __aarch64__
                float* tm2p = tm2.row(i/12 + (i%12)/8 + (i%12%8)/4 + (i%12%4)/2);
#else
                float* tm2p = tm2.row(i/8 + (i%8)/4 + (i%4)/2);
#endif

                const float* r0 = bottom_blob_tm;

                r0 += (r*tiles + i) * 4;

                for (int q=0; q<inch; q++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%0]        \n"
                        "st1    {v0.4s, v1.4s}, [%1], #32   \n"
                        : "=r"(r0),     // %0
                          "=r"(tm2p)    // %1
                        : "0"(r0),
                          "1"(tm2p)
                        : "memory", "v0", "v1"
                    );
#else
                    asm volatile(
                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d0-d3}, [%0 :128]  \n"
                        "vst1.f32   {d0-d3}, [%1 :128]! \n"
                        : "=r"(r0),     // %0
                          "=r"(tm2p)    // %1
                        : "0"(r0),
                          "1"(tm2p)
                        : "memory", "q0", "q1"
                    );
#endif // __aarch64__
                    r0 += bottom_blob_tm.cstep * 4;
                }
            }
            for (; i<tiles; i++)
            {
#if __aarch64__
                float* tm2p = tm2.row(i/12 + (i%12)/8 + (i%12%8)/4 + (i%12%4)/2 + i%12%2);
#else
                float* tm2p = tm2.row(i/8 + (i%8)/4 + (i%4)/2 + i%2);
#endif

                const float* r0 = bottom_blob_tm;

                r0 += (r*tiles + i) * 4;

                for (int q=0; q<inch; q++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v0.4s}, [%0]               \n"
                        "st1    {v0.4s}, [%1], #16          \n"
                        : "=r"(r0),     // %0
                          "=r"(tm2p)    // %1
                        : "0"(r0),
                          "1"(tm2p)
                        : "memory", "v0"
                    );
#else
                    asm volatile(
                        "pld        [%0, #128]          \n"
                        "vld1.f32   {d0-d1}, [%0 :128]  \n"
                        "vst1.f32   {d0-d1}, [%1 :128]! \n"
                        : "=r"(r0),     // %0
                          "=r"(tm2p)    // %1
                        : "0"(r0),
                          "1"(tm2p)
                        : "memory", "q0"
                    );
#endif // __aarch64__
                    r0 += bottom_blob_tm.cstep * 4;
                }
            }
        }

    }
}
}
