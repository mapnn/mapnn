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
static void conv3x3s1_winograd64_transform_kernel_neon_pack(const Mat& kernel_tm, Mat& kernel_tm2, int inch, int outch)
{
    // optimized layout for winograd4
    // interleave weights
    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;

    kernel_tm2.create(8*8 * inch * 4, 1, nn_outch + (outch % 4 + 3) / 4);

    #pragma omp parallel for
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 4;

        float* ktm2 = kernel_tm2.channel(pp);

        const Mat kernel0_tm = kernel_tm.channel(p);
        const Mat kernel1_tm = kernel_tm.channel(p+1);
        const Mat kernel2_tm = kernel_tm.channel(p+2);
        const Mat kernel3_tm = kernel_tm.channel(p+3);

        int q=0;

#if __ARM_NEON && __aarch64__
        for (; q+3<inch; q+=4)
        {
            const float* k00 = kernel0_tm.row(q);
            const float* k01 = kernel0_tm.row(q+1);
            const float* k02 = kernel0_tm.row(q+2);
            const float* k03 = kernel0_tm.row(q+3);
            const float* k10 = kernel1_tm.row(q);
            const float* k11 = kernel1_tm.row(q+1);
            const float* k12 = kernel1_tm.row(q+2);
            const float* k13 = kernel1_tm.row(q+3);
            const float* k20 = kernel2_tm.row(q);
            const float* k21 = kernel2_tm.row(q+1);
            const float* k22 = kernel2_tm.row(q+2);
            const float* k23 = kernel2_tm.row(q+3);
            const float* k30 = kernel3_tm.row(q);
            const float* k31 = kernel3_tm.row(q+1);
            const float* k32 = kernel3_tm.row(q+2);
            const float* k33 = kernel3_tm.row(q+3);

            for (int r=0; r<16; r++)
            {
            // split into two asm blocks for gcc reject over 30 oprands :(
            asm volatile(
                "ld1    {v0.4s}, [%1], #16  \n"
                "ld1    {v1.4s}, [%2], #16  \n"
                "ld1    {v2.4s}, [%3], #16  \n"
                "ld1    {v3.4s}, [%4], #16  \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64  \n"

                "ld1    {v0.4s}, [%5], #16  \n"
                "ld1    {v1.4s}, [%6], #16  \n"
                "ld1    {v2.4s}, [%7], #16  \n"
                "ld1    {v3.4s}, [%8], #16  \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64  \n"

                : "=r"(ktm2),   // %0
                  "=r"(k00),    // %1
                  "=r"(k01),    // %2
                  "=r"(k02),    // %3
                  "=r"(k03),    // %4
                  "=r"(k10),    // %5
                  "=r"(k11),    // %6
                  "=r"(k12),    // %7
                  "=r"(k13)     // %8
                : "0"(ktm2),
                  "1"(k00),
                  "2"(k01),
                  "3"(k02),
                  "4"(k03),
                  "5"(k10),
                  "6"(k11),
                  "7"(k12),
                  "8"(k13)
                : "cc", "memory", "v0", "v1", "v2", "v3"
            );
            asm volatile(
                "ld1    {v0.4s}, [%1], #16  \n"
                "ld1    {v1.4s}, [%2], #16  \n"
                "ld1    {v2.4s}, [%3], #16  \n"
                "ld1    {v3.4s}, [%4], #16  \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64  \n"

                "ld1    {v0.4s}, [%5], #16  \n"
                "ld1    {v1.4s}, [%6], #16  \n"
                "ld1    {v2.4s}, [%7], #16  \n"
                "ld1    {v3.4s}, [%8], #16  \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64  \n"

                : "=r"(ktm2),   // %0
                  "=r"(k20),    // %1
                  "=r"(k21),    // %2
                  "=r"(k22),    // %3
                  "=r"(k23),    // %4
                  "=r"(k30),    // %5
                  "=r"(k31),    // %6
                  "=r"(k32),    // %7
                  "=r"(k33)     // %8
                : "0"(ktm2),
                  "1"(k20),
                  "2"(k21),
                  "3"(k22),
                  "4"(k23),
                  "5"(k30),
                  "6"(k31),
                  "7"(k32),
                  "8"(k33)
                : "cc", "memory", "v0", "v1", "v2", "v3"
            );
            }
        }
#endif // __ARM_NEON && __aarch64__

        for (; q+1<inch; q+=2)
        {
            const float* k00 = kernel0_tm.row(q);
            const float* k01 = kernel0_tm.row(q+1);
            const float* k10 = kernel1_tm.row(q);
            const float* k11 = kernel1_tm.row(q+1);
            const float* k20 = kernel2_tm.row(q);
            const float* k21 = kernel2_tm.row(q+1);
            const float* k30 = kernel3_tm.row(q);
            const float* k31 = kernel3_tm.row(q+1);

            for (int r=0; r<16; r++)
            {
#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%1], #16  \n"
                "ld1    {v1.4s}, [%2], #16  \n"
                "st1    {v0.4s, v1.4s}, [%0], #32  \n"

                "ld1    {v0.4s}, [%3], #16  \n"
                "ld1    {v1.4s}, [%4], #16  \n"
                "st1    {v0.4s, v1.4s}, [%0], #32  \n"

                "ld1    {v0.4s}, [%5], #16  \n"
                "ld1    {v1.4s}, [%6], #16  \n"
                "st1    {v0.4s, v1.4s}, [%0], #32  \n"

                "ld1    {v0.4s}, [%7], #16  \n"
                "ld1    {v1.4s}, [%8], #16  \n"
                "st1    {v0.4s, v1.4s}, [%0], #32  \n"

                : "=r"(ktm2),   // %0
                  "=r"(k00),    // %1
                  "=r"(k01),    // %2
                  "=r"(k10),    // %3
                  "=r"(k11),    // %4
                  "=r"(k20),    // %5
                  "=r"(k21),    // %6
                  "=r"(k30),    // %7
                  "=r"(k31)     // %8
                : "0"(ktm2),
                  "1"(k00),
                  "2"(k01),
                  "3"(k10),
                  "4"(k11),
                  "5"(k20),
                  "6"(k21),
                  "7"(k30),
                  "8"(k31)
                : "cc", "memory", "v0", "v1"
            );
#else
            asm volatile(
                "vld1.f32   {d0-d1}, [%1 :128]! \n"
                "vld1.f32   {d2-d3}, [%2 :128]! \n"
                "vst1.f32   {d0-d3}, [%0 :128]! \n"

                "vld1.f32   {d0-d1}, [%3 :128]! \n"
                "vld1.f32   {d2-d3}, [%4 :128]! \n"
                "vst1.f32   {d0-d3}, [%0 :128]! \n"

                "vld1.f32   {d0-d1}, [%5 :128]! \n"
                "vld1.f32   {d2-d3}, [%6 :128]! \n"
                "vst1.f32   {d0-d3}, [%0 :128]! \n"

                "vld1.f32   {d0-d1}, [%7 :128]! \n"
                "vld1.f32   {d2-d3}, [%8 :128]! \n"
                "vst1.f32   {d0-d3}, [%0 :128]! \n"

                : "=r"(ktm2),   // %0
                  "=r"(k00),    // %1
                  "=r"(k01),    // %2
                  "=r"(k10),    // %3
                  "=r"(k11),    // %4
                  "=r"(k20),    // %5
                  "=r"(k21),    // %6
                  "=r"(k30),    // %7
                  "=r"(k31)     // %8
                : "0"(ktm2),
                  "1"(k00),
                  "2"(k01),
                  "3"(k10),
                  "4"(k11),
                  "5"(k20),
                  "6"(k21),
                  "7"(k30),
                  "8"(k31)
                : "cc", "memory", "q0", "q1"
            );
#endif // __aarch64__
#else
                for (int m=0; m<4; m++)
                {
                    ktm2[0 +m] = k00[m];
                    ktm2[4 +m] = k01[m];
                    ktm2[8 +m] = k10[m];
                    ktm2[12+m] = k11[m];
                    ktm2[16+m] = k20[m];
                    ktm2[20+m] = k21[m];
                    ktm2[24+m] = k30[m];
                    ktm2[28+m] = k31[m];
                }

                k00 += 4;
                k01 += 4;
                k10 += 4;
                k11 += 4;
                k20 += 4;
                k21 += 4;
                k30 += 4;
                k31 += 4;
                ktm2 += 32;
#endif // __ARM_NEON
            }
        }

        for (; q<inch; q++)
        {
            const float* k00 = kernel0_tm.row(q);
            const float* k10 = kernel1_tm.row(q);
            const float* k20 = kernel2_tm.row(q);
            const float* k30 = kernel3_tm.row(q);

            for (int r=0; r<16; r++)
            {
#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%1], #16  \n"
                "ld1    {v1.4s}, [%2], #16  \n"
                "st1    {v0.4s, v1.4s}, [%0], #32  \n"

                "ld1    {v0.4s}, [%3], #16  \n"
                "ld1    {v1.4s}, [%4], #16  \n"
                "st1    {v0.4s, v1.4s}, [%0], #32  \n"

                : "=r"(ktm2),   // %0
                  "=r"(k00),    // %1
                  "=r"(k10),    // %2
                  "=r"(k20),    // %3
                  "=r"(k30)     // %4
                : "0"(ktm2),
                  "1"(k00),
                  "2"(k10),
                  "3"(k20),
                  "4"(k30)
                : "cc", "memory", "v0", "v1"
            );
#else
            asm volatile(
                "vld1.f32   {d0-d1}, [%1 :128]! \n"
                "vld1.f32   {d2-d3}, [%2 :128]! \n"
                "vst1.f32   {d0-d3}, [%0 :128]! \n"

                "vld1.f32   {d0-d1}, [%3 :128]! \n"
                "vld1.f32   {d2-d3}, [%4 :128]! \n"
                "vst1.f32   {d0-d3}, [%0 :128]! \n"

                : "=r"(ktm2),   // %0
                  "=r"(k00),    // %1
                  "=r"(k10),    // %2
                  "=r"(k20),    // %3
                  "=r"(k30)     // %4
                : "0"(ktm2),
                  "1"(k00),
                  "2"(k10),
                  "3"(k20),
                  "4"(k30)
                : "cc", "memory", "q0", "q1"
            );
#endif // __aarch64__
#else
                for (int m=0; m<4; m++)
                {
                    ktm2[0 +m] = k00[m];
                    ktm2[4 +m] = k10[m];
                    ktm2[8 +m] = k20[m];
                    ktm2[12+m] = k30[m];
                }

                k00 += 4;
                k10 += 4;
                k20 += 4;
                k30 += 4;
                ktm2 += 16;
#endif // __ARM_NEON
            }
        }
    }

    #pragma omp parallel for
    for (int p = remain_outch_start; p<outch; p++)
    {
        float* ktm2 = (float*)kernel_tm2.channel(nn_outch) + 8*8 * inch * (p-remain_outch_start);

        const Mat kernel0_tm = kernel_tm.channel(p);

        int q = 0;

        for (; q<inch; q++)
        {
            const float* k00 = kernel0_tm.row(q);

            for (int r=0; r<16; r++)
            {
#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%1], #16  \n"
                "st1    {v0.4s}, [%0], #16  \n"
                : "=r"(ktm2),   // %0
                  "=r"(k00)     // %1
                : "0"(ktm2),
                  "1"(k00)
                : "cc", "memory", "v0"
            );
#else
            asm volatile(
                "vld1.f32   {d0-d1}, [%1 :128]! \n"
                "vst1.f32   {d0-d1}, [%0 :128]! \n"
                : "=r"(ktm2),   // %0
                  "=r"(k00)     // %1
                : "0"(ktm2),
                  "1"(k00)
                : "cc", "memory", "q0"
            );
#endif // __aarch64__
#else
                for (int m=0; m<4; m++)
                {
                    ktm2[m] = k00[m];
                }

                k00 += 4;
                ktm2 += 4;
#endif // __ARM_NEON
            }
        }
    }
}
}
