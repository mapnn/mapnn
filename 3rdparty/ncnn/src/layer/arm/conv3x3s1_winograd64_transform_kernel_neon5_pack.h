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
static void conv3x3s1_winograd64_transform_kernel_neon5_pack(const Mat& kernel_tm, Mat& kernel_tm2, int inch, int outch)
{
    // optimized layout for winograd5
    // interleave weights
//     Mat kernel_tm2(8*8, inch, outch);
//     Mat kernel_tm2(inch, 64, outch);
    int p=0;
#if __aarch64__
    for (; p+7<outch; p+=8)
    {
        const Mat kernel0_tm = kernel_tm.channel(p);
        const Mat kernel1_tm = kernel_tm.channel(p+1);
        const Mat kernel2_tm = kernel_tm.channel(p+2);
        const Mat kernel3_tm = kernel_tm.channel(p+3);
        const Mat kernel4_tm = kernel_tm.channel(p+4);
        const Mat kernel5_tm = kernel_tm.channel(p+5);
        const Mat kernel6_tm = kernel_tm.channel(p+6);
        const Mat kernel7_tm = kernel_tm.channel(p+7);

        Mat ktm2 = kernel_tm2.channel(p/8);

        for (int r=0; r<64; r++)
        {
            float* ktm2p = ktm2.row(r);

            for (int q=0; q<inch; q++)
            {
                const float* ktm0_0 = kernel0_tm.row(q);
                const float* ktm1_0 = kernel1_tm.row(q);
                const float* ktm2_0 = kernel2_tm.row(q);
                const float* ktm3_0 = kernel3_tm.row(q);
                const float* ktm4_0 = kernel4_tm.row(q);
                const float* ktm5_0 = kernel5_tm.row(q);
                const float* ktm6_0 = kernel6_tm.row(q);
                const float* ktm7_0 = kernel7_tm.row(q);

                ktm2p[0] = ktm0_0[r];
                ktm2p[1] = ktm1_0[r];
                ktm2p[2] = ktm2_0[r];
                ktm2p[3] = ktm3_0[r];
                ktm2p[4] = ktm4_0[r];
                ktm2p[5] = ktm5_0[r];
                ktm2p[6] = ktm6_0[r];
                ktm2p[7] = ktm7_0[r];

                ktm2p += 8;
            }
        }
    }
#endif // __aarch64__
    for (; p+3<outch; p+=4)
    {
        const Mat kernel0_tm = kernel_tm.channel(p);
        const Mat kernel1_tm = kernel_tm.channel(p+1);
        const Mat kernel2_tm = kernel_tm.channel(p+2);
        const Mat kernel3_tm = kernel_tm.channel(p+3);

#if __ARM_NEON && __aarch64__
        Mat ktm2 = kernel_tm2.channel(p/8+(p%8)/4);
#else
        Mat ktm2 = kernel_tm2.channel(p/4);
#endif

        for (int r=0; r<64; r++)
        {
            float* ktm2p = ktm2.row(r);

            for (int q=0; q<inch; q++)
            {
                const float* ktm0_0 = kernel0_tm.row(q);
                const float* ktm1_0 = kernel1_tm.row(q);
                const float* ktm2_0 = kernel2_tm.row(q);
                const float* ktm3_0 = kernel3_tm.row(q);

                ktm2p[0] = ktm0_0[r];
                ktm2p[1] = ktm1_0[r];
                ktm2p[2] = ktm2_0[r];
                ktm2p[3] = ktm3_0[r];

                ktm2p += 4;
            }
        }
    }
    for (; p<outch; p++)
    {
        const Mat kernel0_tm = kernel_tm.channel(p);

#if __ARM_NEON && __aarch64__
        Mat ktm2 = kernel_tm2.channel(p/8+(p%8)/4+p%4);
#else
        Mat ktm2 = kernel_tm2.channel(p/4+p%4);
#endif

        for (int r=0; r<64; r++)
        {
            float* ktm2p = ktm2.row(r);

            for (int q=0; q<inch; q++)
            {
                const float* ktm0_0 = kernel0_tm.row(q);

                ktm2p[0] = ktm0_0[r];

                ktm2p += 1;
            }
        }
    }
}
}
