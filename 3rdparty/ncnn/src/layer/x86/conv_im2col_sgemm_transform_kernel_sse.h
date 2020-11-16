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
static void conv_im2col_sgemm_transform_kernel_sse(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_size)
{
    const float* kernel = _kernel;

    // kernel memory packed 4 x 4
    kernel_tm.create(4*kernel_size, inch, outch/4 + outch%4);
    
    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 2;
    remain_outch_start = nn_outch << 2;

    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 4;

        const float* k0 = kernel + (p+0)*inch*kernel_size;
        const float* k1 = kernel + (p+1)*inch*kernel_size;
        const float* k2 = kernel + (p+2)*inch*kernel_size;
        const float* k3 = kernel + (p+3)*inch*kernel_size;

        float* ktmp = kernel_tm.channel(p/4);

        for (int q=0; q<inch*kernel_size; q++)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp += 4;

            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
        }
    }
    
    for (int p=remain_outch_start; p<outch; p++)
    {
        const float* k0 = kernel + (p+0)*inch*kernel_size;

        float* ktmp = kernel_tm.channel(p/4 + p%4);

        for (int q=0; q<inch*kernel_size; q++)
        {
            ktmp[0] = k0[0];
            ktmp++;
            k0++;
        }
    }
}
}
