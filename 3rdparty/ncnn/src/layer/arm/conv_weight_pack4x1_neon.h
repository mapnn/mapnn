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
static void conv_weight_pack4x1_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt,
        int num_output, int num_input, int maxk)
{
    Mat weight_data_r2 = bottom_blob;
    Mat weight_data_pack1to4 = top_blob;
    for (int q=0; q+3<num_output; q+=4)
    {
        const Mat k0 = weight_data_r2.channel(q);
        const Mat k1 = weight_data_r2.channel(q+1);
        const Mat k2 = weight_data_r2.channel(q+2);
        const Mat k3 = weight_data_r2.channel(q+3);

        Mat g0 = weight_data_pack1to4.channel(q/4);

        for (int p=0; p<num_input; p++)
        {
            const float* k00 = k0.row(p);
            const float* k10 = k1.row(p);
            const float* k20 = k2.row(p);
            const float* k30 = k3.row(p);

            float* g00 = g0.row(p);

            for (int k=0; k<maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00 += 4;
            }
        }
    }
}
}
