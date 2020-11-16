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
static void conv_weight_pack1x4_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt,
        int num_output, int num_input, int maxk)
{
    Mat weight_data_r2 = bottom_blob;
    Mat weight_data_pack4to1 = top_blob;
    for (int q=0; q<num_output; q++)
    {
        const Mat k0 = weight_data_r2.channel(q);
        Mat g0 = weight_data_pack4to1.channel(q);

        for (int p=0; p+3<num_input; p+=4)
        {
            const float* k00 = k0.row(p);
            const float* k01 = k0.row(p+1);
            const float* k02 = k0.row(p+2);
            const float* k03 = k0.row(p+3);

            float* g00 = g0.row(p/4);

            for (int k=0; k<maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k01[k];
                g00[2] = k02[k];
                g00[3] = k03[k];

                g00 += 4;
            }
        }
    }

}
}
