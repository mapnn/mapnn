/* Copyright 2020 The Mapnn Team. All Rights Reserved. 
 *                                                                            
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *                                                                            
 *     http://www.apache.org/licenses/LICENSE-2.0
 *                                                                            
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "reference.h"
namespace mapnn {
void RefUpsample::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Upsample upsample(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    output.c = input.c;
    output.h = upsample.height;
    output.w = upsample.width;
    if(upsample.height == 0 || upsample.width == 0) {
        output.h = input.h * upsample.height_scale;
        output.w = input.w * upsample.width_scale;
    }
    if(!ins[1].empty() && ins[1].size() == 4) {
        L111W scale(ins[1]); 
        output.h = scale.data[2];
        output.w = scale.data[3];
    }
}
void RefUpsample::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Upsample upsample(op);
    L1CHW output(out); 
    L1CHW input(ins[0]); 

    if (upsample.resize_mode == Upsample::NEAREST) {   
        const float hs = (float)input.h / output.h;
        const float ws = (float)input.w / output.w;
        for (int c = 0; c < input.c; c++) {   
            const float* ptr = input.data + c  * input.hw;
            float* outptr = output.data + c  * output.hw;
            for (int h = 0; h < output.h; h++) {   
                int in_y = std::min((int)(h * hs), (input.h - 1));
                for (int w = 0; w < output.w; w++) {
                    int in_x = std::min((int)(w * ws), (input.w - 1));
                    *outptr++ = ptr[in_y * input.w + in_x];
                }
            }
        }
    }
    else {
       /* int* buf = new int[outw + outh + outw * 2 + outh * 2];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
        float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

        linear_coeffs(w, outw, xofs, alpha, align_corner);
        linear_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; ++q)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
        }
        */
    }
}
}
