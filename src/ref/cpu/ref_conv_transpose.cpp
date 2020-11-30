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
void RefConvTranspose::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    L111W weight(ins[1]); 
    L111W bias(ins[2]);
    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    output.c = conv.outch;
    output.h = (input.h - 1) * conv.hstride + extented_filter_h;
    output.w = (input.w - 1) * conv.wstride + extented_filter_w;
}
void RefConvTranspose::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    L111W weight(ins[1]); 
    L111W bias(ins[2]);
    const int maxk = conv.wkernel * conv.hkernel;
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = output.w * conv.hdilation - conv.wkernel* conv.wdilation;
        for (int i = 0; i < conv.hkernel; i++) {
            for (int j = 0; j < conv.wkernel; j++) {
                space_ofs[p1] = p2;
                p1++;
                p2 += conv.wdilation;
            }
            p2 += gap;
        }
    }
    int input_c_g = input.c / conv.g;
    int output_c_g = output.c / conv.g;
    for (int g=0; g<conv.g; g++) {
        for (int o=0; o < output_c_g; o++) {
            float* out = output.data + (g*output_c_g+o)*output.hw;
            const float* weight_data_ptr = weight.data + maxk * input_c_g * output_c_g * g;
            float bias_v = (bias.empty())?0.f:bias[g * output_c_g + o];
            for (int i = 0; i < output.hw; i++) {
                out[i] =  bias_v;
            }
            for (int h = 0; h < input.h; h++) {
                for (int w = 0; w < input.w; w++) {
                    float* outptr = out + h * output.w * conv.hstride + w * conv.wstride;
                    const float* kptr = weight_data_ptr + maxk * input_c_g * o;
                    for (int c=0; c<input_c_g; c++) {
                        const float* m = input.data + (input_c_g * g + c) * input.hw ;
                        float val = *(m + h * input.w + w);
                        for (int k = 0; k < maxk; k++) {
                            outptr[space_ofs[k]] += val * kptr[k];
                        }
                        kptr += maxk;
                    }
                }
            }
        }
    }
}
}
