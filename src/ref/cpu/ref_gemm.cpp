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
void RefGemm::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    L111W weight(ins[1]); 
    L111W bias(ins[2]);
    output.c = weight.w/input.chw;
    output.h = 1;
    output.w = 1;
}
void RefGemm::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    L111W weight(ins[1]); 
    L111W bias(ins[2]);
    const float* wptr = weight.data;
    float* outptr = output.data;
    int size = input.c*input.h*input.w;
    for(int i = 0; i < output.c; i++) {
        const float* inptr = input.data;
        float sum = bias[i];
        for(int j = 0; j < size; j++) {
            sum += *inptr++ * *wptr++;
        }   
        *outptr++ = sum;
    }
}
