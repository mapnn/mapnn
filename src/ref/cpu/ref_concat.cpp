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
void RefConcat::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    output.c = input.c;
    output.h = input.h;
    output.w = input.w;
    for(size_t i = 1; i < ins.size(); i++) {
        if(ins[i].empty()) continue;
        L1CHW input(ins[i]); 
        output.c += input.c;
    }
}
void RefConcat::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW output(out); 
    float* outptr = output.data;
    for(size_t i = 0; i < ins.size(); i++) {
        if(ins[i].empty()) continue;
        L1CHW input(ins[i]); 
        const float* inptr = input.data; 
        memcpy(outptr, inptr, input.chw*sizeof(float));
        outptr += input.chw;
    }
}
}
