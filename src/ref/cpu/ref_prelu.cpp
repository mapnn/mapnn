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
void RefPRelu::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW input(ins[0]); 
    L111W slope(ins[1]); 
    L1CHW output(out); 
    const float* ptr = input.data;
    float* outptr = output.data;
    int size = output.hw;
    for(int c = 0; c < output.c; c++) {
        float s = slope.w>1? slope.data[c] : slope.data[0];
        for(int i = 0; i < size; i++) {
            if (*ptr < 0) *outptr = *ptr * s;
            else *outptr = *ptr;
            ptr++;
            outptr++;
        }
    }
}
}
