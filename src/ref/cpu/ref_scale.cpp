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
void RefScale::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    L111W scale(ins[1]); 
    L111W bias(ins[2]); 
    for(int q = 0; q < output.c; q++) {
        const float* ptr = input.data + input.hw*q;
        float* outptr = output.data + output.hw*q;
        float s = scale.data[q];
        float b = bias.data[q];
        int size = output.hw;
        for (int i=0; i<size; i++) {
            *outptr++ = *ptr++ * s + b;
        }
    }

}
}
