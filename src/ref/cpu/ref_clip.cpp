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
void RefClip::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Clip clip(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    int size = input.hw;

    for (int q=0; q<input.c; q++) {
        const float* ptr = input.data + input.hw*q;
        float* outptr = output.data + output.hw*q;

        for (int i=0; i<size; i++) {
            if (ptr[i] < clip.min) outptr[i] = clip.min;
            else if (ptr[i] > clip.max) outptr[i] = clip.max;
            else outptr[i] = ptr[i];
        }
    }
}
}
