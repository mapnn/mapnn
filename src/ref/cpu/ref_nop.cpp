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
void RefNop::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    output.c = input.c;
    output.h = input.h;
    output.w = input.w;
}
void RefNop::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    const float* inptr = input.data;
    float* outptr = output.data;
    memcpy(outptr, inptr, input.chw*4);
}
