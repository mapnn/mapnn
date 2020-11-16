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
#include <math.h>
void RefBatchnormal::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW input(ins[0]); 
    L111W scale(ins[1]); 
    L111W bias(ins[2]); 
    L111W mean(ins[3]); 
    L111W var(ins[4]); 
    L1CHW output(out); 
    int size = input.h*input.w;;
    for (int c=0; c<input.c; c++) {
        const float* ptr = input.data + size * c;
        float* outptr = output.data + size * c;
        float sqrt_var = 1.f / sqrt(var.data[c] + 1e-5);
        float a = bias.data[c] - scale.data[c] * mean.data[c] * sqrt_var;
        float b = scale.data[c] * sqrt_var;
        for (int i=0; i<size; i++) {
            *outptr++ = b * *ptr++ + a;
        }
    }
}
