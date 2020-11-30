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
namespace mapnn {
void RefLRN::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    LRN lrn(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    Tensor square(input.c, input.h, input.w, FLOAT);

    int size = input.hw;;
    for (int c=0; c<input.c; c++) {
        const float* ptr = input.data + c * size;
        float* outptr = square.data() + c * size;
        for (int i=0; i<size; i++) {
            outptr[i] = ptr[i] * ptr[i];
        }
    }
    Tensor sum(input.c, input.h, input.w, FLOAT);
    sum.fill(0.f);
    const float alpha_div_size = lrn.alpha / lrn.local_size;
    for (int c=0; c<input.c; c++) {
        float* ssptr = sum.data() + c*size;
        for (int p=c - lrn.local_size / 2; p<=c + lrn.local_size / 2; p++) {
            if (p < 0 || p >= input.c) continue;
            float* sptr = square.data() + size * p;
            for (int i=0; i<size; i++) {
                ssptr[i] += sptr[i];
            }
        }
        const float* ptr = input.data + c * size;
        float* outptr = output.data + c * size;
        for (int i=0; i<size; i++)
        {
            outptr[i] = ptr[i] * pow(lrn.bias + alpha_div_size * ssptr[i], - lrn.beta);
        }
    }
}
}
