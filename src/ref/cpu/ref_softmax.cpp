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
void RefSoftmax::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    int size = input.chw;
    float max = -0x1.p31;
    float sum = 0;

    const float* ptr = input.data;
    for (int i=0; i<size; i++) {
            max = std::max(max, ptr[i]);
    }

    ptr = input.data;
    float* outptr = output.data;
    for (int i=0; i<size; i++) {
            outptr[i] = exp(ptr[i] - max);
    }

    outptr = output.data;
    for (int i=0; i<size; i++) {
            sum += outptr[i];
    }

    outptr = output.data;
    for (int i=0; i<size; i++) {
            outptr[i] /= sum;
    }
#ifdef __OP_DEBUG__
    printf("\tSoftmax: run\n");
    printf("\tinput: %d %d %d   %p\n", input.c, input.h, input.w, input.data);
    printf("\toutput: %d %d %d  %p\n", output.c, output.h, output.w, output.data);
#endif
}
