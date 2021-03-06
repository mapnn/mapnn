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
void RefDiv::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW A(ins[0]); 
    L1CHW B(ins[1]); 
    L1CHW output(out); 
    if(B.c!=A.c||B.h!=A.h||B.w!=A.w) {
        LOGE("Div not support broadcasting\n");
    }
    for(int q = 0; q < output.c; q++) {
        const float* ptr = A.data + B.hw*q;
        const float* pptr = A.data + B.hw*q;
        float* outptr = output.data + output.hw*q;
        int size = output.hw;
        for (int i=0; i<size; i++) {
            *outptr++ = *ptr++ / *pptr++;
        }
    }
}
}
