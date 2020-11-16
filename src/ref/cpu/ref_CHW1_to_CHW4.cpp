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
void RefCHW1ToCHW4::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW input(ins[0]); 
    LCHW4 output(out); 
    output.c = (input.c+3)/4;
    output.h = input.h;
    output.w4 = input.w*4;
}
void RefCHW1ToCHW4::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW input(ins[0]); 
    LCHW4 output(out); 
    int c4 = input.c/4;
    int cp = input.c%4;

    for (int i=0; i<c4; i++) {
        const float* r0 = input.data + input.hw*(i*4);
        const float* r1 = input.data + input.hw*(i*4+1);
        const float* r2 = input.data + input.hw*(i*4+2);
        const float* r3 = input.data + input.hw*(i*4+3);
        float* outptr = output.data + output.hw4*i;
        int remain = input.hw;
        for (; remain>0; remain--) {
            outptr[0] = *r0++;
            outptr[1] = *r1++;
            outptr[2] = *r2++;
            outptr[3] = *r3++;
            outptr += 4;
        }
    }
    switch(cp) {
        case 3: {
            const float* r0 = input.data + input.hw*(c4*4);
            const float* r1 = input.data + input.hw*(c4*4+1);
            const float* r2 = input.data + input.hw*(c4*4+2);
            float* outptr = output.data + output.hw4*c4;
            int remain = input.hw;
            for (; remain>0; remain--) {
                outptr[0] = *r0++;
                outptr[1] = *r1++;
                outptr[2] = *r2++;
                outptr[3] = 0;
                outptr += 4;
            }
            break;
        }
        case 2: {
            const float* r0 = input.data + input.hw*(c4*4);
            const float* r1 = input.data + input.hw*(c4*4+1);
            float* outptr = output.data + output.hw4*c4;
            int remain = input.hw;
            for (; remain>0; remain--) {
                outptr[0] = *r0++;
                outptr[1] = *r1++;
                outptr[2] = 0;
                outptr[3] = 0;
                outptr += 4;
            }
            break;
        }
        case 1: {
            const float* r0 = input.data + input.hw*(c4*4);
            float* outptr = output.data + output.hw4*c4;
            int remain = input.hw;
            for (; remain>0; remain--) {
                outptr[0] = *r0++;
                outptr[1] = 0;
                outptr[2] = 0;
                outptr[3] = 0;
                outptr += 4;
            }
            break;
        }
    }
}
