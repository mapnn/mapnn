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
void RefCHW4ToCHW1::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    LCHW4 input(ins[0]); 
    L1CHW output(out); 
    output.c = op.oc;
    output.h = input.h;
    output.w = input.w4/4;
}
void RefCHW4ToCHW1::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    LCHW4 input(ins[0]); 
    L1CHW output(out); 
    int c4 = op.oc/4;
    int cp = op.oc%4;
    int size = output.hw;
    for (int q=0; q<c4; q++) {
        const float* r0 = input.data + input.hw4*(q);
        float* outptr0 = output.data + output.hw*(q*4);
        float* outptr1 = output.data + output.hw*(q*4+1);
        float* outptr2 = output.data + output.hw*(q*4+2);
        float* outptr3 = output.data + output.hw*(q*4+3);
        int remain = size;
        for (; remain>0; remain--) {
            *outptr0++ = r0[0];
            *outptr1++ = r0[1];
            *outptr2++ = r0[2];
            *outptr3++ = r0[3];
            r0 += 4;
        }
    }
    switch(cp) {
        case 3: {
            const float* r0 = input.data + input.hw4*(c4);
            float* outptr0 = output.data + output.hw*(c4*4);
            float* outptr1 = output.data + output.hw*(c4*4+1);
            float* outptr2 = output.data + output.hw*(c4*4+2);
            int remain = size;
            for (; remain>0; remain--) {
                *outptr0++ = r0[0];
                *outptr1++ = r0[1];
                *outptr2++ = r0[2];
                r0 += 4;
            }
            break;
        }
        case 2: {
            const float* r0 = input.data + input.hw4*(c4);
            float* outptr0 = output.data + output.hw*(c4*4);
            float* outptr1 = output.data + output.hw*(c4*4+1);
            int remain = size;
            for (; remain>0; remain--) {
                *outptr0++ = r0[0];
                *outptr1++ = r0[1];
                r0 += 4;
            }
            break;
        }
        case 1: {
            const float* r0 = input.data + input.hw4*(c4);
            float* outptr0 = output.data + output.hw*(c4*4);
            int remain = size;
            for (; remain>0; remain--) {
                *outptr0++ = r0[0];
                r0 += 4;
            }
            break;
        }
    }
}
