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
void RefPad::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Pad pad(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    output.c = input.c;
    output.h = input.h +  pad.hpad0 +  pad.hpad1;
    output.w = input.w +  pad.wpad0 +  pad.wpad1;
}
void RefPad::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Pad pad(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    float* ptr = input.data;
    float* outptr = output.data;
    const size_t wpad0b = pad.wpad0;
    const size_t wpad1b = pad.wpad1;
    if(pad.mode != Pad::CONSTANT) {
        LOGE("Pad not support the mode.\n");
    }
    if(pad.value == 0) {
        for(int c = 0; c < output.c; c++) {
            for(size_t h = 0; h < pad.hpad0; h++) {
                memset(outptr, 0, output.w*sizeof(float)); 
                outptr+=output.w;
            }
            for(int h = 0; h < input.h; h++) {
                memset(outptr, 0, wpad0b*sizeof(float));
                outptr += wpad0b;
                memcpy(outptr, ptr, input.w*sizeof(float));
                outptr += input.w;
                ptr += input.w;
                memset(outptr, 0, wpad1b*sizeof(float));
                outptr += wpad1b;
            }
            for(size_t h = 0; h < pad.hpad1; h++) {
                memset(outptr, 0, output.w*sizeof(float)); 
                outptr+=output.w;
            }
        }
    } 
    else {
        for(int c = 0; c < output.c; c++) {
            for(int h = 0; h < pad.hpad0; h++) {
                for(int i = 0; i < output.w; i++) *outptr++ = pad.value;
            }
            for(int h = 0; h < input.h; h++) {
                for(int i = 0; i < wpad0b; i++) *outptr++ = pad.value;
                for(int i = 0; i < input.w; i++) *outptr++ = *ptr++;
                for(int i = 0; i < wpad1b; i++) *outptr++ = pad.value;
            }
            for(int h = 0; h < pad.hpad1; h++) {
                for(int i = 0; i < output.w; i++) *outptr++ = pad.value;
            }
        }
    }
}
}
