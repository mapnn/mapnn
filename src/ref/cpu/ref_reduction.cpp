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
void RefReduction::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Reduction reduce(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    if(!reduce.keepdims) {
        output.c = input.c;
        output.h = input.h;
        output.w = input.w;
        if(reduce.c) output.c = 1;
        if(reduce.h) output.h = 1;
        if(reduce.w) output.w = 1;
    }
    else {
        output.c = 1;
        output.h = 1;
        output.w = 1;
        if(!reduce.c) output.c = input.c;
        if(!reduce.h) output.h = input.h;
        if(!reduce.w) output.w = input.w;
    }
}
void RefReduction::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Reduction reduce(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    if(output.c==1&&output.h==1&&output.w==1) {
        float sum = 0;
        float* ptr = input.data;
        for(int c = 0; c < input.c; c++) {
            for(int h = 0; h < input.h; h++) {
                for(int w = 0; w < input.w; w++) {
                    sum += *ptr++; 
                }
            }
        }
        output.data[0] = sum/input.chw;
    }
    if(output.h==1&&output.w==1) {
        for(int c = 0; c < input.c; c++) {
            float sum = 0;
            float* ptr = input.data + c*input.hw;
            for(int h = 0; h < input.h; h++) {
                for(int w = 0; w < input.w; w++) {
                    sum += *ptr++; 
                }
            }
            output.data[c] = sum/input.chw;
        }
    }

}
}
