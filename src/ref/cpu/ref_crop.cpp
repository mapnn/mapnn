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
void RefCrop::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Crop crop(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    output.c = input.c;
    output.h = input.h - crop.hcrop0 - crop.hcrop1;
    output.w = input.w - crop.wcrop0 - crop.wcrop1;
    if(!ins[1].empty()) {
        L1CHW shape(ins[1]); 
        output.c = shape.c;
        output.h = shape.h;
        output.w = shape.w;
    }
}
void RefCrop::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Crop crop(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    float* outptr = output.data;;
    for(int c = 0; c < output.c; c++) {
        float* ptr = input.data + input.hw*c + input.w*crop.hcrop0 + crop.wcrop0;
        for(int h = 0; h < output.h; h++) {
            memcpy(outptr, ptr, output.w*4);
            outptr += output.w;
            ptr += input.w;
        }
    }
}
}
