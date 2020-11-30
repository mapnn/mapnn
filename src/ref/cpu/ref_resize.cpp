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
#include "log.h"
namespace mapnn {
void RefResize::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Resize resize(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 

    switch(resize.coordinate_transformation_mode) {
        case Resize::ASYMMETRIC: {
            if(ins[2].empty()) LOGE("resize error!\n");
            L111W scale(ins[2]);
            output.c = input.c;
            output.h = input.h * scale[2];
            output.w = input.w * scale[3];
            break;
        }
        default:
            LOGE("resize error!\n");
            break;
    }
}
void RefResize::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Resize resize(op);
    L1CHW output(out); 
    L1CHW input(ins[0]); 

    if (resize.mode== Resize::NEAREST) {   
        const float hs = (float)input.h / output.h;
        const float ws = (float)input.w / output.w;
        for (int c = 0; c < input.c; c++) {   
            const float* ptr = input.data + c  * input.hw;
            float* outptr = output.data + c  * output.hw;
            for (int h = 0; h < output.h; h++) {   
                int in_y = std::min((int)(h * hs), (input.h - 1));
                for (int w = 0; w < output.w; w++) {
                    int in_x = std::min((int)(w * ws), (input.w - 1));
                    *outptr++ = ptr[in_y * input.w + in_x];
                }
            }
        }
    }
    else {
    }
}
}
