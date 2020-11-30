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
void RefReshape::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    LNCHW input(ins[0]); 
    LNCHW output(out); 
    Reshape reshape(op);
    if(!ins[1].empty() && reshape.channel == 0) {
        L111W_s64 shape(ins[1]);
        switch(shape.w) {
            case 2:
                if(shape[0] == 0 && shape[1] == -1) {
                    output.n = 1;
                    output.c = 1;
                    output.h = 1;
                    output.w = input.n*input.c*input.h*input.w;
                }
                else {
                    output.n = 1;
                    output.c = 1;
                    output.h = shape[0];
                    output.w = shape[1];
                }
                break;
            case 3:
                if(shape[0] == 0 && shape[2] == -1) {
                    output.n = 1;
                    output.c = 1;
                    output.h = shape[1];
                    output.w = input.n*input.c*input.h*input.w/shape[1];
                }
                else {
                    output.n = 1;
                    output.c = shape[0];
                    output.h = shape[1];
                    output.w = shape[2];
                }
                break;
            case 4:
                if(shape[0] == 0 && shape[3] == -1) {
                    output.n = 1;
                    output.c = shape[1];
                    output.h = shape[2];
                    output.w = input.n*input.c*input.h*input.w/shape[1]/shape[2];
                }
                else {
                    output.n = shape[0];
                    output.c = shape[1];
                    output.h = shape[2];
                    output.w = shape[3];
                }
                break;
            case 5:
                if(shape[0] == 1) {
                    output.n = shape[1];
                    output.c = shape[2];
                    output.h = shape[3];
                    output.w = shape[4];
                }
                else {
                    LOGE("reshape error\n");
                }
            default:
                break;
        }

    }
    else {
        output.n = input.n;
        output.c = input.c;
        output.h = input.h;
        output.w = input.w;
    }
}
void RefReshape::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Reshape reshape(op);
    LNCHW output(out); 
    if(reshape.channel == 0 && reshape.height == 0 && reshape.width == 0) {
        LNCHW input(ins[0]);
        const float* inptr = input.data;
        float* outptr = output.data;
        memcpy(outptr, inptr, input.nchw*4);
    } 
    else {
        LNCHW input(ins[1]);
        const float* inptr = input.data;
        float* outptr = output.data;
        memcpy(outptr, inptr, input.nchw*4);
    }
}
}
