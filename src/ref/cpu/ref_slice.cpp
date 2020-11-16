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
void RefSlice::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Slice slice(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    output.c = input.c;
    output.h = input.h;
    output.w = input.w;

    if(slice.end != slice.begin) {
        switch(slice.axis) {
            case 3:
                output.w = slice.end - slice.begin;
                break;
            case 2:
                output.h = slice.end - slice.begin;
                break;
            case 1:
                output.c = slice.end - slice.begin;
                break;
        }
    }
    else {
        switch(slice.axis) {
            case 3:
                output.w = input.w / slice.max;
                break;
            case 2:
                output.w = input.h / slice.max;
                break;
            case 1:
                output.c = input.c / slice.max;
                break;
        }
    }
}
void RefSlice::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Slice slice(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    const float* inptr = input.data;
    float* outptr = output.data;
    if(slice.end != slice.begin) {
        switch(slice.axis) {
            case 3:
                for(int c = 0; c < input.c; c++) {
                    for(int h = 0; h < input.h; h++) {
                        int strip = output.hw * c + output.w*h + slice.begin;
                        int size = (slice.end-slice.begin);
                        memcpy(outptr, inptr + strip, size*sizeof(float));
                    }
                }
                break;
            case 2:
                for(int c = 0; c < input.c; c++) {
                    int strip = output.hw * c + output.w*slice.begin;
                    int size = output.w*(slice.end-slice.begin);
                    memcpy(outptr, inptr + strip, size*sizeof(float));
                }
                break;
            case 1:
                int strip = output.hw*slice.begin;
                int size = output.hw*(slice.end-slice.begin);
                memcpy(outptr, inptr + strip, size*sizeof(float));
                break;
        }
    }
    else {
        switch(slice.axis) {
            case 3:
                for(int c = 0; c < input.c; c++) {
                    int size = input.h / slice.max;
                    int begin = size * slice.index;
                    for(int h = 0; h < input.h; h++) {
                        int strip = output.hw * c + output.w*h + begin;
                        memcpy(outptr, inptr + strip, size*sizeof(float));
                    }
                }
                break;
            case 2:
                for(int c = 0; c < input.c; c++) {
                    int size = output.w*(input.h / slice.max);
                    int begin = slice.index;
                    int strip = output.hw * c + output.w*begin;
                    memcpy(outptr, inptr + strip, size*sizeof(float));
                }
                break;
            case 1:
                int size = output.hw*(input.c / slice.max);
                int begin = slice.index;
                int strip = output.hw*begin;
                memcpy(outptr, inptr + strip, size*sizeof(float));
                break;
        }

    }
}
