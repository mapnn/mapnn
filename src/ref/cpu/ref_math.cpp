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
static double fneg(double x) { return -x; }
void RefMath::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    double (*m) (double x);
    switch(op.type) {
        case OpType_Neg: m = fneg; break;
        case OpType_Abs: m = fabs; break;
        case OpType_Acos: m = acos; break;
        case OpType_Acosh: m = acosh; break;
        case OpType_Asin: m = asin; break;
        case OpType_Asinh: m = asinh; break;
        case OpType_Atan: m = atan; break;
        case OpType_Atanh: m = atanh; break;
        case OpType_Ceil: m = ceil; break;
        case OpType_Cos: m = cos; break;
        case OpType_Cosh: m = cosh; break;
        case OpType_Exp: m = exp; break;
        case OpType_Floor: m = floor; break;
        case OpType_Log: m = log; break;
        case OpType_Sin: m = sin; break;
        case OpType_Sinh: m = sinh; break;
        case OpType_Sqrt: m = sqrt; break;
        case OpType_Tan: m = tan; break;
        case OpType_Tanh: m = tanh; break;
        default: return;

    }
    const float* ptr = input.data;
    float* outptr = output.data;
    for(int c = 0; c < output.c; c++) {
        for(int h = 0; h < output.h; h++) {
            for(int w = 0; w < output.w; w++) {
                *outptr++ = m(*ptr++);
            }
        }
    }
}
}
