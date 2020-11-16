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
void RefAdd::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW A(ins[0]); 
    L1CHW B(ins[1]); 
    L1CHW output(out); 
    if(A.c == B.c && A.h == B.h && A.w == B.w) {
        output.c = A.c;
        output.h = A.h;
        output.w = A.w;
    }
    else if(A.c == B.c && A.h == B.h && A.w == 1){
        output.c = B.c;
        output.h = B.h;
        output.w = B.w;
    }
    else if(A.c == B.c && A.h == B.h && B.w == 1){
        output.c = A.c;
        output.h = A.h;
        output.w = A.w;
    }
    else if(A.c == B.c && A.h == 1 && A.w == 1){
        output.c = B.c;
        output.h = B.h;
        output.w = B.w;
    }
    else if(A.c == B.c && B.h == 1 && B.w == 1){
        output.c = A.c;
        output.h = A.h;
        output.w = A.w;
    }
    else {
        printf("\tAdd: run\n");
        printf("\tA: %d %d %d   %p\n", A.c, A.h, A.w, A.data);
        printf("\tB: %d %d %d   %p\n", B.c, B.h, B.w, B.data);
        printf("\toutput: %d %d %d   %p\n", output.c, output.h, output.w, output.data);
        printf("error\n");
    }
}
void RefAdd::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW A(ins[0]); 
    L1CHW B(ins[1]); 
    L1CHW output(out); 
    if(A.c == B.c && A.h == B.h && A.w == B.w) {
        int size = output.c * output.h * output.w;
        float* A_ptr = A.data;
        float* B_ptr = B.data;
        float* output_ptr = output.data;
        for(int chw = 0; chw < size; chw++) {
            *output_ptr++ = *A_ptr++ + *B_ptr++;
        }
    }
    else if(A.c == B.c && A.h == B.h && A.w  == 1){
        int size = output.c * output.h;
        float* B_ptr = B.data;
        float* A_ptr = A.data;
        float* output_ptr = output.data;
        for(int ch = 0; ch < size; ch++) {
            for(int w = 0; w < output.w; w++) {
                *output_ptr++ = *A_ptr + *B_ptr++;
            }
            A_ptr++;
        }
    }
    else if(A.c == B.c && A.h == B.h && B.w == 1){
        int size = output.c * output.h;
        float* B_ptr = B.data;
        float* A_ptr = A.data;
        float* output_ptr = output.data;
        for(int ch = 0; ch < size; ch++) {
            for(int w = 0; w < output.w; w++) {
                *output_ptr++ = *A_ptr++ + *B_ptr;
            }
            B_ptr++;
        }
    }
    else if(A.c == B.c && A.h == 1 && A.w == 1){
        int size = output.h * output.w;
        float* B_ptr = B.data;
        float* A_ptr = A.data;
        float* output_ptr = output.data;
        for(int c = 0; c < output.c; c++) {
            for(int hw = 0; hw < size; hw++) {
                *output_ptr++ = *A_ptr + *B_ptr++;
            }
            A_ptr++;
        }
    }
    else if(A.c == B.c && B.h == 1 && B.w == 1){
        int size = output.h * output.w;
        float* B_ptr = B.data;
        float* A_ptr = A.data;
        float* output_ptr = output.data;
        for(int c = 0; c < output.c; c++) {
            for(int hw = 0; hw < size; hw++) {
                *output_ptr++ = *A_ptr++ + *B_ptr;
            }
            B_ptr++;
        }
    }
}
