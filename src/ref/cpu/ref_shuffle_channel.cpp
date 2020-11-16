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
void RefShuffleChannel::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    ShuffleChannel sc(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 

    int chs_per_group = input.c / sc.group;

    if (input.c != chs_per_group * sc.group) {
        printf("error shuffle channel\n");
    }

    for (int i = 0; i < sc.group; i++) {   
        for (int j = 0; j < chs_per_group; j++) {   
            int src_q = chs_per_group * i + j;
            int dst_q = sc.group * j + i;
            float* src = input.data + src_q * input.hw;
            float* dst = output.data + dst_q * output.hw;
            memcpy(dst, src, input.hw);
        }   
    } 
}
