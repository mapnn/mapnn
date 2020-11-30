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
void RefMax::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    if((L1CHW::check(ins[0])  && L1CHW::check(out))||
            (LCHW4::check(ins[0]) && LCHW4::check(out))) {
        L1CHW input(ins[0]); 
        L1CHW output(out); 
        const float* ptr = input.data;
        float* outptr = output.data;
        int size = output.chw;
        memcpy(outptr, ptr, size*sizeof(float));
        for(size_t n = 1; n < ins.size(); n++) {
            if(ins[n].empty()) continue;
            if(L1CHW::check(ins[n])) {
                L1CHW input(ins[n]); 
                const float* ptr = input.data;
                float* outptr = output.data;
                for(int i = 0; i < size; i++) {
                    if(*outptr<*ptr) *outptr = *ptr;
                    ptr++;
                    outptr++;
                }
            }
            else {
                LOGE("max not support this layout!\n");
            }
        }
    }
    else {
        LOGE("max not support this layout!\n");
    }

}
}
