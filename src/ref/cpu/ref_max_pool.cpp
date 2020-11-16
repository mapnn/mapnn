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
void RefMaxPool::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Pool pool(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    output.c = input.c;
    output.h = (input.h - pool.hkernel) / pool.hstride + 1;
    output.w = (input.w - pool.wkernel) / pool.wstride + 1;
}
void RefMaxPool::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Pool pool(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    const int maxk = pool.wkernel * pool.hkernel;
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = input.w - pool.wkernel;
        for (int i = 0; i < pool.hkernel; i++) {
            for (int j = 0; j < pool.wkernel; j++) {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }
    for (int c=0; c<output.c; c++) {
        const float* m = input.data + c * input.h*input.w;
        float* outptr = output.data + c * output.h*output.w;
        for (int h = 0; h < output.h; h++) {
            for (int w = 0; w < output.w; w++) {
                const float* sptr = m + h*input.w*pool.hstride + w*pool.wstride;
                float max = sptr[0];
                for (int k = 0; k < maxk; k++) {
                    float val = sptr[ space_ofs[k] ];
                    max = std::max(max, val);
                }
                outptr[w] = max;
            }
            outptr += output.w;
        }
    }
#ifdef __OP_DEBUG__
    printf("\trun maxpool\n");
    printf("\tinput: %d %d %d   %p\n", input.c, input.h, input.w, input.data);
    printf("\toutput: %d %d %d  %p\n", output.c, output.h, output.w, output.data);
#endif
}
