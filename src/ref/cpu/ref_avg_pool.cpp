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
void RefAvgPool::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Pool pool(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    output.c = input.c;
    output.h = (input.h - pool.hkernel) / pool.hstride + 1;
    output.w = (input.w - pool.wkernel) / pool.wstride + 1;
}
void RefAvgPool::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Pool pool(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    const int maxk = pool.wkernel*pool.hkernel;
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
        const float* m = input.data + input.hw*c;
        float* outptr = output.data + output.hw*c;
        for (int h = 0; h < output.h; h++) {
            for (int w = 0; w < output.w; w++) {
                const float* sptr = m + input.w*h*pool.hstride + w*pool.wstride;
                float sum = 0.f;
                for (int k = 0; k < maxk; k++) {
                    sum += sptr[ space_ofs[k] ];
                }
                outptr[w] = sum / maxk;
            }
            outptr += output.w;
        }
        if (!pool.count_pad) {
            int wtail = (input.w - pool.wkernel) % pool.wstride;
            int htail = (input.h - pool.hkernel) % pool.hstride;
            int wtailpad = (wtail != 0)?pool.wstride-wtail:0;
            int htailpad = (htail != 0)?pool.hstride-htail:0;
            wtailpad = - wtailpad;
            htailpad = - htailpad;

            if (pool.hpad0 != 0) {
                const float scale = (float)pool.hkernel / (pool.hkernel - pool.hpad0);
                float* outptr = output.data + output.hw*c;
                for (int i = 0; i < output.w; i++) {
                    outptr[i] *= scale;
                }
            }
            if (pool.hpad1 + htailpad != 0) {
                const float scale = (float)pool.hkernel / (pool.hkernel - pool.hpad1 - htailpad);
                float* outptr = output.data + output.hw*c + output.w*(output.h-1);
                for (int i = 0; i < output.w; i++) {
                    outptr[i] *= scale;
                }
            }
            if (pool.wpad0 != 0) {
                const float scale = (float)pool.wkernel / (pool.wkernel - pool.wpad0);
                float* outptr = output.data + output.hw*c;
                for (int i = 0; i < output.h; i++) {
                    *outptr *= scale;
                    outptr += output.w;
                }
            }
            if (pool.wpad1 + wtailpad != 0) {
                const float scale = (float)pool.wkernel / (pool.wkernel - pool.wpad1 - wtailpad);
                float* outptr = output.data + output.hw*c + output.w - 1;
                for (int i = 0; i < output.h; i++) {
                    *outptr *= scale;
                    outptr += output.w;
                }
            }
        }
    }
}
}
