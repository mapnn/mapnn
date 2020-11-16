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
#include "MVN.h"
#include <math.h>
void RefMVN::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    MVN mvn(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    //bool normalize_variance = true;
    //bool across_channels = true;
    float eps = 1e-5;

    int size = input.w * input.h;
    if (mvn.normalize_variance)
    {
        Tensor sqsum(1, 1, input.c, FLOAT);
        for (int q=0; q<input.c; q++) {
            const float* ptr = input.data + input.hw*q;
            float s = 0.f;
            for (int i=0; i<size; i++) {
                s += ptr[i] * ptr[i];
            }
            sqsum[q] = s;
        }

        if (mvn.across_channels) {
            float sqmean = 0.f;
            for (int q=0; q<input.c; q++) {
                sqmean += sqsum[q];
            }
            sqmean = sqmean / (input.c * size);

            float norm_var = sqrt(sqmean) + eps;
            float norm_var_inv = 1.f / norm_var;

            for (int q=0; q<output.c; q++) {
                const float* ptr = input.data + input.hw*(q);
                float* outptr = output.data + output.hw*(q);
                for (int i=0; i<size; i++) {
                    outptr[i] = ptr[i] * norm_var_inv;
                }
            }
        }
        else {
            for (int q=0; q<input.c; q++) {
                const float* ptr = input.data + input.hw*(q);
                float* outptr = output.data + output.hw*(q);
                float sqmean = sqsum[q] / size;
                float norm_var = sqrt(sqmean) + eps;
                float norm_var_inv = 1.f / norm_var;
                for (int i=0; i<size; i++) {
                    outptr[i] = ptr[i] * norm_var_inv;
                }
            }
        }
    }
    else {
        Tensor sum(1, 1, input.c, FLOAT);
        for (int q=0; q<input.c; q++) {
            const float* ptr = input.data + input.hw*(q);
            for (int i=0; i<size; i++) {
                sum[q] += ptr[i];
            }
        }
        if (mvn.across_channels) {
            float mean = 0.f;
            for (int q=0; q<input.c; q++) {
                mean += sum[q];
            }
            mean = mean / (input.c * size);
            for (int q=0; q<input.c; q++) {
                const float* ptr = input.data + input.hw*(q);
                float* outptr = output.data + output.hw*(q);
                for (int i=0; i<size; i++) {
                    outptr[i] = ptr[i] - mean;
                }
            }
        }
        else {
            for (int q=0; q<input.c; q++) {
                const float* ptr = input.data + input.hw*(q);
                float* outptr = output.data + output.hw*(q);
                float mean = sum[q] / size;
                for (int i=0; i<size; i++) {
                    outptr[i] = ptr[i] - mean;
                }
            }
        }
    }
}
