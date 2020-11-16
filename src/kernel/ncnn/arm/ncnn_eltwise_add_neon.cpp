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

#include "ncnn_kernel.h"
#include <layer/arm/eltwise_add_arm.h>
void ncnn_eltwise_add_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW A(ins[0]); 
    L1CHW B(ins[1]); 
    L1CHW output(out); 
    const ncnn::Mat bottom_blob(A.w, A.h, A.c, A.data, 4u, 1);
    const ncnn::Mat bottom_blob1(B.w, B.h, B.c, B.data, 4u, 1);
    ncnn::Mat top_blob(output.w, output.h, output.c, output.data, 4u, 1);
    ncnn::Option opt;
    ncnn::eltwise_add_arm(bottom_blob, bottom_blob1, top_blob, opt);
}
