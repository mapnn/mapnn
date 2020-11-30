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

#include "map.h"
#include "tengine_kernel.h"

DECLARE_OPTIMAL_MAP(map_tengine_conv_2d_dw_k5s2);

namespace mapnn {
inline bool map_tengine_conv_2d_dw_k5s2::request(Operator& op) {
    return op.type == OpType_Conv    &&
        op[Conv::WKERNEL].i == 5     &&
        op[Conv::HKERNEL].i == 5     &&
        op[Conv::WSTRIDE].i == 2     &&
        op[Conv::HSTRIDE].i == 2     &&
        op[Conv::WDILATION].i == 1   &&
        op[Conv::HDILATION].i == 1   &&
        op[Conv::GROUP].i > 1        &&
        op[Conv::GROUP].i == op.ic   &&
        op[Conv::GROUP].i == op.oc;
}
inline bool map_tengine_conv_2d_dw_k5s2::run(Graph* graph, Node* node) {
    Operator op = node->getOp();
    node->setKernel(new tengine_conv_2d_dw_k5s2());
    return true;
}
}
