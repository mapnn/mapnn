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

DECLARE_OPTIMAL_MAP(map_tengine_conv_2d_dw);

inline bool map_tengine_conv_2d_dw::request(Operator& op) {
    return op.type == OpType_Conv    &&
        op[Conv::GROUP].i > 1        &&
        op[Conv::GROUP].i == op.ic   &&
        op[Conv::GROUP].i == op.oc;
}
inline bool map_tengine_conv_2d_dw::run(Graph* graph, Node* node) {
    Operator op = node->getOp();
    node->setKernel(new tengine_conv_2d_dw());
    return true;
}
