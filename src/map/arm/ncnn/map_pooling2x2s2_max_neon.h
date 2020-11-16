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

DECLARE_OPTIMAL_MAP(map_pooling2x2s2_max_neon);

inline bool map_pooling2x2s2_max_neon::request(Operator& op) {
    return op.type == OpType_MaxPool &&
        op[Pool::WKERNEL].i == 2     &&
        op[Pool::HKERNEL].i == 2     &&
        op[Pool::WSTRIDE].i == 2     &&
        op[Pool::HSTRIDE].i == 2;
}
inline bool map_pooling2x2s2_max_neon::run(Graph* graph, Node* node) {
    Operator op = node->getOp();
    node->setKernel(new ncnn_pooling2x2s2_max_neon());
    return true;
}
