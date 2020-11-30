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
#include "mnn_kernel.h"

DECLARE_OPTIMAL_MAP(map_ConvolutionWinogradF23);

namespace mapnn {
inline bool map_ConvolutionWinogradF23::request(Operator& op) {
    return op.type == OpType_Conv    &&
        op[Conv::WKERNEL].i == 3     &&
        op[Conv::HKERNEL].i == 3     &&
        op[Conv::WSTRIDE].i == 1     &&
        op[Conv::HSTRIDE].i == 1     &&
        op[Conv::WDILATION].i == 1   &&
        op[Conv::HDILATION].i == 1   &&
        op[Conv::GROUP].i == 1       &&
        op.oc % 4 == 0;
}
inline bool map_ConvolutionWinogradF23::run(Graph* graph, Node* node) {
    std::string nodename = node->name();
    Operator op = node->getOp();
    Node* w_pack = graph->createNode(nodename+"_w_", op);
    Node* i_pack = graph->createNode(nodename+"_i4_", op);
    Node* o_pack = graph->createNode(nodename+"_o1_", op);
    node->setKernel(new mnn_ConvolutionWinogradF23());
    w_pack->setKernel(new mnn_transformWeightF23());
    i_pack->setKernel(new RefCHW1ToCHW4());
    o_pack->setKernel(new RefCHW4ToCHW1());
    node->src_insert(i_pack);
    node->cst_insert(w_pack);
    node->sik_insert(o_pack);
    return true;
}
}
