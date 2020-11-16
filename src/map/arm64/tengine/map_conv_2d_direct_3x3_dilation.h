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

DECLARE_OPTIMAL_MAP(map_conv_2d_direct_3x3_dilation);

inline bool map_conv_2d_direct_3x3_dilation::request(Operator& op) {
    return op.type == OpType_Conv                       &&
        op[Conv::WKERNEL].i == 3                        &&
        op[Conv::HKERNEL].i == 3                        &&
        op[Conv::WSTRIDE].i == 1                        &&
        op[Conv::HSTRIDE].i == 1                        &&
        op[Conv::WDILATION].i > 1                       &&
        op[Conv::GROUP].i == 1                          &&
        op[Conv::WDILATION].i == op[Conv::HDILATION].i;
}
inline bool map_conv_2d_direct_3x3_dilation::run(Graph* graph, Node* node) {
    std::string nodename = node->name();
    Operator op = node->getOp();
    op[Conv::HPAD0].i = op[Conv::WDILATION].i;
    op[Conv::WPAD0].i = op[Conv::WDILATION].i;
    Operator op_crop(OpType_Crop);
    op_crop[Crop::WCROP1].i = op[Conv::WDILATION].i;
    op_crop[Crop::HCROP1].i = op[Conv::WDILATION].i;
    Node* cro = graph->createNode(nodename+"_cro_", op_crop);
    node->setKernel(new tengine_conv_2d_direct_3x3_dilation(), op);
    node->sik_insert(cro);
    return true;
}
