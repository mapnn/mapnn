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
#include "reference.h"

DECLARE_KERNEL_MAP(map_pad_conv3x3s1_same);

inline bool map_pad_conv3x3s1_same::request(Operator& op) {
    return op.type == OpType_Conv &&
        op[Conv::WKERNEL].i == 3  &&
        op[Conv::HKERNEL].i == 3  &&
        op[Conv::WSTRIDE].i == 1  &&
        op[Conv::HSTRIDE].i == 1  &&
        op[Conv::PADMODE].i == Conv::SAME;
}
inline bool map_pad_conv3x3s1_same::run(Graph* graph, Node* node) {
    std::string nodename = node->name();
    Operator op = node->getOp();
    Operator pad(OpType_Pad);
    pad[Pad::WPAD0].i = op[Conv::WPAD0].i+1;
    pad[Pad::HPAD0].i = op[Conv::HPAD0].i+1;
    pad[Pad::WPAD1].i = op[Conv::WPAD1].i+1;
    pad[Pad::HPAD1].i = op[Conv::HPAD1].i+1;
    op[Conv::WPAD0].i = 0;
    op[Conv::HPAD0].i = 0;
    op[Conv::WPAD1].i = 0;
    op[Conv::HPAD1].i = 0;
    Node* npad = graph->createNode(nodename+"_p", pad);
    node->src_insert(npad);
    node->setKernel(new RefConv(), op);
    return true;
}
