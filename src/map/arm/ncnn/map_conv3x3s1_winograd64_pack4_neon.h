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

DECLARE_OPTIMAL_MAP(map_conv3x3s1_winograd64_pack4_neon);

inline bool map_conv3x3s1_winograd64_pack4_neon::request(Operator& op) {
    return  op.type == OpType_Conv   &&
        op[Conv::WKERNEL].i == 3     &&
        op[Conv::HKERNEL].i == 3     &&
        op[Conv::WSTRIDE].i == 1     &&
        op[Conv::HSTRIDE].i == 1     &&
        op[Conv::WDILATION].i == 1   &&
        op[Conv::HDILATION].i == 1   &&
        op[Conv::GROUP].i == 1       &&
        op.ic % 4 == 0               &&
        op.oc % 4 == 0;
}
inline bool map_conv3x3s1_winograd64_pack4_neon::run(Graph* graph, Node* node) {
    std::string nodename = node->name();
    Operator op = node->getOp();
    int gap_w = (6 - op.ow % 6)%6;
    int gap_h = (6 - op.oh % 6)%6;
    Operator op_pad(OpType_Pad);
    op_pad[Pad::WPAD1].i = gap_w;
    op_pad[Pad::HPAD1].i = gap_h;
    Operator op_crop(OpType_Crop);
    op_crop[Crop::WCROP1].i = gap_w;
    op_crop[Crop::HCROP1].i = gap_h;
    Node* i_pack = graph->createNode(nodename+"_i_", op);
    Node* pad = graph->createNode(nodename+"_p_", op_pad);
    Node* GgG = graph->createNode(nodename+"_w_", op);
    Node* cro = graph->createNode(nodename+"_c_", op_crop);
    Node* o_pack = graph->createNode(nodename+"_o_", op);

    pad->setKernel(new RefPad());
    GgG->setKernel(new ncnn_conv3x3s1_winograd64_transform_kernel_pack4_neon());
    node->setKernel(new ncnn_conv3x3s1_winograd64_pack4_neon());
    cro->setKernel(new RefCrop());
    i_pack->setKernel(new RefCHW1ToCHW4());
    o_pack->setKernel(new RefCHW4ToCHW1());
    node->src_insert(i_pack);
    node->src_insert(pad);
    node->cst_insert(GgG);
    node->sik_insert(o_pack);
    node->sik_insert(cro);
    return true;
}
