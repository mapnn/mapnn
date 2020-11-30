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
#include "crop.h"

DECLARE_OPTIMAL_MAP(map_conv3x3s1_winograd64_neon5);

namespace mapnn {
inline bool map_conv3x3s1_winograd64_neon5::request(Operator& op) {
    return  op.type == OpType_Conv   &&
        op[Conv::WKERNEL].i == 3     &&
        op[Conv::HKERNEL].i == 3     &&
        op[Conv::WSTRIDE].i == 1     &&
        op[Conv::HSTRIDE].i == 1     &&
        op[Conv::WDILATION].i == 1   &&
        op[Conv::HDILATION].i == 1   &&
        op[Conv::GROUP].i == 1       &&
        op.iw >= 15 && op.ih >= 15          // error for iw|ih<15
        ;
}
inline bool map_conv3x3s1_winograd64_neon5::run(Graph* graph, Node* node) {
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
    Node* pad = graph->createNode(nodename+"_pad_", op_pad);
    Node* GgG = graph->createNode(nodename+"_GgG_", op);
    Node* cro = graph->createNode(nodename+"_cro_", op_crop);

    pad->setKernel(new RefPad());
    GgG->setKernel(new ncnn_conv3x3s1_winograd64_transform_kernel_neon5());
    node->setKernel(new ncnn_conv3x3s1_winograd64_neon5());
    cro->setKernel(new RefCrop());
    node->src_insert(pad);
    node->cst_insert(GgG);
    node->sik_insert(cro);
    return true;
}
}
