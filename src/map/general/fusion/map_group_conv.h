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

DECLARE_FUSION_MAP(map_group_conv);

namespace mapnn {
inline bool map_group_conv::request(Operator& op) {
    return op.type == OpType_Conv    &&
        op[Conv::GROUP].i > 1        &&
        !(op[Conv::OUTCH].i == op[Conv::GROUP].i && op[Conv::INCH].i == op[Conv::GROUP].i);
}
inline bool map_group_conv::run(Graph* graph, Node* node) {
    std::string nodename = node->name();
    Operator conv_op = node->getOp();
    Conv conv(conv_op);

    Node* last = node->src_get(0);
    Node* weight = node->cst_get(0);
    Node* bias = node->cst_get(1);
    Operator op_concat(OpType_Concat);
    node->setKernel(new RefConcat());
    node->src_reduce(last);
    last->sik_reduce(node);
    node->cst_reduce(weight);
    weight->sik_reduce(node);
    if(bias) {
        node->cst_reduce(bias);
        bias->sik_reduce(node);
    }

    for(int g = 0; g < conv.g; g++) {
        Operator op_slice(OpType_Slice);
        op_slice[Slice::AXIS].i = 1;
        op_slice[Slice::BEGIN].i = g*conv.inch/conv.g;
        op_slice[Slice::END].i = (g+1)*conv.inch/conv.g;
        Operator op_conv = conv_op;
        op_conv[Conv::GROUP].i = 1;
        op_conv[Conv::OUTCH].i = conv.outch / conv.g;
        op_conv[Conv::INCH].i = conv.inch / conv.g;
        Operator op_slicew(OpType_Slice);
        op_slicew[Slice::AXIS].i = 3;
        op_slicew[Slice::BEGIN].i = g*conv.outch*conv.inch*conv.wkernel*conv.hkernel/conv.g/conv.g;
        op_slicew[Slice::END].i = (g+1)*conv.outch*conv.inch*conv.wkernel*conv.hkernel/conv.g/conv.g;
        Node* slice = graph->createNode(nodename+"_s"+std::to_string(g), op_slice);
        Node* slice_w = graph->createNode(nodename+"_sw"+std::to_string(g), op_slicew);
        Node* convolution = graph->createNode(nodename+"_c"+std::to_string(g), op_conv);
        slice->setKernel(new RefSlice());
        slice_w->setKernel(new RefSlice());
        convolution->setKernel(new RefConv());

        node->src_extend(convolution);
        convolution->sik_extend(node);

        slice->sik_extend(convolution);
        convolution->src_extend(slice);

        slice->src_extend(last);
        last->sik_extend(slice);

        convolution->cst_extend(slice_w);
        slice_w->sik_extend(convolution);

        slice_w->src_extend(weight);
        weight->sik_extend(slice_w);

        if(bias) {
            Operator op_sliceb(OpType_Slice);
            op_sliceb[Slice::AXIS].i = 3;
            op_sliceb[Slice::BEGIN].i = g*conv.outch/conv.g;
            op_sliceb[Slice::END].i = (g+1)*conv.outch/conv.g;
            Node* slice_b = graph->createNode(nodename+"_sb"+std::to_string(g), op_sliceb);
            slice_b->setKernel(new RefSlice());

            convolution->cst_extend(slice_b);
            slice_b->sik_extend(convolution);

            slice_b->src_extend(bias);
            bias->sik_extend(slice_b);
        }

    }
    return false;
}
}
