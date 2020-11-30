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

DECLARE_FUSION_MAP(map_conv_bn_scale);

namespace mapnn {
inline bool map_conv_bn_scale::request(Operator& op) {
    return op.type == OpType_Conv;
}
inline bool map_conv_bn_scale::run(Graph* graph, Node* node) {
    std::string conv_name = node->name();
    Operator conv_op = node->getOp();
    Conv conv(conv_op);

    if(node->sik_num() != 1)  return false;
    Node* bn_node = node->sik_get(0);
    Operator bn_op = bn_node->getOp();
    if(bn_op.type != OpType_BatchNormalization) return false;
    Node* scale = bn_node->cst_get(0);
    Node* bias = bn_node->cst_get(1);
    Node* mean = bn_node->cst_get(2);
    Node* var = bn_node->cst_get(3);
    Tensor st = scale->getTensor();
    Tensor bt = bias->getTensor();
    Tensor mt = mean->getTensor();
    Tensor vt = var->getTensor();

    if(bn_node->sik_num() != 1)  return false;
    Node* scale_node = bn_node->sik_get(0);
    Operator scale_op = scale_node->getOp();
    if(scale_op.type != OpType_Scale) return false;
    Node* scale_scale = scale_node->cst_get(0);
    Node* scale_bias = scale_node->cst_get(1);
    Tensor sst = scale_scale->getTensor();
    Tensor sbt = scale_bias->getTensor();

    Node* conv_weight = node->cst_get(0);
    Node* conv_bias = node->cst_get(1);
    Tensor cwt = conv_weight->getTensor();
    Tensor cbt;
    if(conv_bias == NULL) {
        cbt = Tensor(conv.outch, 1, 1, 1, FLOAT);
        cbt.fill(0);
        conv_bias = graph->createNode(conv_name+"_b", cbt);
        node->cst_insert(conv_bias, POSITION_BACK);
    }
    cbt = conv_bias->getTensor();

    float* p_conv_weight = cwt.data();
    float* p_conv_bias   = cbt.data();
    float* p_scale  = st.data();
    float* p_bias   = bt.data();
    float* p_mean   = mt.data();
    float* p_var    = vt.data();
    float* p_scale_scale  = sst.data();
    float* p_scale_bias   = sbt.data();
    for(int o = 0; o < conv.outch; o++) {
        float w_scale = 1.0f/std::sqrt(p_var[o] + 1e-5)*p_scale[o]*p_scale_scale[o];
        float b_bias =  - p_mean[o] * w_scale + p_bias[o] + p_scale_bias[o];
        int size = conv.inch * conv.hkernel * conv.wkernel / conv.g;
        for(int i = 0; i < size; i++) {
            *p_conv_weight = *p_conv_weight * w_scale;
            p_conv_weight++;
        }
        *p_conv_bias   = *p_conv_bias + b_bias;
        p_conv_bias++;
    }
    bn_node->sik_remove(POSITION_FRONT);
    graph->releaseNode(scale_node);
    graph->releaseNode(scale_scale);
    graph->releaseNode(scale_bias);

    node->sik_remove(POSITION_FRONT);
    graph->releaseNode(bn_node);
    graph->releaseNode(mean);
    graph->releaseNode(var);
    graph->releaseNode(scale);
    graph->releaseNode(bias);
    return false;
}
}
