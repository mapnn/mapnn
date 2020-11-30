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

#include "onnx_model.h"

#include <fstream>
#include <climits>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

#include "log.h"
#include "graph.h"
#include "reference.h"

using onnx::ModelProto;

namespace mapnn {
OnnxModel::OnnxModel() {
    model_ = new ModelProto();
}
OnnxModel::~OnnxModel() {
    delete model_;
}
int OnnxModel::load(const char* filepath) {
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        LOGE("open failed %s\n\n", filepath);
        return false;
    }
    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);
    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
    google::protobuf::Message* message = (google::protobuf::Message*)model_;
    bool success = message->ParseFromCodedStream(&codedstr);
    fs.close();
    return success;
}
int OnnxModel::draw(Graph* graph) {
    create_tensor_node_(graph);
    create_operater_node_(graph);
    const onnx::GraphProto& graphP = model_->graph();
    for(int i = 0; i < graphP.node_size(); i++) {
        const onnx::NodeProto& nodeP = graphP.node(i);
        std::string name1 = nodeP.output(0);
        for (int j=0; j < nodeP.input_size(); ++j)
        {
            std::string name2 = nodeP.input(j);
            graph->link(name2.c_str(), name1.c_str());
        }
    }
    return 0;
}
int OnnxModel::create_tensor_node_(Graph* graph) {
    const onnx::GraphProto& graphP = model_->graph();
    for(int n = 0; n < graphP.initializer_size(); n++) {
        const onnx::TensorProto& tensorP = graphP.initializer(n);
        std::string name = tensorP.name();
        const void* data = NULL;
        switch(tensorP.data_type()) {
            case 1:
                data = tensorP.has_raw_data() ? (const float*)tensorP.raw_data().data() : tensorP.float_data().data();
                break;
            case 7:
                data = tensorP.has_raw_data() ? (const int64_t*)tensorP.raw_data().data() : tensorP.int64_data().data();
                break;
            default:
                printf("ERROR!\n");
        }

        int N = 1, C = 1, H = 1, W = 1;
        switch(tensorP.dims_size()) {
            default:
            case 0:
                if (tensorP.has_raw_data()) {
                    const std::string& raw_data = tensorP.raw_data();
                    W = (int)raw_data.size() / 4;
                }
                else if (tensorP.data_type() == 1) {
                    W = tensorP.float_data_size();
                }
                break;
            case 1:
                W = tensorP.dims(0);
                break;
            case 2:
                H = tensorP.dims(0);
                W = tensorP.dims(1);
                break;
            case 3:
                C = tensorP.dims(0);
                H = tensorP.dims(1);
                W = tensorP.dims(2);
                break;
            case 4:
                N = tensorP.dims(0);
                C = tensorP.dims(1);
                H = tensorP.dims(2);
                W = tensorP.dims(3);
                break;
        }
        Tensor tensor(1, 1, 1, N*C*H*W, FLOAT, data);
        tensor_node_list_.insert(std::make_pair(name, &tensorP));
        graph->createNode(name, tensor);
    }
    return 0;
}
int OnnxModel::create_operater_node_(Graph* graph) {
    const onnx::GraphProto& graphP = model_->graph();
    for(int n = 0; n < graphP.node_size(); n++) {
        const onnx::NodeProto node = graphP.node(n);
        std::string name = node.output(0);
        const std::string& op = node.op_type();
        if (op == "Abs") {
            Operator op(OpType_Abs);
            graph->createNode(name, op);
        }
        else if (op == "Acos") {
            Operator op(OpType_Acos);
            graph->createNode(name, op);
        }
        else if (op == "Add") {
            Operator op(OpType_Add);
            graph->createNode(name, op);
        }
        else if (op == "Asin") {
            Operator op(OpType_Asin);
            graph->createNode(name, op);
        }
        else if (op == "Atan") {
            Operator op(OpType_Atan);
            graph->createNode(name, op);
        }
        else if (op == "AveragePool" ) {
            Operator op(OpType_AveragePool);
            op[Pool::WKERNEL].i = 1;
            op[Pool::HKERNEL].i = 1;
            op[Pool::WSTRIDE].i = 1;
            op[Pool::HSTRIDE].i = 1;
            op[Pool::WPAD0].i = 0;
            op[Pool::HPAD0].i = 0;
            op[Pool::WPAD1].i = 0;
            op[Pool::HPAD1].i = 0;
            op[Pool::PADMODE].i = Pool::FLOOR;
            op[Pool::COUNT_PAD].i = 0;
            for (int i=0; i<node.attribute_size(); i++) {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "kernel_shape") {
                    op[Pool::HKERNEL].i = attr.ints(0);
                    op[Pool::WKERNEL].i = attr.ints(1);
                }
                else if(attr.name() == "strides") {
                    op[Pool::HSTRIDE].i = attr.ints(0);
                    op[Pool::WSTRIDE].i = attr.ints(1);
                }
                else if(attr.name() == "pads") {
                    op[Pool::HPAD0].i = attr.ints(0);
                    op[Pool::WPAD0].i = attr.ints(1);
                    op[Pool::HPAD1].i = attr.ints(2);
                    op[Pool::WPAD1].i = attr.ints(3);
                }
                else if (attr.name() == "auto_pad") {
                    if(attr.s() == "SAME_LOWER" || attr.s() == "SAME_LOWER") {
                        op[Pool::PADMODE].i = Pool::SAME;
                    }
                }
                else if (attr.name() == "count_include_pad") {
                    op[Pool::COUNT_PAD].i = attr.i();
                }
            }
            graph->createNode(name, op);
        }
        else if(op == "MaxPool") {
            Operator op(OpType_MaxPool);
            op[Pool::WKERNEL].i = 1;
            op[Pool::HKERNEL].i = 1;
            op[Pool::WSTRIDE].i = 1;
            op[Pool::HSTRIDE].i = 1;
            op[Pool::WPAD0].i = 0;
            op[Pool::HPAD0].i = 0;
            op[Pool::WPAD1].i = 0;
            op[Pool::HPAD1].i = 0;
            op[Pool::PADMODE].i = Pool::FLOOR;
            for (int i=0; i<node.attribute_size(); i++) {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "kernel_shape") {
                    op[Pool::HKERNEL].i = attr.ints(0);
                    op[Pool::WKERNEL].i = attr.ints(1);
                }
                else if(attr.name() == "strides") {
                    op[Pool::HSTRIDE].i = attr.ints(0);
                    op[Pool::WSTRIDE].i = attr.ints(1);
                }
                else if(attr.name() == "pads") {
                    op[Pool::HPAD0].i = attr.ints(0);
                    op[Pool::WPAD0].i = attr.ints(1);
                    op[Pool::HPAD1].i = attr.ints(2);
                    op[Pool::WPAD1].i = attr.ints(3);
                }
                else if (attr.name() == "auto_pad") {
                    if(attr.s() == "SAME_LOWER" || attr.s() == "SAME_LOWER") {
                        op[Pool::PADMODE].i = Pool::SAME;
                    }
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "BatchNormalization") {
            Operator op(OpType_BatchNormalization);
            graph->createNode(name, op);
        }
        else if (op == "Ceil") {
            Operator op(OpType_Ceil);
            graph->createNode(name, op);
        }
        else if (op == "Clip") {
            Operator op(OpType_Clip);
            for (int i=0; i<node.attribute_size(); i++) {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "max") {
                    op[Clip::MAX].i = attr.f();
                }
                else if(attr.name() == "min") {
                    op[Clip::MAX].i = attr.f();
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "Concat") {
            Operator op(OpType_Concat);
            graph->createNode(name, op);
        }
        else if (op == "Constant") {
            const void* data = NULL;
            int N = 1, C = 1, H = 1, W = 1;
            for (int i=0; i<node.attribute_size(); i++) {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "values") {
                    const onnx::TensorProto tensorP = attr.t();
                    std::string name = tensorP.name();
                    switch(tensorP.data_type()) {
                        case 1:
                            data = tensorP.has_raw_data() ? (const float*)tensorP.raw_data().data() : tensorP.float_data().data();
                            break;
                        case 7:
                            data = tensorP.has_raw_data() ? (const int64_t*)tensorP.raw_data().data() : tensorP.int64_data().data();
                            break;
                        default:
                            printf("ERROR!\n");
                    }
                    switch(tensorP.dims_size()) {
                        default:
                        case 0:
                            if (tensorP.has_raw_data()) {
                                const std::string& raw_data = tensorP.raw_data();
                                W = (int)raw_data.size() / 4;
                            }
                            else if (tensorP.data_type() == 1) {
                                W = tensorP.float_data_size();
                            }
                            break;
                        case 1:
                            W = tensorP.dims(0);
                            break;
                        case 2:
                            H = tensorP.dims(0);
                            W = tensorP.dims(1);
                            break;
                        case 3:
                            C = tensorP.dims(0);
                            H = tensorP.dims(1);
                            W = tensorP.dims(2);
                            break;
                        case 4:
                            N = tensorP.dims(0);
                            C = tensorP.dims(1);
                            H = tensorP.dims(2);
                            W = tensorP.dims(3);
                            break;
                    }
                }
            }
            Tensor tensor(1, 1, 1, N*C*H*W, FLOAT, data);
            graph->createNode(name, tensor);
        }
        else if (op == "Conv") {
            std::string weight_name = node.input(1);
            const auto it = tensor_node_list_.find(weight_name);
            if (it == tensor_node_list_.end()) {LOGE("Conv have not weight.\n");}
            Operator op(OpType_Conv);
            op[Conv::OUTCH].i = it->second->dims(0);
            op[Conv::INCH].i  = it->second->dims(1);
            op[Conv::WKERNEL].i = it->second->dims(2);
            op[Conv::HKERNEL].i = it->second->dims(3);
            op[Conv::WDILATION].i = 1;
            op[Conv::HDILATION].i = 1;
            op[Conv::WSTRIDE].i = 1;
            op[Conv::HSTRIDE].i = 1;
            op[Conv::WPAD0].i = 0;
            op[Conv::HPAD0].i = 0;
            op[Conv::WPAD1].i = 0;
            op[Conv::HPAD1].i = 0;
            op[Conv::GROUP].i = 1;
            for (int i=0; i<node.attribute_size(); i++) {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "dilations") {
                    op[Conv::HDILATION].i = attr.ints(0);
                    op[Conv::WDILATION].i = attr.ints(1);
                }
                else if(attr.name() == "group") {
                    op[Conv::GROUP].i = attr.i();
                }
                else if(attr.name() == "strides") {
                    op[Conv::HSTRIDE].i = attr.ints(0);
                    op[Conv::WSTRIDE].i = attr.ints(1);
                }
                else if(attr.name() == "pads") {
                    op[Conv::WPAD0].i = attr.ints(0);
                    op[Conv::HPAD0].i = attr.ints(1);
                    op[Conv::WPAD1].i = attr.ints(2);
                    op[Conv::HPAD1].i = attr.ints(3);
                }
                else if (attr.name() == "auto_pad") {
                    if(attr.s() == "SAME_LOWER" || attr.s() == "SAME_LOWER") {
                        op[Conv::PADMODE].i = Conv::SAME;
                    }
                }
            }
            op[Conv::INCH].i *= op[Conv::GROUP].i;
            graph->createNode(name, op);
        }
        else if (op == "ConvTranspose") {
            std::string weight_name = node.input(1);
            const auto it = tensor_node_list_.find(weight_name);
            if (it == tensor_node_list_.end()) {LOGE("Conv have not weight.\n");}
            Operator op(OpType_ConvTranspose);
            op[Conv::OUTCH].i = it->second->dims(0);
            op[Conv::INCH].i  = it->second->dims(1);
            op[Conv::WKERNEL].i = it->second->dims(2);
            op[Conv::HKERNEL].i = it->second->dims(3);
            op[Conv::WDILATION].i = 1;
            op[Conv::HDILATION].i = 1;
            op[Conv::WSTRIDE].i = 1;
            op[Conv::HSTRIDE].i = 1;
            op[Conv::WPAD0].i = 0;
            op[Conv::HPAD0].i = 0;
            op[Conv::WPAD1].i = 0;
            op[Conv::HPAD1].i = 0;
            op[Conv::GROUP].i = 1;
            for (int i=0; i<node.attribute_size(); i++) {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "dilations")
                {
                    op[Conv::HDILATION].i = attr.ints(0);
                    op[Conv::WDILATION].i = attr.ints(1);
                }
                else if(attr.name() == "group") {
                    op[Conv::GROUP].i = attr.i();
                }
                else if(attr.name() == "strides") {
                    op[Conv::HSTRIDE].i = attr.ints(0);
                    op[Conv::WSTRIDE].i = attr.ints(1);
                }
                else if(attr.name() == "pads") {
                    op[Conv::WPAD0].i = attr.ints(0);
                    op[Conv::HPAD0].i = attr.ints(1);
                    op[Conv::WPAD1].i = attr.ints(2);
                    op[Conv::HPAD1].i = attr.ints(3);
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "Cos") {
            Operator op(OpType_Cos);
            graph->createNode(name, op);
        }
        else if (op == "Div") {
            Operator op(OpType_Div);
            graph->createNode(name, op);
        }
        else if (op == "Dropout") {
            Operator op(OpType_Dropout);
            op[Dropout::SCALE].f = 1.0;
            for (int i=0; i<node.attribute_size(); i++)
            {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "ratio") {
                    op[Dropout::SCALE].f = attr.f();
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "Elu") {
            Operator op(OpType_Elu);
            op[Elu::ALPHA].f = 1.0;
            for (int i=0; i<node.attribute_size(); i++)
            {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "alpha") {
                    op[Elu::ALPHA].f = attr.f();
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "Exp") {
            Operator op(OpType_Exp);
            graph->createNode(name, op);
        }
        else if (op == "Flatten") {
            Operator op(OpType_Flatten);
            op[Flatten::AXIS].i = 1;
            for (int i=0; i<node.attribute_size(); i++)
            {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "axis") {
                    op[Flatten::AXIS].i = attr.i();
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "Floor") {
            Operator op(OpType_Floor);
            graph->createNode(name, op);
        }
        else if (op == "Gemm") {
            Operator op(OpType_Gemm);
            op[Gemm::ALPHA].f = 1;
            op[Gemm::BETA].f = 1;
            op[Gemm::TRANSA].i = 1;
            op[Gemm::TRANSB].i = 1;
            for (int i=0; i<node.attribute_size(); i++) {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "alpha") {
                    op[Gemm::ALPHA].f = attr.f();
                }
                else if(attr.name() == "beta") {
                    op[Gemm::BETA].f = attr.f();
                }
                else if(attr.name() == "transA") {
                    op[Gemm::TRANSA].i = attr.i();
                }
                else if(attr.name() == "transB") {
                    op[Gemm::TRANSB].i = attr.i();
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "GlobalAveragePool") {
            Operator op(OpType_GlobalAveragePool);
            graph->createNode(name, op);
        }
        else if (op == "GlobalMaxPool") {
            Operator op(OpType_GlobalMaxPool);
            graph->createNode(name, op);
        }
        else if (op == "InstanceNormalization") {
            LOGE("%-16s\n", "InstanceNorm");
        }
        else if (op == "LeakyRelu") {
            Operator op(OpType_LeakyRelu);
            op[Elu::ALPHA].f = 1.0;
            for (int i=0; i<node.attribute_size(); i++)
            {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "alpha") {
                    op[Elu::ALPHA].f = attr.f();
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "Log") {
            Operator op(OpType_Log);
            graph->createNode(name, op);
        }
        else if (op == "LRN") {
            Operator op(OpType_LRN);
            op[LRN::ALPHA].f = 0;
            op[LRN::BETA].f = 0;
            op[LRN::BIAS].f = 0;
            op[LRN::LOCAL_SIZE].i = 1;
            for (int i=0; i<node.attribute_size(); i++)
            {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "alpha") {
                    op[LRN::ALPHA].f = attr.f();
                }
                else if(attr.name() == "beta") {
                    op[LRN::BETA].f = attr.f();
                }
                else if(attr.name() == "bias") {
                    op[LRN::BIAS].f = attr.f();
                }
                else if(attr.name() == "size") {
                    op[LRN::LOCAL_SIZE].i = attr.i();
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "MatMul") {
            LOGE("%-16s\n", "InnerProduct");
        }
        else if (op == "Max") {
            Operator op(OpType_Max);
            graph->createNode(name, op);
        }
        else if (op == "Min") {
            Operator op(OpType_Min);
            graph->createNode(name, op);
        }
        else if (op == "Mul") {
            Operator op(OpType_Mul);
            graph->createNode(name, op);
        }
        else if (op == "Neg") {
            Operator op(OpType_Neg);
            graph->createNode(name, op);
        }
        else if (op == "Pad") {
            Operator op(OpType_Pad);
            for (int i=0; i<node.attribute_size(); i++) {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "mode") {
                    op[Pad::MODE].i  = attr.i();
                }
                else if (attr.name() == "value") {
                    op[Pad::VALUE].f  = attr.f();
                }
                else if (attr.name() == "pads") {
                    if (attr.ints_size() == 8) {
                        op[Pad::HPAD0].i = attr.ints(2);
                        op[Pad::HPAD1].i = attr.ints(6);
                        op[Pad::WPAD1].i = attr.ints(3);
                        op[Pad::WPAD1].i = attr.ints(7);
                    }
                    else if (attr.ints_size() == 6) {
                        op[Pad::HPAD0].i = attr.ints(1);
                        op[Pad::HPAD1].i = attr.ints(4);
                        op[Pad::WPAD1].i = attr.ints(2);
                        op[Pad::WPAD1].i = attr.ints(5);
                    }
                    else {
                        op[Pad::HPAD0].i = attr.ints(0);
                        op[Pad::HPAD1].i = attr.ints(2);
                        op[Pad::WPAD1].i = attr.ints(1);
                        op[Pad::WPAD1].i = attr.ints(3);
                    }
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "Pow") {
            Operator op(OpType_Pow);
            graph->createNode(name, op);
        }
        else if (op == "PRelu") {
            Operator op(OpType_PRelu);
            graph->createNode(name, op);
        }
        else if (op == "Reciprocal") {
            LOGE("%-16s\n", "UnaryOp");
        }
        else if (op == "Relu") {
            Operator op(OpType_Relu);
            graph->createNode(name, op);
        }
        else if (op == "Reshape") {
            Operator op(OpType_Reshape);
            op[Reshape::CHANNEL].i  = 0;
            op[Reshape::HEIGHT].i   = 0;
            op[Reshape::WIDTH].i    = 0;
            for (int i=0; i<node.attribute_size(); i++) {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "shape") {
                    op[Reshape::CHANNEL].i  = 1;
                    op[Reshape::HEIGHT].i   = attr.ints(0);
                    op[Reshape::WIDTH].i    = attr.ints(1);
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "Sigmoid") {
            Operator op(OpType_Sigmoid);
            graph->createNode(name, op);
        }
        else if (op == "Sin") {
            Operator op(OpType_Sin);
            graph->createNode(name, op);
        }
        else if (op == "Softmax") {
            Operator op(OpType_Softmax);
            graph->createNode(name, op);
        }
        else if (op == "Sqrt") {
            LOGE("%-16s\n", "UnaryOp");
        }
        else if (op == "Sub") {
            Operator op(OpType_Sub);
            graph->createNode(name, op);
        }
        else if (op == "Sum") {
            Operator op(OpType_Add);
            graph->createNode(name, op);
        }
        else if (op == "Tan") {
            Operator op(OpType_Tan);
            graph->createNode(name, op);
        }
        else if (op == "Transpose") {
            Operator op(OpType_Transpose);
            op[Transpose::NTO].i = 0;
            op[Transpose::CTO].i = 0;
            op[Transpose::HTO].i = 0;
            op[Transpose::WTO].i = 0;
            for (int i=0; i<node.attribute_size(); i++) {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "perm") {
                    if(attr.ints_size() == 5) {
                        op[Transpose::NTO].i  = attr.ints(1);
                        op[Transpose::CTO].i  = attr.ints(2);
                        op[Transpose::HTO].i  = attr.ints(3);
                        op[Transpose::WTO].i  = attr.ints(4);
                    }
                    else if(attr.ints_size() == 4) {
                        op[Transpose::CTO].i  = attr.ints(1);
                        op[Transpose::HTO].i  = attr.ints(2);
                        op[Transpose::WTO].i  = attr.ints(3);
                    }
                    else if(attr.ints_size() == 3) {
                        op[Transpose::CTO].i  = attr.ints(0);
                        op[Transpose::HTO].i  = attr.ints(1);
                        op[Transpose::WTO].i  = attr.ints(2);
                    }
                    else if(attr.ints_size() == 2) {
                        op[Transpose::HTO].i  = attr.ints(0);
                        op[Transpose::WTO].i  = attr.ints(1);
                    }
                    else {
                        LOGE("transpose error %d\n",attr.ints_size());
                    }
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "Upsample") {
            Operator op(OpType_Upsample);
            op[Upsample::HEIGHT].i = 0;
            op[Upsample::WIDTH].i = 0;
            op[Upsample::HEIGHT_SCALE].f = 1.f;
            op[Upsample::WIDTH_SCALE].f = 1.f;
            op[Upsample::UPSAMPLE_MODE].i = 1;
            for (int i=0; i<node.attribute_size(); i++) {
                const onnx::AttributeProto& attr = node.attribute(i);
                if (attr.name() == "mode") {
                    if(attr.s() == "nearest") {
                        op[Upsample::UPSAMPLE_MODE].i = Upsample::NEAREST;
                    }
                    else if(attr.s() == "bilinear" || attr.s() == "linear") {
                        op[Upsample::UPSAMPLE_MODE].i = Upsample::BILINEAR;
                    }
                    else if(attr.s() == "trilinear") {
                        op[Upsample::UPSAMPLE_MODE].i = Upsample::BICUBIC;
                    }
                    else {
                        LOGE("resize error %s\n",attr.s().c_str());
                    }
                }
                else if(attr.name() == "scales") {
                    if (attr.floats_size() == 2) {
                        op[Upsample::WIDTH_SCALE].f = attr.floats(1);
                        printf("%f %f\n", attr.floats(0), attr.floats(1));
                    }
                    else if (attr.floats_size() == 3) {
                        op[Upsample::HEIGHT_SCALE].f = attr.floats(1);
                        op[Upsample::WIDTH_SCALE].f = attr.floats(2);
                        printf("%f %f %f\n", attr.floats(0), attr.floats(1), attr.floats(2));
                    }
                    else {
                        LOGE("resize error scale size %d\n",attr.floats_size());
                    }
                }
                else if(attr.name() == "align_corners") {
                    op[Upsample::PAD_MODE].i = 1;
                }
            }
            graph->createNode(name, op);
        }
        else if (op == "Resize") {
            Operator op(OpType_Resize);
            for (int i=0; i<node.attribute_size(); i++) {
                const onnx::AttributeProto& attr = node.attribute(i);
                if(attr.name() == "coordinate_transformation_mode") {
                    if(attr.s() == "half_pixel") op[Resize::TRANSFORMATION_MODE].i = Resize::HALF_PIXEL;
                    if(attr.s() == "pytorch_half_pixel") op[Resize::TRANSFORMATION_MODE].i = Resize::PYTORCH_HALF_PIXEL;
                    if(attr.s() == "align_corners") op[Resize::TRANSFORMATION_MODE].i = Resize::ALIGN_CORNERS;
                    if(attr.s() == "asymmetric") op[Resize::TRANSFORMATION_MODE].i = Resize::ASYMMETRIC;
                    if(attr.s() == "tf_crop_and_resize") op[Resize::TRANSFORMATION_MODE].i = Resize::TF_CROP_AND_RESIZE;
                }
                else if (attr.name() == "cubic_coeff_a") {
                    op[Resize::CUBIC_COEFF_A].f = attr.f();
                }
                else if (attr.name() == "exclude_outside") {
                    op[Resize::EXCLUDE_OUTSIDE].i = attr.i();
                }
                else if (attr.name() == "cubic_coeff_a") {
                    op[Resize::EXTRAPOLATION_VALUE].f = attr.f();
                }
                else if (attr.name() == "mode") {
                    if(attr.s() == "nearest") op[Resize::MODE].i = Resize::NEAREST;
                    if(attr.s() == "linear") op[Resize::MODE].i = Resize::LINEAR;
                    if(attr.s() == "cubix") op[Resize::MODE].i = Resize::CUBIC;
                }
                else if(attr.name() == "nearest_mode") {
                    op[Resize::NEAREST_MODE].i = 1;
                }
            }
            graph->createNode(name, op);
        }
        else {
            LOGE("not support %s!\n", op.c_str());
        }
    }
    return 0;
}
}
