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

#include "caffe_model.h"
#include <fstream>
#include <climits>
#include <vector>
#include <string>
#include <map>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

#include "graph.h"
#include "reference.h"

using caffe::NetParameter;
CaffeModel::CaffeModel() {
    m_prototxt   = new NetParameter();
    m_model = new NetParameter();
}
CaffeModel::~CaffeModel() {
    delete m_model;
    delete m_prototxt;
}
int CaffeModel::load(const char* prototxt, const char* model) {
    bool success = true;
    {
        std::ifstream fs(prototxt, std::ifstream::in);
        if (!fs.is_open()) {
            fprintf(stderr, "open failed %s\n", prototxt);
            return false;
        }
        google::protobuf::io::IstreamInputStream input(&fs);
        google::protobuf::Message* message = (google::protobuf::Message*)m_prototxt;
        success = google::protobuf::TextFormat::Parse(&input, message);
        fs.close();
    }

    {
        std::ifstream fs(model, std::ifstream::in | std::ifstream::binary);
        if (!fs.is_open()) {
            printf("open failed %s\n", model);
            return false;
        }   
        google::protobuf::io::IstreamInputStream input(&fs);
        google::protobuf::io::CodedInputStream codedstr(&input);
        google::protobuf::Message* message = (google::protobuf::Message*)m_model;
        success = message->ParseFromCodedStream(&codedstr);
        fs.close();
    }
    return success;
}
const caffe::LayerParameter& CaffeModel::getParam(std::string name) {
    const caffe::LayerParameter* layer = nullptr;
    for (int v = 0; v < m_model->layer_size(); ++v) {
        auto& l = m_model->layer(v);
        if (l.name() == name) {
            layer = &l;
            break;
        }
    }
    return *layer;
}
int CaffeModel::draw(Graph* graph) {
    std::map<std::string, std::string> inplace;
    std::map<std::string, std::string> top2name;
    for (int l = 0; l < m_prototxt->layer_size(); ++l) {
        const caffe::LayerParameter& layer  = m_prototxt->layer(l);
        const std::string name = layer.name();

        if(layer.type() == "AbsVal") {
            Operator op(OpType_Abs);
            graph->createNode(name, op);
        }
        else if(layer.type() == "ArgMax") {
            Operator op(OpType_ArgMax);
            const caffe::ArgMaxParameter& argmaxP = layer.argmax_param();
            if (argmaxP.has_out_max_val()) op[ArgMax::OUT_MAX_VAL].i = argmaxP.out_max_val();
            if (argmaxP.has_top_k()) op[ArgMax::TOP_K].i = argmaxP.top_k();
            if (argmaxP.has_axis()) op[ArgMax::AXIS].i = argmaxP.axis();
            graph->createNode(name, op);
        }
        else if(layer.type() == "Bias") {
            fprintf(stderr, "Not support %s\n", layer.type().c_str());
        }
        else if(layer.type() == "BNLL") { 
            fprintf(stderr, "Not support %s\n", layer.type().c_str());
        }
        else if(layer.type() == "BatchNorm") {
            Operator op(OpType_BatchNormalization);
            const caffe::LayerParameter& param       = getParam(name);
            const caffe::BlobProto& meanBlob         = param.blobs(0);
            const caffe::BlobProto& varBlob          = param.blobs(1);
            int num = 0;
            if (meanBlob.has_shape()) {
                num = meanBlob.data_size();
            }
            else {
                num = meanBlob.num() * meanBlob.channels() 
                    * meanBlob.height() * meanBlob.width();
            }
            Tensor mean(1, 1, num, FLOAT, meanBlob.data().data());
            Tensor var(1, 1, num, FLOAT, varBlob.data().data());
            Tensor scale(1, 1, num, FLOAT);
            Tensor bias(1, 1, num, FLOAT);
            scale.fill(1.f);
            bias.fill(0.f);
            graph->createNode(name, op);
            graph->createNode(name+"_m_", mean);
            graph->createNode(name+"_v_", var);
            graph->createNode(name+"_s_", scale);
            graph->createNode(name+"_b_", bias);
            graph->link(name+"_s_", name);
            graph->link(name+"_b_", name);
            graph->link(name+"_m_", name);
            graph->link(name+"_v_", name);
        }
        else if(layer.type() == "Crop") {
            Operator op(OpType_Crop);
            const caffe::CropParameter& cropProto = layer.crop_param();
            int num_offset = cropProto.offset_size();
            if (num_offset == 1) {
                int offset = cropProto.offset(0);
                int axis = cropProto.axis();
                if (axis == 1) {
                    fprintf(stderr, "Crop not support corp c\n");
                }
                else if (axis == 2) {
                    op[Crop::WCROP0].i = offset;
                    op[Crop::HCROP0].i = offset;
                }
                else if (axis == 3) {
                    op[Crop::HCROP0].i = offset;
                }
            }
            else if (num_offset == 2) {
                op[Crop::WCROP0].i = cropProto.offset(1);
                op[Crop::HCROP0].i = cropProto.offset(0);
            }
            else if (num_offset == 3) {
                fprintf(stderr, "Crop not support corp c\n");
            }
            graph->createNode(name, op);
        }
        else if(layer.type() == "Concat") {
            Operator op(OpType_Concat);
            graph->createNode(name, op);
        }
        else if(layer.type() == "Convolution") {
            Operator op(OpType_Conv);
            op[Conv::WDILATION].i = 1;
            op[Conv::HDILATION].i = 1;
            op[Conv::WSTRIDE].i = 1;
            op[Conv::HSTRIDE].i = 1;
            op[Conv::WPAD0].i = 0;
            op[Conv::HPAD0].i = 0;
            op[Conv::WPAD1].i = 0;
            op[Conv::HPAD1].i = 0;
            op[Conv::GROUP].i = 1;
            const caffe::ConvolutionParameter& convProto = layer.convolution_param();
            const caffe::LayerParameter& param = getParam(name);
            const caffe::BlobProto& weightBlob = param.blobs(0);
            int weight_n, weight_c, weight_h, weight_w;
            if (weightBlob.has_shape()) {
                weight_n = weightBlob.shape().dim(0);
                weight_c = weightBlob.shape().dim(1);
                weight_h = weightBlob.shape().dim(2);
                weight_w = weightBlob.shape().dim(3);
            }
            else {
                weight_n = weightBlob.num();
                weight_c = weightBlob.channels();
                weight_h = weightBlob.height();
                weight_w = weightBlob.width();
            }
            if(convProto.kernel_size_size() == 1)   op[Conv::WKERNEL].i = convProto.kernel_size(0);
            if(convProto.kernel_size_size() == 1)   op[Conv::HKERNEL].i = convProto.kernel_size(0);
            if(convProto.has_kernel_w())       op[Conv::WKERNEL].i = convProto.kernel_w();
            if(convProto.has_kernel_h())       op[Conv::HKERNEL].i = convProto.kernel_h();
            if(convProto.dilation_size() == 1) op[Conv::WSTRIDE].i = convProto.dilation(0);
            if(convProto.dilation_size() == 1) op[Conv::HSTRIDE].i = convProto.dilation(0);
            if(convProto.dilation_size() == 2) op[Conv::WSTRIDE].i = convProto.dilation(0);
            if(convProto.dilation_size() == 2) op[Conv::HSTRIDE].i = convProto.dilation(1);
            if(convProto.stride_size() == 1)   op[Conv::WSTRIDE].i = convProto.stride(0);
            if(convProto.stride_size() == 1)   op[Conv::HSTRIDE].i = convProto.stride(0);
            if(convProto.stride_size() == 2)   op[Conv::WSTRIDE].i = convProto.stride(0);
            if(convProto.stride_size() == 2)   op[Conv::HSTRIDE].i = convProto.stride(1);
            if(convProto.has_stride_w())       op[Conv::WSTRIDE].i = convProto.stride_w();
            if(convProto.has_stride_h())       op[Conv::HSTRIDE].i = convProto.stride_h();
            if(convProto.pad_size() == 1)      op[Conv::WPAD0].i = convProto.pad(0);
            if(convProto.pad_size() == 1)      op[Conv::HPAD0].i = convProto.pad(0);
            if(convProto.pad_size() == 1)      op[Conv::WPAD1].i = convProto.pad(0);
            if(convProto.pad_size() == 1)      op[Conv::HPAD1].i = convProto.pad(0);
            if(convProto.has_pad_w())          op[Conv::WPAD0].i = convProto.pad_w();
            if(convProto.has_pad_h())          op[Conv::HPAD0].i = convProto.pad_h();
            if(convProto.has_pad_w())          op[Conv::WPAD1].i = convProto.pad_w();
            if(convProto.has_pad_h())          op[Conv::HPAD1].i = convProto.pad_h();
            if(convProto.has_group())          op[Conv::GROUP].i = convProto.group();
            op[Conv::OUTCH].i = convProto.num_output();
            op[Conv::INCH].i  = weight_n * weight_c / op[Conv::OUTCH].i * op[Conv::GROUP].i;
            Tensor tensor(1, 1, 1, weight_n*weight_c*weight_h*weight_w, FLOAT, weightBlob.data().data());
            graph->createNode(name, op);
            graph->createNode(name+"_weight", tensor);
            graph->link(name+"_weight", name);
            if (convProto.bias_term() && param.blobs_size() >= 2) {
                const caffe::BlobProto& biasBlob = param.blobs(1);
                Tensor tensor(1, 1, weight_n, FLOAT, biasBlob.data().data());
                graph->createNode(name+"_bias", tensor);
                graph->link(name+"_bias", name);
            }
        }
        else if(layer.type() == "Deconvolution") {
            Operator op(OpType_ConvTranspose);
            op[Conv::WDILATION].i = 1;
            op[Conv::HDILATION].i = 1;
            op[Conv::WSTRIDE].i = 1;
            op[Conv::HSTRIDE].i = 1;
            op[Conv::WPAD0].i = 0;
            op[Conv::HPAD0].i = 0;
            op[Conv::WPAD1].i = 0;
            op[Conv::HPAD1].i = 0;
            op[Conv::GROUP].i = 1;
            const caffe::ConvolutionParameter& convProto = layer.convolution_param();
            const caffe::LayerParameter& param = getParam(name);
            const caffe::BlobProto& weightBlob = param.blobs(0);
            int weight_n, weight_c, weight_h, weight_w;
            if (weightBlob.has_shape()) {
                weight_n = weightBlob.shape().dim(0);
                weight_c = weightBlob.shape().dim(1);
                weight_h = weightBlob.shape().dim(2);
                weight_w = weightBlob.shape().dim(3);
            }
            else {
                weight_n = weightBlob.num();
                weight_c = weightBlob.channels();
                weight_h = weightBlob.height();
                weight_w = weightBlob.width();
            }
            if(convProto.kernel_size_size() == 1)   op[Conv::WKERNEL].i = convProto.kernel_size(0);
            if(convProto.kernel_size_size() == 1)   op[Conv::HKERNEL].i = convProto.kernel_size(0);
            if(convProto.has_kernel_w())       op[Conv::WKERNEL].i = convProto.kernel_w();
            if(convProto.has_kernel_h())       op[Conv::HKERNEL].i = convProto.kernel_h();
            if(convProto.dilation_size() == 1) op[Conv::WSTRIDE].i = convProto.dilation(0);
            if(convProto.dilation_size() == 1) op[Conv::HSTRIDE].i = convProto.dilation(0);
            if(convProto.dilation_size() == 2) op[Conv::WSTRIDE].i = convProto.dilation(0);
            if(convProto.dilation_size() == 2) op[Conv::HSTRIDE].i = convProto.dilation(1);
            if(convProto.stride_size() == 1)   op[Conv::WSTRIDE].i = convProto.stride(0);
            if(convProto.stride_size() == 1)   op[Conv::HSTRIDE].i = convProto.stride(0);
            if(convProto.stride_size() == 2)   op[Conv::WSTRIDE].i = convProto.stride(0);
            if(convProto.stride_size() == 2)   op[Conv::HSTRIDE].i = convProto.stride(1);
            if(convProto.has_stride_w())       op[Conv::WSTRIDE].i = convProto.stride_w();
            if(convProto.has_stride_h())       op[Conv::HSTRIDE].i = convProto.stride_h();
            if(convProto.pad_size() == 1)      op[Conv::WPAD0].i = convProto.pad(0);
            if(convProto.pad_size() == 1)      op[Conv::HPAD0].i = convProto.pad(0);
            if(convProto.pad_size() == 1)      op[Conv::WPAD1].i = convProto.pad(0);
            if(convProto.pad_size() == 1)      op[Conv::HPAD1].i = convProto.pad(0);
            if(convProto.has_pad_w())          op[Conv::WPAD1].i = convProto.pad_w();
            if(convProto.has_pad_h())          op[Conv::HPAD1].i = convProto.pad_h();
            if(convProto.has_group())          op[Conv::GROUP].i = convProto.group();
            op[Conv::OUTCH].i = convProto.num_output();
            op[Conv::INCH].i  = weight_n * weight_c / op[Conv::OUTCH].i * op[Conv::GROUP].i;
            Tensor tensor(weight_n, weight_c, weight_h, weight_w, FLOAT, weightBlob.data().data());
            graph->createNode(name, op);
            graph->createNode(name+"_weight", tensor);
            graph->link(name+"_weight", name);
            if (convProto.bias_term() && param.blobs_size() >= 2) {
                const caffe::BlobProto& biasBlob = param.blobs(1);
                Tensor tensor(1, 1, weight_n, FLOAT, biasBlob.data().data());
                graph->createNode(name+"_bias", tensor);
                graph->link(name+"_bias", name);
            }
        }
        else if(layer.type() == "DetectionOutput") {
            fprintf(stderr, "Not support %s\n", layer.type().c_str());
        }
        else if(layer.type() == "Exp") {
            Operator op(OpType_Exp);
            graph->createNode(name, op);
        }
        else if(layer.type() == "Eltwise") {
            Operator op(OpType_Add);
            graph->createNode(name, op);
        }
        else if(layer.type() == "ELU") {
            Operator op(OpType_Elu);
            const caffe::ELUParameter& eluProto = layer.elu_param();
            if(eluProto.has_alpha()) op[Elu::ALPHA].f = eluProto.alpha();
            graph->createNode(name, op);
        }
        else if(layer.type() == "Flatten") {
            Operator op(OpType_Flatten);
            op[Flatten::AXIS].i = 1;
            graph->createNode(name, op);
        }
        else if(layer.type() == "InnerProduct") {
            Operator op(OpType_Gemm);
            const caffe::InnerProductParameter& convProto = layer.inner_product_param();
            const caffe::LayerParameter& param = getParam(name);
            const caffe::BlobProto& weightBlob = param.blobs(0);
            int weight_n, weight_c;
            if (weightBlob.has_shape()) {
                weight_n = weightBlob.shape().dim(0);
                weight_c = weightBlob.shape().dim(1);
            }
            else {
                weight_n = weightBlob.height();
                weight_c = weightBlob.width();
            }
            Tensor tensor(1, weight_n, weight_c, FLOAT, weightBlob.data().data());
            graph->createNode(name, op);
            graph->createNode(name+"_weight", tensor);
            graph->link(name+"_weight", name);
            if (convProto.bias_term() && param.blobs_size() >= 2) {
                const caffe::BlobProto& biasBlob = param.blobs(1);
                Tensor tensor(1, 1, weight_n, FLOAT, biasBlob.data().data());
                graph->createNode(name+"_bias", tensor);
                graph->link(name+"_bias", name);
            }
        }
        else if(layer.type() == "Log") {
            Operator op(OpType_Log);
            graph->createNode(name, op);
        }
        else if(layer.type() == "LRN") {
            const caffe::LRNParameter& lrnProto = layer.lrn_param();
            Operator op(OpType_LRN);
            op[LRN::ALPHA].f = 1;
            op[LRN::BETA].f = 0.75;
            op[LRN::BIAS].f = 1.f;
            op[LRN::LOCAL_SIZE].i = 5;
            op[LRN::NORM_REGION].i = LRN::ACROSS_CHANNELS;
            if(lrnProto.has_alpha())op[LRN::ALPHA].f = lrnProto.alpha();
            if(lrnProto.has_beta()) op[LRN::BETA].f = lrnProto.beta();
            if(lrnProto.has_local_size())op[LRN::LOCAL_SIZE].i = lrnProto.local_size();
            if(lrnProto.has_norm_region())op[LRN::NORM_REGION].i = lrnProto.norm_region();
            graph->createNode(name, op);
        }
        else if(layer.type() == "MVN") {
            const caffe::MVNParameter& mvnProto = layer.mvn_param();
            Operator op(OpType_LRN);
            op[MVN::NORMALIZE_VARIANCE].f = 0;
            op[MVN::ACROSS_CHANNELS].f = 0;
            if(mvnProto.has_across_channels())op[MVN::NORMALIZE_VARIANCE].f = mvnProto.across_channels();
            if(mvnProto.has_normalize_variance()) op[MVN::ACROSS_CHANNELS].f = mvnProto.normalize_variance();
            graph->createNode(name, op);
        }
        else if(layer.type() == "Normalize") {
            fprintf(stderr, "Not support %s\n", layer.type().c_str());
        }
        else if(layer.type() == "Power") {
            const caffe::PowerParameter& powerProto = layer.power_param();
            Operator op(OpType_Power);
            op[Power::POWER].f = 1.f;
            op[Power::SCALE].f = 1.f;
            op[Power::SHIFT].f = 0.f;
            if(powerProto.has_power()) op[Power::POWER].f = powerProto.power();
            if(powerProto.has_scale()) op[Power::SCALE].f = powerProto.scale();
            if(powerProto.has_shift()) op[Power::SHIFT].f = powerProto.shift();
            graph->createNode(name, op);
        }
        else if(layer.type() == "Permute") {
            const caffe::PermuteParameter& pmProto = layer.permute_param();
            Operator op(OpType_Transpose);
            op[Transpose::CTO].i = 0;
            op[Transpose::HTO].i = 1;
            op[Transpose::WTO].i = 2;
            if(pmProto.order_size() == 4) {
                op[Transpose::CTO].i = pmProto.order(1);
                op[Transpose::HTO].i = pmProto.order(2);
                op[Transpose::WTO].i = pmProto.order(3);
            }
            else if(pmProto.order_size() == 3) {
                op[Transpose::CTO].i = pmProto.order(0);
                op[Transpose::HTO].i = pmProto.order(1);
                op[Transpose::WTO].i = pmProto.order(2);
            }
            else if(pmProto.order_size() == 2) {
                op[Transpose::HTO].i = pmProto.order(0);
                op[Transpose::WTO].i = pmProto.order(1);
            }
            graph->createNode(name, op);
        }
        else if(layer.type() == "Pooling") {
            const caffe::PoolingParameter& poolP = layer.pooling_param();
            Operator op;
            op[Pool::PADMODE].i = Pool::CEIL;
            if(poolP.has_global_pooling() && poolP.global_pooling()) {
                if(poolP.pool() == caffe::PoolingParameter::AVE) {
                    op = Operator(OpType_GlobalAveragePool);
                }
                else if(poolP.pool() == caffe::PoolingParameter::MAX) {
                    fprintf(stderr, "POOL ERROR\n");
                }
            }
            else {
                if(poolP.pool() == caffe::PoolingParameter::AVE) {
                    op = Operator(OpType_AveragePool);
                }
                else if(poolP.pool() == caffe::PoolingParameter::MAX) {
                    op = Operator(OpType_MaxPool);
                }
            }
            if (poolP.has_kernel_size()) op[Pool::WKERNEL].i = poolP.kernel_size();
            if (poolP.has_kernel_size()) op[Pool::HKERNEL].i = poolP.kernel_size();
            if (poolP.has_kernel_w()) op[Pool::WKERNEL].i = poolP.kernel_w();
            if (poolP.has_kernel_h()) op[Pool::HKERNEL].i = poolP.kernel_h();
            if (poolP.has_stride()) op[Pool::WSTRIDE].i = poolP.stride();
            if (poolP.has_stride()) op[Pool::HSTRIDE].i = poolP.stride();
            if (poolP.has_stride_w()) op[Pool::WSTRIDE].i = poolP.stride_w();
            if (poolP.has_stride_h()) op[Pool::HSTRIDE].i = poolP.stride_h();
            if (poolP.has_pad()) op[Pool::WPAD0].i = poolP.pad();
            if (poolP.has_pad()) op[Pool::HPAD0].i = poolP.pad();
            if (poolP.has_pad()) op[Pool::WPAD1].i = poolP.pad();
            if (poolP.has_pad()) op[Pool::HPAD1].i = poolP.pad();
            if (poolP.has_pad_w()) op[Pool::WPAD0].i = poolP.pad_w();
            if (poolP.has_pad_w()) op[Pool::WPAD1].i = poolP.pad_w();
            if (poolP.has_pad_h()) op[Pool::HPAD0].i = poolP.pad_h();
            if (poolP.has_pad_h()) op[Pool::HPAD1].i = poolP.pad_h();
            graph->createNode(name, op);
        }
        else if(layer.type() == "Embed") {
            fprintf(stderr, "Not support %s\n", layer.type().c_str());
        }
        else if(layer.type() == "Reduction") { 
            fprintf(stderr, "Not support %s\n", layer.type().c_str());
        }
        else if(layer.type() == "ReLU") {
            Operator op(OpType_Relu);
            graph->createNode(name, op);
        }
        else if(layer.type() == "ReLU6") {
            Operator op(OpType_Relu6);
            graph->createNode(name, op);
        }
        else if(layer.type() == "PriorBox") {
            Operator op(OpType_Priobox);
            const caffe::PriorBoxParameter& param = layer.prior_box_param();

            if (param.variance_size() == 4) {
                op[Priorbox::VARIANCES0].f = param.variance(0);
                op[Priorbox::VARIANCES1].f = param.variance(1);
                op[Priorbox::VARIANCES2].f = param.variance(2);
                op[Priorbox::VARIANCES3].f = param.variance(3);
            }
            else if (param.variance_size() == 1) {
                op[Priorbox::VARIANCES0].f = param.variance(0);
                op[Priorbox::VARIANCES1].f = param.variance(0);
                op[Priorbox::VARIANCES2].f = param.variance(0);
                op[Priorbox::VARIANCES3].f = param.variance(0);
            }
            if(param.has_flip()) op[Priorbox::FLIP].i = param.flip();
            if(param.has_clip()) op[Priorbox::CLIP].i = param.clip();
            if (param.has_img_size()) {
                op[Priorbox::IMAGE_WIDTH].i  = param.img_size();
                op[Priorbox::IMAGE_HEIGHT].i = param.img_size();
            }
            else if (param.has_img_w() && param.has_img_h()) {
                op[Priorbox::IMAGE_WIDTH].i  = param.img_w();
                op[Priorbox::IMAGE_HEIGHT].i = param.img_h();
            }

            if (param.has_step()) {
                op[Priorbox::STEP_WIDTH].i  = param.step();
                op[Priorbox::STEP_HEIGHT].i = param.step();
            }
            else if (param.has_step_w() && param.has_step_h()) {
                op[Priorbox::STEP_WIDTH].i  = param.step_w();
                op[Priorbox::STEP_HEIGHT].i = param.step_h();
            }

            Tensor min(1, 1, param.min_size_size(), FLOAT);
            memcpy(min.data(), param.min_size().data(), param.min_size_size());
            Tensor max(1, 1, param.max_size_size(), FLOAT);
            memcpy(max.data(), param.max_size().data(), param.max_size_size());
            Tensor ratio(1, 1, param.aspect_ratio_size(), FLOAT);
            memcpy(ratio.data(), param.aspect_ratio().data(), param.aspect_ratio_size());
            graph->createNode(name+"_min", min);
            graph->createNode(name+"_max", max);
            graph->createNode(name+"_ratio", ratio);
            graph->link(name+"_min", name);
            graph->link(name+"_max", name);
            graph->link(name+"_ratio", name);

        }
        else if(layer.type() == "PReLU") {
            Operator op(OpType_PRelu);
            const caffe::LayerParameter& param = getParam(name);
            const caffe::BlobProto& slopeBlob = param.blobs(0);
            Tensor slope(1, 1, slopeBlob.data_size(), FLOAT, slopeBlob.data().data());
            graph->createNode(name+"_slope", slope);
            graph->createNode(name, op);
            graph->link(name+"_slope", name);
        }
        else if(layer.type() == "Reshape") {
            Operator op(OpType_Reshape);
            const caffe::ReshapeParameter& reshape_param = layer.reshape_param();
            const caffe::BlobShape& bs = reshape_param.shape();
            switch(bs.dim_size()) {
                case 1:
                    op[Reshape::CHANNEL].i  = 1;
                    op[Reshape::HEIGHT].i   = 1;
                    op[Reshape::WIDTH].i    = bs.dim(0);
                    break;
                case 2:
                    op[Reshape::CHANNEL].i  = 1;
                    op[Reshape::HEIGHT].i   = bs.dim(0);
                    op[Reshape::WIDTH].i    = bs.dim(1);
                    break;
                case 3:
                    op[Reshape::CHANNEL].i  = bs.dim(0);
                    op[Reshape::HEIGHT].i   = bs.dim(1);
                    op[Reshape::WIDTH].i    = bs.dim(2);
                    break;
                case 4:
                    op[Reshape::CHANNEL].i  = bs.dim(1);
                    op[Reshape::HEIGHT].i   = bs.dim(2);
                    op[Reshape::WIDTH].i    = bs.dim(3);
                    break;
            }
            graph->createNode(name, op);
        }
        else if(layer.type() == "ROIPooling") {
            fprintf(stderr, "Not support %s\n", layer.type().c_str());
        }
        else if(layer.type() == "Dropout") {
            Operator op(OpType_Dropout);
            const caffe::DropoutParameter& dropProto = layer.dropout_param();
            op[Dropout::SCALE].f = 1.0;
            if (dropProto.has_dropout_ratio()) op[Dropout::SCALE].f = dropProto.dropout_ratio();
            graph->createNode(name, op);
        }
        else if(layer.type() == "Scale") {
            const caffe::ScaleParameter& scaleProto = layer.scale_param();
            const caffe::LayerParameter& param = getParam(name);
            const caffe::BlobProto& scaleBlob = param.blobs(0);
            int scale_size = 1;
            if (scaleBlob.has_shape()) {
                for(int d = 0; d < scaleBlob.shape().dim_size(); d++) {
                    scale_size *= scaleBlob.shape().dim(d);
                }
            }
            else {
                scale_size = scaleBlob.num() * scaleBlob.channels() *
                    scaleBlob.height() * scaleBlob.width();
            }
            Tensor scale(1, 1, scale_size, FLOAT, scaleBlob.data().data());
            if(scaleProto.has_bias_term()) {
                const caffe::BlobProto& biasBlob = param.blobs(1);
                Tensor bias(1, 1, scale_size, FLOAT, biasBlob.data().data());
                Operator op(OpType_Scale);
                graph->createNode(name+"_s_", scale);
                graph->createNode(name+"_b_", bias);
                graph->createNode(name, op);
                graph->link(name+"_s_", name);
                graph->link(name+"_b_", name);
            }
            else {
                Operator op(OpType_Mul);
                graph->createNode(name+"_s_", scale);
                graph->createNode(name, op);
                graph->link(name+"_s_", name);
            }
        }
        else if(layer.type() == "ShuffleChannel") {
            Operator op(OpType_ShuffleChannel);
            const caffe::ShuffleChannelParameter& shuffle_channel_param = layer.shuffle_channel_param();
            op[ShuffleChannel::GROUP].i = shuffle_channel_param.group();
            graph->createNode(name, op);
        }
        else if(layer.type() == "Sigmoid") {
            Operator op(OpType_Sigmoid);
            graph->createNode(name, op);
        }
        else if(layer.type() == "Slice") {
            const caffe::SliceParameter& sliceP = layer.slice_param();
            int point = sliceP.slice_point_size();
            for(int i = 0; i < layer.top_size(); i++ ) {
                Operator op(OpType_Slice);
                std::string top = layer.top(i);
                op[Slice::AXIS].i = sliceP.axis();
                if(point == 0) {
                    op[Slice::BEGIN].i = 0;
                    op[Slice::END].i = 0;
                    op[Slice::INDEX].i = i;
                    op[Slice::MAX].i = layer.top_size();
                }
                else {
                    op[Slice::BEGIN].i = (i==0)?0:sliceP.slice_point().data()[i-1];
                    op[Slice::END].i = sliceP.slice_point().data()[i];
                    op[Slice::INDEX].i = 0;
                    op[Slice::MAX].i = 0;
                }
                graph->createNode(top, op);
            }
        }
        else if(layer.type() == "Softmax") {
            Operator op(OpType_Softmax);
            graph->createNode(name, op);
        }
        else if(layer.type() == "TanH") {
            Operator op(OpType_Tanh);
            graph->createNode(name, op);
        }
        else if(layer.type() == "Threshold") { 
            fprintf(stderr, "Not support %s\n", layer.type().c_str());
        }
        else if(layer.type() == "Input") {
            const caffe::InputParameter& inputP = layer.input_param();
            auto shape = inputP.shape(0);
            switch(shape.dim_size()) {
                case 1:
                    w = shape.dim(0);
                    break;
                case 2:
                    h = shape.dim(0);
                    w = shape.dim(1);
                    break;
                case 3:
                    c = shape.dim(0);
                    h = shape.dim(1);
                    w = shape.dim(2);
                    break;
                case 4:
                    n = shape.dim(0);
                    c = shape.dim(1);
                    h = shape.dim(2);
                    w = shape.dim(3);
                    break;
            }
        }
        else {
            fprintf(stderr, "Not support %s\n", layer.type().c_str());
        }
        if(layer.bottom_size() == 1 && layer.top_size() == 1 &&
                layer.bottom(0) == layer.top(0)) {
            std::string bottom = layer.bottom(0);
            std::string top= layer.top(0);
            if(inplace[bottom] != "") {
                graph->link(inplace[bottom], name);
            }
            else {
                graph->link(bottom, name);
            }
            inplace[bottom] = name;
        }
        else if(layer.top_size() == 1){
            std::string top= layer.top(0);
            if(top != name ) {  // top !=  name
                top2name[layer.top(0)] = name;
            }
            for (int t = 0; t < layer.bottom_size(); ++t) {
                std::string bottom = layer.bottom(t);
                if(inplace[bottom] != "") {
                    graph->link(inplace[bottom], name);
                }
                else if(top2name[bottom] != "") {
                    graph->link(top2name[bottom], name);
                }
                else {
                    graph->link(bottom, name);
                }
            }
        }
        else if(layer.top_size() != 1 && layer.bottom_size() == 1){
            // we presume that size of top mush be 1.
            //std::string bottom = layer.bottom(0);
            //for (int t = 0; t < layer.top_size(); ++t) {
            //    std::string top = layer.top(t);
            //    graph->link(bottom, top);
            //}
        }
        else {
            fprintf(stderr, "link ERROR\n");
        }
    }
    return 0;
}
