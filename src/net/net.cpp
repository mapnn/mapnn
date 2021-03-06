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

#include "net.h"

#include <vector>
#include <stack>

#include "model.h"
#include "onnx_model.h"
#include "caffe_model.h"
#include "graph.h"
#include "operator.h"
#include "bctime.h"
#include "kernel_stage.h"
#include "fusion_stage.h"
#include "optimal_stage.h"
namespace mapnn {
Net::Net()
{
    m_graph_ = new Graph();
}
Net::~Net() {
    delete m_model_;
    delete m_graph_;
}
int Net::load(const char* filepath, const char* filepath1) {
    if(m_model_) delete m_model_;
    m_model_ = new CaffeModel();
    return m_model_->load(filepath, filepath1);
}
int Net::load(const char* filepath) {
    if(m_model_) delete m_model_;
    m_model_ = new OnnxModel();
    return m_model_->load(filepath);
}
int Net::prepare(int channel, int height, int width) {
    m_model_->draw(m_graph_);
    if(!channel||!height||!width) {
        channel_ = m_model_->c;
        height_  = m_model_->h;
        width_   = m_model_->w;
    }
    else {
        channel_ = channel;
        height_  = height;
        width_   = width;
    }
    Tensor input_tensor(channel_, height_, width_, FLOAT, (void*)NULL);

    m_graph_->show("graph_initial.dot");
    {
        BCTime tr("graph Kernel Stage");
        KernelStage ks;
        m_graph_->mapping(ks.maps);
        m_graph_->inferShape(input_tensor, true);
    }

    m_graph_->show("graph_kernel.dot");
    {
        BCTime tr("graph Fusion Stage");
        FusionStage fs;
        m_graph_->mapping(fs.maps);
        m_graph_->inferShape(input_tensor, true);
    }

    m_graph_->show("graph_fusion.dot");
    {
        BCTime tr("graph optimal map");
        OptimalStage os;
        m_graph_->mapping(os.maps);
        m_graph_->inferShape(input_tensor, false);
    }

    m_graph_->show("graph.dot");

    {
        BCTime tr("graph alloc");
        m_graph_->allocMemory();
    }
    {
        BCTime tr("graph infer side load");
        m_graph_->inferSideRoad();
    }
    return 0;
}
int Net::inference(const float* data, int channel, int height, int width) {
    if(channel != channel_ || height != height_ || width != width_) return -1;
    Tensor input(channel, height, width, FLOAT, data);
    {
        BCTime tr("graph infer main load");
        m_graph_->inferMainRoad(input);
    }
    return 0;
}

int Net::channel() {
    return channel_;
}
int Net::height() {
    return height_;
}
int Net::width() {
    return width_;
}

int Net::getTensorNum() {
    int count = 0;
    for(auto it = m_graph_->bOutput(); it != m_graph_->eOutput(); ++it) {
        count++;
    }
    return count;
}
const char* Net::getTensorName(int n) {
    int count = 0;
    for(auto it = m_graph_->bOutput(); it != m_graph_->eOutput(); ++it) {
        if(count++ == n) return (*it)->name();
    }
    return (*(m_graph_->bOutput()))->name();
}
Tensor Net::getTensor(const char* name) {
    if(name == NULL) return (*(m_graph_->bOutput()))->getTensor();
    for(auto it = m_graph_->bOutput(); it != m_graph_->eOutput(); ++it) {
        if(0 == strcmp((*it)->name(), name)) {
            return (*it)->getTensor();
        }
    }
    return Tensor();
}
}
