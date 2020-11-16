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

#ifndef __MAPNN_CAFFE_MODEL_H__
#define __MAPNN_CAFFE_MODEL_H__

#include "model.h"
#include "caffe.pb.h"
#include "node.h"
#include "graph.h"

class CaffeModel : public Model
{
private:
    caffe::NetParameter* m_model = NULL;
    caffe::NetParameter* m_prototxt= NULL;
    //std::map<std::string, const caffe::TensorProto*> tensor_node_list_;
private:
    const caffe::LayerParameter& getParam(std::string name);

public:
    CaffeModel();
    ~CaffeModel();
    int load(const char* prototxt)override {return false;}
    int load(const char* prototxt, const char* model)override;
    int draw(Graph* graph)override;
};
#endif // __MAPNN_CAFFE_MODEL_H__
