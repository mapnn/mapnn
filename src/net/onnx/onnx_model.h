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

#ifndef __MAPNN_ONNX_MODEL_H__
#define __MAPNN_ONNX_MODEL_H__

#include <vector>
#include <string>
#include <map>
#include "onnx.pb.h"
#include "node.h"
#include "graph.h"
#include "model.h"

namespace mapnn {
class OnnxModel : public Model
{
private:
    onnx::ModelProto* model_ = NULL;
    std::map<std::string, const onnx::TensorProto*> tensor_node_list_;
private:
    int create_tensor_node_(Graph* graph);
    int create_operater_node_(Graph* graph);
public:
    OnnxModel();
    ~OnnxModel();
    int load(const char* filepath)override;
    int load(const char* filepath, const char* filepath1)override {return false;}
    int draw(Graph* graph)override;
};
}
#endif // __MAPNN_ONNX_MODEL_H__
