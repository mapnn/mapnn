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

#ifndef __MAPNN_NET_H__
#define __MAPNN_NET_H__

#include "model.h"
#include "tensor.h"

namespace mapnn {
class Graph;
class Net
{
private:
    Graph* m_graph_ = NULL;
    Model* m_model_ = NULL;
public:
    Net();
    ~Net();
    bool load(const char* filepath);
    bool load(const char* filepath, const char* filepath1);
    bool prepare(int channel, int height, int width);
    bool inference(const float* data, int channle, int height, int width);
    bool tear();
    int getTensorNum();
    const char*  getTensorName(int n);
    Tensor& getTensor(const char* name);
};
}
#endif // __MAPNN_NET_H__
