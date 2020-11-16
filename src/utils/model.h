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

#ifndef __MAPNN_MODEL_H__
#define __MAPNN_MODEL_H__

class Graph;
class Model {
public:
    int n = 0, c = n, h = n, w = n;
    virtual int load(const char* filepath) = 0;
    virtual int load(const char* filepath, const char* filepath1) = 0;
    virtual int draw(Graph* graph) = 0;
    virtual ~Model() = default;
};

#endif // __MAPNN_MODEL_H__
