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

#ifndef __MAPNN_MEM_POOL_H__
#define __MAPNN_MEM_POOL_H__
#include <cstdlib>
#include <string>
#include <list>
#include <vector>
#include "node.h"
#define MAX_LIST 10
using std::vector;
using std::list;
class MemPool
{
private:
    typedef list<Node*> Queue;
    vector<Queue*> queues_;
    vector<Node*> consts_;
    vector<bool> queues_used_;
    vector<size_t> flux_length_;
    vector<size_t> temp_length_;
    vector<size_t> const_length_;
    vector<void*> flux_memory_;
    vector<void*> temp_memory_;
    vector<void*> const_memory_;
public:
    MemPool();
    ~MemPool();
    void insert(Node* node);
    void release(Node* node);
    void pushConst(Node* node);
    void show();
    void alloc();
    void free();
};

#endif // __MAPNN_MEM_POOL_H__
