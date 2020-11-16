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

#ifndef __MAPNN_GRAPH_H__
#define __MAPNN_GRAPH_H__

#include <list>
#include "node.h"
#include "mem_pool.h"

using std::list;

class Map;
class Graph
{
private:
    list<Node*> nodes_;
    list<Node*> sorts_;
    list<Node*> main_road_;
    list<Node*> side_road_;
    list<Node*> inputs_;
    list<Node*> outputs_;
    MemPool*  m_list_ = NULL;
public:
    Graph();
    ~Graph();
    Node* createNode(const string& name, Tensor& tensor);
    Node* createNode(const string& name, Operator op);
    Node* createNode(const string& name, OpType type);
    Node* createNode(const string& name);
    Node* cloneNode(Node* n);
    void releaseNode(Node* node);
    void show(const char* name);
    void link(const Node* p1, const Node* p2);
    void link(const string& name1, const string& name2);
    void link(const char* name1, const char* name2);
    void loosen(const Node* p1, const Node* p2);
    void loosen(const string& name1, const string& name2);
    void loosen(const char* name1, const char* name2);
    std::list<Node*>::iterator bInput();
    std::list<Node*>::iterator eInput();
    std::list<Node*>::iterator bOutput();
    std::list<Node*>::iterator eOutput();
    Graph* clone();
private:
    void sort_graph_();
    void init_tensor_(Tensor& input_tensor, bool setShape);
    Node* get_vertex_idx_(const string& name);
public:
    int allocMemory();
    int freeMemory();
    int mapping(std::vector<Map*>& maps);
    int mapping(Map* map);
    int mapping(Tensor& input_tensor, Map& map);
    int mapping(Tensor& input_tensor, Map* map);
    int inferShape(Tensor& input, bool setShape);
    int infer(Tensor& input);
    int inferSideRoad();
    int inferMainRoad(Tensor& input);
};
#endif // __MAPNN_GRAPH_H__
