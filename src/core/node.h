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

#ifndef __MAPNN_NODE_H__
#define __MAPNN_NODE_H__

#include <string>
#include <list>
#include <memory>

#include "tensor.h"
#include "operator.h"
#include "kernel.h"

#define RES_TENSOR_NUM 5
#define POSITION_BACK 100
#define POSITION_FRONT 0
using std::list;
using std::string;
using std::shared_ptr;

namespace mapnn {
class Node {
private:
    friend class Graph;
    int flag_degree_ = 0;
    bool flag_side_road_ = false;
private:
    const string name_;
    Tensor output_;
    Tensors temps_;
    Operator op_;
    shared_ptr<Kernel> kr_;
    bool isConst_ = false;
    bool isInput_ = false;
    list<Node*>* cst_;
    list<Node*>* src_;
    list<Node*>* sik_;
    Node(const string& name, Tensor& tensor);
    Node(const string& name, Operator op);
    Node(const string& name);
public:
    ~Node();
    list<Node*>* getConst();
    list<Node*>* getSource();
    list<Node*>* getSink();
    const char* name();
    const char* name() const;
    bool isConst() const;
    bool isInput() const;
    void setKernel(Kernel* kr);
    void setKernel(Kernel* kr, Operator op);
    void initTensor(bool setShape = false);
    void initTensor(Tensor& t, bool setShape = false);
    void allocTensor(void* buffer, size_t length);
    void allocTemps(int n, void* buffer, size_t length);
    void initMemory();
    void run();
    void run(Tensor& input);
    void deinit();
    const Operator& getOp();
    Tensor& getTensor();
    Tensors& getTemps();
    Node* clone();
public:
    const MapStage getStage();
    void setStage(MapStage stage);
public:
    void src_extend(const Node* node, size_t pos = POSITION_BACK) const;
    void cst_extend(const Node* node, size_t pos = POSITION_BACK) const;
    void sik_extend(const Node* node, size_t pos = POSITION_BACK) const;
    void src_reduce(const Node* node) const;
    void cst_reduce(const Node* node) const;
    void sik_reduce(const Node* node) const;
    Node* src_get(size_t pos = POSITION_BACK);
    Node* cst_get(size_t pos = POSITION_BACK);
    Node* sik_get(size_t pos = POSITION_BACK);
    int src_num();
    int cst_num();
    int sik_num();

    void src_insert(Node* node, size_t pos = POSITION_FRONT);
    void cst_insert(Node* node, size_t pos = POSITION_FRONT);
    void sik_insert(Node* node);
    Node* src_remove(size_t pos = POSITION_FRONT);
    Node* cst_remove(size_t pos = POSITION_FRONT);
    Node* sik_remove(size_t pos = POSITION_FRONT);
};
}
#endif // __MAPNN_NODE_H__
