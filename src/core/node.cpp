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

#include "node.h"
#include "log.h"

namespace mapnn {
Node::Node(const string& name, Tensor& tensor):Node(name) {
    output_ = tensor;
    isConst_ = true;
    isInput_ = false;
}
Node::Node(const string& name, Operator op):Node(name) {
    op_ = op;
    isConst_ = false;
    isInput_ = false;
}
Node::Node(const string& name):
    name_(name),isConst_(false),isInput_(true) {
    temps_.resize(RES_TENSOR_NUM);
    cst_ = new list<Node*>();
    src_ = new list<Node*>();
    sik_ = new list<Node*>();
}
Node::~Node() {
    delete cst_;
    delete src_;
    delete sik_;
}
list<Node*>* Node::getConst() {
    return cst_;
}
list<Node*>* Node::getSource() {
    return src_;
}
list<Node*>* Node::getSink() {
    return sik_;
}
const char* Node::name() {
    return name_.c_str();
}
const char* Node::name() const{
    return name_.c_str();
}
bool Node::isConst() const {
    return isConst_;
}
bool Node::isInput() const {
    return isInput_;
}
void Node::setKernel(Kernel* kr) {
    kr_ = shared_ptr<Kernel>(kr);
}
void Node::setKernel(Kernel* kr, Operator op) {
    kr_ = shared_ptr<Kernel>(kr);
    op_ = op;
}
void Node::initTensor(bool setShape) {
    std::vector<Tensor> inputs(src_->size()+cst_->size()+RES_TENSOR_NUM);
    int count = 0;
    for(auto it = src_->begin(); it != src_->end(); ++it) {
        inputs[count++] = (*it)->getTensor();
    }
    for(auto it = cst_->begin(); it != cst_->end(); ++it) {
        inputs[count++] = (*it)->getTensor();
    }
    if(kr_ == NULL) { LOGE("Please  implement %s(%d)\n", name(), op_.type); exit(-1); } // TODO assert
    kr_->init(inputs, output_, temps_, op_);
    LOGDG("init[%5s]\t%-30s => (%d %d %d %d)\n", output_.valid()?"valid":"no", 
        name(), output_.u(), output_.v(), output_.a(), output_.b());
    if(setShape) {
        LNCHW input(inputs[0]);
        LNCHW output(output_);
        op_.in = input.n;
        op_.ic = input.c;
        op_.ih = input.h;
        op_.iw = input.w;
        op_.on = output.n;
        op_.oc = output.c;
        op_.oh = output.h;
        op_.ow = output.w;
    }
}
void Node::initTensor(Tensor& t, bool setShape) {
    if(isInput_) {
        output_ = t;
        output_.setLayout(L_1CHW);
        return;
    }
    std::vector<Tensor> inputs(src_->size()+cst_->size()+RES_TENSOR_NUM);
    int count = 1;
    inputs[0] = t;
    for(auto it = src_->begin(); it != src_->end(); ++it) {
        inputs[count++] = (*it)->getTensor();
    }
    for(auto it = cst_->begin(); it != cst_->end(); ++it) {
        inputs[count++] = (*it)->getTensor();
    }
    kr_->init(inputs, output_, temps_, op_);
    LOGDG("init[%5s]\t%-30s => (%d %d %d %d)\n", output_.valid()?"valid":"no",
         name(), output_.u(), output_.v(), output_.a(), output_.b());
    if(setShape) {
        LNCHW input(inputs[0]);
        LNCHW output(output_);
        op_.in = input.n;
        op_.ic = input.c;
        op_.ih = input.h;
        op_.iw = input.w;
        op_.on = output.n;
        op_.oc = output.c;
        op_.oh = output.h;
        op_.ow = output.w;
    }
}
void Node::allocTensor(void* buffer, size_t length) {
    output_ = Tensor(output_, buffer);
}
void Node::allocTemps(int n, void* buffer, size_t length) {
    if(n >= RES_TENSOR_NUM) return;
    temps_[n] = Tensor(temps_[n], buffer);
}
void Node::run()
{
    std::vector<Tensor> inputs(src_->size()+cst_->size()+RES_TENSOR_NUM);
    int count = 0;
    for(auto it = src_->begin(); it != src_->end(); ++it) {
        inputs[count++] = ((*it)->getTensor());
    }
    for(auto it = cst_->begin(); it != cst_->end(); ++it) {
        inputs[count++] = ((*it)->getTensor());
    }
    kr_->run(inputs, output_, temps_, op_);
    LOGDG("run [%5s]\t%-30s(%d) => (%d %d %d %d)\n", output_.valid()?"valid":"no",
            name(), op_.type, output_.u(), output_.v(), output_.a(), output_.b());
#ifdef PRINT_TENSOR
    std::string n(name()); 
    if(n.find('/') != string::npos) { 
        n = n.replace(n.find('/'),1, "_");
    }
    if(n.find('/') != string::npos) { 
        n = n.replace(n.find('/'),1, "_");
    }
    FILE* fp = fopen(("output/"+n).c_str(), "w");
    fprintf(fp, "%d %d %d %d\n", output_.u(), output_.v(), output_.a(), output_.b());
    int size = output_.v()*output_.a()*output_.b();
    for(int n = 0; n < output_.u(); n++) {
        float* p = (float*)output_.data() + n * size;
        for(int c = 0; c < output_.v(); c++) {
            for(int h = 0; h < output_.a(); h++) {
                for(int w = 0; w < output_.b(); w++) {
                    fprintf(fp, "(%d %d %d %d) %f\n", n, c, h ,w, *p++);
                }
            }
        }
    }
    fclose(fp);
#endif
}
void Node::run(Tensor& input)
{
    if(isInput_) {
        output_ = input;
        output_.setLayout(L_1CHW);
#ifdef PRINT_TENSOR
        std::string n(name()); 
        if(n.find('/') != string::npos) { 
            n = n.replace(n.find('/'),1, "_");
        }
        if(n.find('/') != string::npos) { 
            n = n.replace(n.find('/'),1, "_");
        }
        FILE* fp = fopen(("output/"+n).c_str(), "w");
        fprintf(fp, "%d %d %d %d\n", output_.u(), output_.v(), output_.a(), output_.b());
        int size = output_.v()*output_.a()*output_.b();
        for(int n = 0; n < output_.u(); n++) {
            float* p = (float*)output_.data() + n * size;
            for(int c = 0; c < output_.v(); c++) {
                for(int h = 0; h < output_.a(); h++) {
                    for(int w = 0; w < output_.b(); w++) {
                        fprintf(fp, "(%d %d %d %d) %f\n", n, c, h ,w, *p++);
                    }
                }
            }
        }
        fclose(fp);
#endif
        return;
    }
    std::vector<Tensor> inputs(src_->size()+cst_->size()+RES_TENSOR_NUM);
    int count = 0;
    for(auto it = src_->begin(); it != src_->end(); ++it) {
        inputs[count++] = ((*it)->getTensor());
    }
    for(auto it = cst_->begin(); it != cst_->end(); ++it) {
        inputs[count++] = ((*it)->getTensor());
    }
    kr_->run(inputs, output_, temps_, op_);
    LOGDG("run [%5s]\t%-30s(%d) => (%d %d %d %d)\n", output_.valid()?"valid":"no",
            name(), op_.type, output_.u(), output_.v(), output_.a(), output_.b());
#ifdef PRINT_TENSOR
    std::string n(name()); 
    if(n.find('/') != string::npos) { 
        n = n.replace(n.find('/'),1, "_");
    }
    if(n.find('/') != string::npos) { 
        n = n.replace(n.find('/'),1, "_");
    }
    FILE* fp = fopen(("output/"+n).c_str(), "w");
    fprintf(fp, "%d %d %d %d\n", output_.u(), output_.v(), output_.a(), output_.b());
    int size = output_.v()*output_.a()*output_.b();
    for(int n = 0; n < output_.u(); n++) {
        float* p = (float*)output_.data() + n * size;
        for(int c = 0; c < output_.v(); c++) {
            for(int h = 0; h < output_.a(); h++) {
                for(int w = 0; w < output_.b(); w++) {
                    fprintf(fp, "(%d %d %d %d) %f\n", n, c, h ,w, *p++);
                }
            }
        }
    }
    fclose(fp);
#endif
}
Node* Node::clone() {
    Node* node = new Node(name_);
    node->output_ = output_;
    node->op_     = op_;
    node->kr_     = kr_;
    node->kr_     = kr_;
    node->isConst_= isConst_;
    node->isInput_= isInput_;
    return node;

}

void Node::deinit() { }
Tensor& Node::getTensor() { return output_; }
Tensors& Node::getTemps() { return temps_; }
const Operator& Node::getOp() { return op_; }
const MapStage Node::getStage() {
    return op_.stage;
}
void Node::setStage(MapStage stage) {
    op_.stage = stage;
}
void Node::src_extend(const Node* node, size_t pos) const {
    Node* cnode = const_cast<Node*>(node);
    if(pos > src_->size()) src_->push_back(cnode);
    else if(pos < 0) src_->push_front(cnode);
    else {
        auto it = src_->begin();
        for(size_t i = 0; i < pos; i++){it++;}
        src_->insert(it, cnode);
    }
}
void Node::cst_extend(const Node* node, size_t pos) const {
    Node* cnode = const_cast<Node*>(node);
    if(pos > cst_->size()) cst_->push_back(cnode);
    else if(pos < 0) cst_->push_front(cnode);
    else {
        auto it = cst_->begin();
        for(size_t i = 0; i < pos; i++){it++;}
        cst_->insert(it, cnode);
    }
}
void Node::sik_extend(const Node* node, size_t pos) const {
    Node* cnode = const_cast<Node*>(node);
    if(pos > sik_->size()) sik_->push_back(cnode);
    else if(pos < 0) sik_->push_front(cnode);
    else {
        auto it = sik_->begin();
        for(size_t i = 0; i < pos; i++){it++;}
        sik_->insert(it, cnode);
    }
}
void Node::src_reduce(const Node* node) const {
    Node* cnode = const_cast<Node*>(node);
    src_->remove(cnode);        
}
void Node::cst_reduce(const Node* node) const {
    Node* cnode = const_cast<Node*>(node);
    cst_->remove(cnode);        
}
void Node::sik_reduce(const Node* node) const {
    Node* cnode = const_cast<Node*>(node);
    sik_->remove(cnode);        
}
Node* Node::src_get(size_t pos) {
    if(pos >= src_->size() || pos < 0) return NULL;
    auto it = src_->begin();
    for(size_t i = 0; i < pos; i++){it++;}
    return *it;
}
Node* Node::cst_get(size_t pos) {
    if(pos >= cst_->size() || pos < 0) return NULL;
    auto it = cst_->begin();
    for(size_t i = 0; i < pos; i++){it++;}
    return *it;
}
Node* Node::sik_get(size_t pos) {
    if(pos >= sik_->size() || pos < 0) return NULL;
    if(pos < 0) return sik_->front();
    auto it = sik_->begin();
    for(size_t i = 0; i < pos; i++){it++;}
    return *it;
}
int Node::src_num() { return src_->size();}
int Node::cst_num() { return cst_->size();}
int Node::sik_num() { return sik_->size();}
void Node::src_insert(Node* node, size_t pos) {
    if(pos < 0 || node == NULL) return;
    if(src_->size() != 0 && pos < src_->size()) {
        Node* front = this->src_get(pos);
        list<Node*>* front_sink = front->getSink();
        node->src_extend(front);
        node->sik_extend(this);
        for(auto it = front_sink->begin(); it != front_sink->end(); ++it) {
            if(*it == this) front_sink->insert(it, node);
        }
        front_sink->remove(this);
        for(auto it = src_->begin(); it != src_->end(); ++it) {
            if(*it == front)src_->insert(it, node);
        }
        src_->remove(front);
    }
    else {
        this->src_extend(node);
        node->sik_extend(this);
    }
}
void Node::cst_insert(Node* node, size_t pos) {
    if(pos < 0 || node == NULL) return;
    if(cst_->size() != 0 && pos < cst_->size()) {
        Node* front = this->cst_get(pos);
        list<Node*>* front_sink = front->getSink();
        node->src_extend(front);
        node->sik_extend(this);
        front_sink->push_back(node);
        front_sink->remove(this);
        src_->push_back(node);
        cst_->remove(front);
    }
    else {
        cst_->push_back(node);
        node->sik_extend(this);
    }
}
void Node::sik_insert(Node* node) {
    node->src_extend(this);
    if(sik_->size() == 0) {
        sik_->push_back(node);
        return ;
    }
    for(int pos = 0; pos < sik_->size(); pos++) {
        Node* back = this->sik_get(pos);
        list<Node*>* back_source = back->getSource();
        node->sik_extend(back);
        for(auto it = back_source->begin(); it != back_source->end(); ++it) {
            if(*it == this)  back_source->insert(it, node);
        }
        back_source->remove(this);
        for(auto it = sik_->begin(); it != sik_->end(); ++it) {
            if(*it == back) sik_->insert(it, node);
        }
        sik_->remove(back);
    }
    sik_->unique();
}
Node* Node::src_remove(size_t pos) {
    if(pos >= src_->size() || pos < 0) return NULL;
    Node* front = this->src_get(pos);
    front->sik_reduce(this);
    this->src_reduce(front);
    return front;
}
Node* Node::cst_remove(size_t pos) {
    if(pos >= cst_->size() || pos < 0) return NULL;
    Node* front = this->cst_get(pos);
    front->sik_reduce(this);
    this->cst_reduce(front);
    return front;
}
Node* Node::sik_remove(size_t pos) {
    if(pos >= cst_->size() || pos < 0) return NULL;
    Node* back = this->sik_get(pos);
    this->sik_reduce(back);
    back->src_reduce(this);
    for(int i = 0; i < back->sik_num(); i++) {
        Node* sik_sik = back->sik_get(i);
        this->sik_extend(sik_sik);
        sik_sik->src_reduce(back);
        sik_sik->src_extend(this, POSITION_FRONT);
    }
    return back;
}
}
