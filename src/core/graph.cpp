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

#include "graph.h"
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <stack>
#include <map>
#include "map.h"
#include "bctime.h"

namespace mapnn {
std::string valid_string_(const char* in) {
    std::string valid("N");
    valid += in;
    while(valid.find('/') != string::npos) {
        valid = valid.replace(valid.find('/'),1, "_");
    }
    while(valid.find('.') != string::npos) {
        valid = valid.replace(valid.find('.'),1, "_");
    }
    while(valid.find('-') != string::npos) {
        valid = valid.replace(valid.find('-'),1, "_");
    }
    return valid;
}

Graph::Graph() {
    m_list_ = new MemPool();
}
Graph::~Graph() {
    for(auto node : nodes_) {
        delete node;
    }
    delete m_list_;
    nodes_.clear();
    sorts_.clear();
    main_road_.clear();
    side_road_.clear();
    inputs_.clear();
    outputs_.clear();
}
Node* Graph::createNode(const string& name, Tensor& tensor) {
    Node* node = new Node(name, tensor);
    nodes_.push_back(node);
    return node;
}
Node* Graph::createNode(const string& name, Operator op) {
    Node* node = new Node(name, op);
    nodes_.push_back(node);
    return node;
}
Node* Graph::createNode(const string& name, OpType type) {
    Operator op(type);
    Node* node = new Node(name, op);
    nodes_.push_back(node);
    return node;
}
Node* Graph::createNode(const string& name) {
    Node* node = new Node(name);
    nodes_.push_back(node);
    return node;
}
void Graph::releaseNode(Node* node) {
    nodes_.remove(node);
    delete node;
}
Node* Graph::cloneNode(Node* n) {
    Node* node =  n->clone();
    nodes_.push_back(node);
    return node;
}
void Graph::show(const char* name) {
#ifdef __DEBUG__
    FILE* file = fopen(name, "w");
    fprintf(file, "digraph G{\n");
    for(auto node : nodes_) {
        Tensor t = node->getTensor();
        Operator op = node->getOp();
        std::string valid = valid_string_(node->name());
        const char* name = valid.c_str(); 
        if(node->isConst()) {
            fprintf(file, 
                    "%s[shape=record,style=filled,fillcolor=green,fontsize=10,height=0.3," \
                    "label=\"{%s|output_shape: %dx%dx%dx%d|%d|%d}\"];\n",
                    name, name, t.u(), t.v(), t.a(), t.b(),t.layout(),op.type);
        }
        else if(node->isInput()) {
            fprintf(file, 
                    "%s[shape=record,style=filled,fillcolor=red,fontsize=10,height=0.3," \
                    "label=\"{%s|output_shape: %dx%dx%dx%d|%d|%d}\"];\n",
                    name, name, t.u(), t.v(), t.a(), t.b(),t.layout(),op.type);
        }
        else if(node->flag_side_road_)
            fprintf(file, "%s[shape=record,style=filled,fillcolor=green," \
                    "label=\"{%s|output_shape: %dx%dx%dx%d|%d|%d}\"];\n",
                    name, name, t.u(), t.v(), t.a(), t.b(),t.layout(),op.type);
        else {
            fprintf(file, "%s[shape=record,style=filled,fillcolor=lightblue," \
                    "label=\"{%s|output_shape: %dx%dx%dx%d|%d|%d}\"];\n",
                    name, name, t.u(), t.v(), t.a(), t.b(),t.layout(),op.type);
        }
    }
    for(auto node : nodes_) {
        list<Node*>* sinks = node->getSink();
        std::string valid = valid_string_(node->name());
        const char* name = valid.c_str(); 
        for(auto sink: *sinks) {
            std::string sink_valid = valid_string_(sink->name());
            const char* sink_name = sink_valid.c_str(); 
            if(node->isConst()) {
                fprintf(file,"%s->%s[weight=1];\n", name, sink_name);
            }
            else {
                fprintf(file,"%s->%s[weight=100];\n", name, sink_name);
            }
        }
    }
    fprintf(file, "}\n");
    fclose(file);
#endif
}
void Graph::link(const Node* p1, const Node* p2) {
    if(p1 == NULL || p2 == NULL) return;
    p1->sik_extend(p2);
    if(p1->isConst()) {
        p2->cst_extend(p1);
    }
    else p2->src_extend(p1);
}
void Graph::link(const string& name1, const string& name2) {
    Node* p1 = get_vertex_idx_(name1);
    Node* p2 = get_vertex_idx_(name2);
    if(p1 == NULL || p2 == NULL) return;
    link(p1, p2);
}
void Graph::link(const char* name1, const char* name2) {
    Node* p1 = get_vertex_idx_(name1);
    Node* p2 = get_vertex_idx_(name2);
    if(p1 == NULL || p2 == NULL) return;
    link(p1, p2);
}
void Graph::loosen(const Node* p1, const Node* p2) {
    if(p1 == NULL || p2 == NULL) return;
    p1->sik_reduce(p2);
    if(p1->isConst()) {
        p2->cst_reduce(p1);
    }
    else p2->src_reduce(p1);
}
void Graph::loosen(const string& name1, const string& name2) {
    Node* p1 = get_vertex_idx_(name1);
    Node* p2 = get_vertex_idx_(name2);
    if(p1 == NULL || p2 == NULL) return;
    loosen(p1, p2);
}
void Graph::loosen(const char* name1, const char* name2) {
    Node* p1 = get_vertex_idx_(name1);
    Node* p2 = get_vertex_idx_(name2);
    if(p1 == NULL || p2 == NULL) return;
    loosen(p1, p2);
}
std::list<Node*>::iterator Graph::bInput() {
    return inputs_.begin();
}
std::list<Node*>::iterator Graph::eInput() {
    return inputs_.end();
}
std::list<Node*>::iterator Graph::bOutput() {
    return outputs_.begin();
}
std::list<Node*>::iterator Graph::eOutput() {
    return outputs_.end();
}

void Graph::sort_graph_() {
    std::stack<Node*> s;
    sorts_.clear();
    outputs_.clear();
    side_road_.clear();
    main_road_.clear();
    for(auto node : nodes_) {
        node->flag_degree_ = node->getConst()->size() + node->getSource()->size();
        if(node->flag_degree_ == 0) {
            s.push(node);
        } 
    }
    while(!s.empty()) {
        Node* node = s.top(); s.pop();
        list<Node*>* sinks = node->getSink();
        for(auto out : *sinks) {
            if(!(--out->flag_degree_)) {
                s.push(out);

            }
        }
        if(!node->isConst())sorts_.push_back(node);
    }

    for(auto node : nodes_) {
        list<Node*>* src = node->getSource();
        list<Node*>* sink = node->getSink();
        if(!src->size() && !node->isConst() && !node->isInput()) {
            std::string name = node->name();
            Node* input = this->createNode((name+"_i").c_str());
            node->src_insert(input);
            inputs_.push_back(input);
        }
        if(!sink->size() && !node->isConst()) outputs_.push_back(node);
    }

    for(auto node : sorts_) {
        list<Node*>* sources = node->getSource();
        for(auto src : *sources) {
            if(!src->isConst() && !src->flag_side_road_) { 
                node->flag_side_road_ = false;
                break;
            }
            node->flag_side_road_ = true;
        }

        if(!node->isInput() && !node->flag_side_road_) {
            main_road_.push_back(node);
        }
        else if(node->flag_side_road_) {
            side_road_.push_back(node);
        }
    }

}

Node* Graph::get_vertex_idx_(const string& name) {
    for(auto node : nodes_) {
        if(name == node->name()) return node;
    }
    return NULL;
}
void Graph::init_tensor_(Tensor& input_tensor, bool setShape) {
    for(auto node : side_road_) {
        node->initTensor(setShape);
    }
    for(auto node : inputs_) {
        node->initTensor(input_tensor, setShape);
    }
    for(auto node : main_road_) {
        node->initTensor(setShape);
    }
}
int Graph::inferShape(Tensor& input, bool setShape) {
    sort_graph_();
    init_tensor_(input, setShape);
    return 0;
}
int Graph::mapping(Map* map) {
    bool running = false;
    for(auto node : nodes_) {
        Operator op = node->getOp();
        if(!node->isConst()) {
            if(map->request(op)) {
                running = true;
                MapStage n_stage = node->getStage();
                MapStage m_stage = map->getStage();
                if(n_stage >= m_stage) break;
                node->setStage(m_stage);
                map->run(this, node);
                node->setStage(n_stage);
            }
        }
    }
    if(!running) return -1;
    return 0;
}
int Graph::mapping(std::vector<Map*>& maps) {
    for(auto node : nodes_) {
        Operator op = node->getOp();
        if(!node->isConst()) {
            for(auto map : maps) {
                if(map->request(op) && map->option(op)) {
                    MapStage n_stage = node->getStage();
                    MapStage m_stage = map->getStage();
                    if(n_stage >= m_stage) break;
                    node->setStage(m_stage);
                    if(map->run(this, node)) break;
                    node->setStage(n_stage);
                }
            }
        }
    }
    return 0;
}
int Graph::allocMemory() {
    for(auto node : nodes_) {
        node->flag_degree_ = node->getSink()->size();
    }
    for(auto node : inputs_) {
        m_list_->insert(node);
        list<Node*>* sources = node->getSource();
        for(auto in : *sources) {
            if(!(--in->flag_degree_)&&!in->isConst()) {
                m_list_->release(in);
            }
        }
    }
    for(auto node : side_road_) {
        m_list_->pushConst(node);
    }
    for(auto node : main_road_) {
        m_list_->insert(node);
        list<Node*>* sources = node->getSource();
        for(auto in: *sources) {
            if(!(--in->flag_degree_)&&!in->isConst()) {
                m_list_->release(in);
            }
        }
    }
    m_list_->alloc();
    //m_list_->show();
    return 0;
}
int Graph::freeMemory() {
    m_list_->free();
    //m_list_->show();
    return 0;
}
int Graph::inferSideRoad() {
    for(auto node : side_road_) {
        //BCTime tr(node->name());
        node->run();
    }
    return 0;
}
int Graph::inferMainRoad(Tensor& input) {
    //std::map<int, float> perf;
    //float sum = 0;
    for(auto node : inputs_) {
        //BCTime tr;
        node->run(input);
        //perf[node->getOp().type] += tr.get();
        //sum += tr.get();
    }
    for(auto node : main_road_) {
        //BCTime tr;
        node->run();
        //perf[node->getOp().type] += tr.get();
        //sum += tr.get();
    }
    //for(auto t : perf) {
    //    printf("\tOP: %10d  => %5.2f(%5.2f%%)\n", t.first, t.second, t.second/sum);
    //}
    return 0;
}
Graph* Graph::clone() {
    Graph* g = new Graph();
    for(auto node : nodes_) {
        g->cloneNode(node);
    }
    for(auto node : nodes_) {
        for(auto sink :*(node->getSink())) {
            g->link(node->name(), sink->name());
        }
    }
    return g;
}
}
