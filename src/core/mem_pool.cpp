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

#include "mem_pool.h"
#include "memory.h"
namespace mapnn {
MemPool::MemPool() {
}
MemPool::~MemPool() {
    free();
}
void MemPool::pushConst(Node* node) {
    consts_.push_back(node);
}
void MemPool::insert(Node* node) {
    for(size_t i = 0; i < queues_.size(); i++) {
        if(*(queues_[i]->begin()) == node) return;
    }
    for(size_t i = 0; i < queues_.size(); i++) {
        if(!queues_used_[i]) {
            queues_[i]->push_back(node);
            queues_used_[i] = true;
            return;
        }
    }
    Queue* queue = new Queue();
    queue->push_back(node);

    queues_.push_back(queue);
    queues_used_.push_back(true);
}
void MemPool::release(Node* node) {
    for(size_t i = 0; i < queues_.size(); i++) {
        if(!queues_[i]->empty() && (*queues_[i]->rbegin()) == node) {
            queues_used_[i] = false;
            return;
        }
    }
}
void MemPool::show() {
    if(flux_length_.size() < queues_.size()) flux_length_.resize(queues_.size());
    if(flux_memory_.size() < queues_.size()) flux_memory_.resize(queues_.size());
    for(size_t i = 0; i < flux_length_.size(); i++) {
        printf("FLUXS: %lu\n", flux_length_[i]);
    }
    for(size_t i = 0; i < const_length_.size(); i++) {
        printf("CONST: %lu\n", const_length_[i]);
    }
    for(size_t i = 0; i < temp_length_.size(); i++) {
        if(temp_length_[i]!=0)printf("TEMPS: %lu\n", temp_length_[i]);
    }
    for(size_t i = 0; i < queues_.size(); i++) {
        printf("%lu\t(%lu)\t", i, flux_length_[i]);
        for(auto node : *queues_[i]) {
            const Tensor& tensor = node->getTensor();
            printf("\t  => %-11s(%lu)\n", node->name(), tensor.length());
        }
        printf("\n");
    }
}
void MemPool::alloc() {
    ////////// alloc flux memory
    flux_length_.resize(queues_.size());
    flux_memory_.resize(queues_.size());
    for(size_t i = 0; i < queues_.size(); i++) {
        for(auto node : *queues_[i]) {
            const Tensor& tensor = node->getTensor();
            if(flux_length_[i]<tensor.length()) flux_length_[i] = tensor.length();
        }
    }
    for(size_t i = 0; i < queues_.size(); i++) {
        flux_memory_[i] = fastMalloc(flux_length_[i]);
    }
    for(size_t i = 0; i < queues_.size(); i++) {
        for(auto node : *queues_[i]) {
            node->allocTensor(flux_memory_[i], flux_length_[i]);
        }
    }

    ////////// alloc const memory
    const_length_.resize(consts_.size());
    const_memory_.resize(consts_.size());
    for(size_t i = 0; i < consts_.size(); i++) {
        const Tensor& tensor = consts_[i]->getTensor();
        const_length_[i] = tensor.length();
        if(const_length_[i]) const_memory_[i] = fastMalloc(const_length_[i]);
        else const_memory_[i] = NULL;
    }
    for(size_t i = 0; i < consts_.size(); i++) {
        const Tensor& tensor = consts_[i]->getTensor();
        consts_[i]->allocTensor(const_memory_[i], tensor.length());
    }

    ////////// alloc temp memory
    temp_length_.resize(RES_TENSOR_NUM);
    temp_memory_.resize(RES_TENSOR_NUM);
    for(size_t i = 0; i < queues_.size(); i++) {
        for(auto node : *queues_[i]) {
            const Tensors& temps = node->getTemps();
            for(size_t t = 0; t < temps.size(); t++) {
                if(temp_length_[t]<temps[t].length()) {
                    temp_length_[t] = temps[t].length();
                }
            }
        }
    }
    for(size_t i = 0; i < consts_.size(); i++) {
        const Tensors& temps = consts_[i]->getTemps();
        for(size_t t = 0; t < temps.size(); t++) {
            if(temp_length_[t]<temps[t].length()) {
                temp_length_[t] = temps[t].length();
            }
        }
    }
    for(size_t t = 0; t < temp_length_.size(); t++) {
        temp_memory_[t] = fastMalloc(temp_length_[t]);
    }
    for(size_t i = 0; i < queues_.size(); i++) {
        for(auto node : *queues_[i]) {
            for(size_t t = 0; t < temp_memory_.size(); t++) {
                node->allocTemps(t, temp_memory_[t], temp_length_[t]);
            }
        }
    }
    for(size_t i = 0; i < consts_.size(); i++) {
        for(size_t t = 0; t < temp_memory_.size(); t++) {
            consts_[i]->allocTemps(t, temp_memory_[t], temp_length_[t]);
        }
    }
}
void MemPool::free() {
    for(size_t i = 0; i < queues_.size(); i++) {
        if(queues_[i]){
            queues_[i]->clear();
            delete queues_[i];
            queues_[i] = NULL;
        }
        if(0 != flux_length_[i] && NULL != flux_memory_[i]) {
            if(flux_memory_[i]) {
                fastFree(flux_memory_[i]);
                flux_memory_[i] = NULL;
            }
        }
    }
    for(size_t i = 0; i < const_memory_.size(); i++) {
        if(const_memory_[i]) {
            fastFree(const_memory_[i]);
            const_memory_[i] = NULL;
        }
    }
    for(size_t i = 0; i < temp_memory_.size(); i++) {
        if(temp_memory_[i]) {
            fastFree(temp_memory_[i]);
            temp_memory_[i] = NULL;
        }
    }
    queues_.clear();
    queues_used_.clear();
    flux_length_.clear();
    flux_memory_.clear();
    const_length_.clear();
    const_memory_.clear();
    temp_length_.clear();
    temp_memory_.clear();
}
}
