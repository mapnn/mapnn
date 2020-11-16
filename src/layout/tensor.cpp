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

#include "tensor.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cstdlib>

#include "time.h"
#include "memory.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

size_t get_unit_from_data_type(DataType dtype) { 
    switch(dtype) {
        case UINT8:
        case INT8:
            return 1u;
        case UINT16:
        case INT16:
            return 2u;
        case FLOAT:
        case INT32:
            return 4u;
        case INT64:
            return 8u;
        default:
            return 0u;
    }
}

Tensor::Tensor(): ref_(0),u_(0),v_(0),a_(0),b_(0),dtype_(UNDEFINED),data_(NULL) { }

Tensor::Tensor(const Tensor& t) { *this = t; }

Tensor::Tensor(const Tensor& t, const void* data):
    ref_(0),u_(t.u_),v_(t.v_),a_(t.a_),b_(t.b_),dtype_(t.dtype_),
    data_(const_cast<void*>(data)) {
    size_t unit = get_unit_from_data_type(dtype_);
    length_ = u_ * v_ * a_ * b_ * unit;
    ltype_  = t.ltype_;
    //printf("%d %d %d %d == %d %d %d %d\n", t.u_, t.v_, t.a_, t.b_, u_, v_, a_, b_);
}

Tensor::Tensor(int u, int v, int a, int b, DataType type, const void* data):
    ref_(0),u_(u),v_(v),a_(a),b_(b),dtype_(type),data_(const_cast<void*>(data)) {
    size_t unit = get_unit_from_data_type(type);
    length_ = u * v * a * b * unit;
}
Tensor::Tensor(int v, int a, int b, DataType type, const void* data):
    ref_(0),u_(1),v_(v),a_(a),b_(b),dtype_(type),data_(const_cast<void*>(data)) {
    size_t unit = get_unit_from_data_type(type);
    length_ = 1 * v * a * b * unit;
}
Tensor::Tensor(int u, int v, int a, int b, DataType type) {
    create_(u, v, a ,b, type);
}
Tensor::Tensor(int v, int a, int b, DataType type) {
    create_(1, v, a ,b, type);
}

Tensor Tensor::clone()const {
    Tensor t;
    t.create_(u_, v_, a_, b_, dtype_);
    size_t unit = get_unit_from_data_type(dtype_);
    size_t length = u_ * v_ * a_ * b_ * unit;
    if(length > 0) {
        memcpy(t.data_, data_, length);
    }
    return t;
}    
Tensor::~Tensor() {
    release_();
}
void Tensor::addref_()const {
    if (ref_) NNOKM_XADD(ref_, 1);
}
void Tensor::create_(int u, int v, int a, int b, DataType type) {
    if( u_ == u && v_ == v && a_  == a && 
        b_ == b && dtype_ == type) return;
    release_();
    u_      = u; 
    v_      = v; 
    a_      = a; 
    b_      = b; 
    dtype_   = type; 
    size_t unit = get_unit_from_data_type(type);
    length_ = alignSize(u_*v_*a_*b_*unit, 32);
    if(length_ > 0) {
        data_ = fastMalloc(length_ + (int)sizeof(*ref_));
        ref_  = (int*)(((unsigned char*)data_) + length_);
        *ref_ = 1;
    }
}
void Tensor::release_() {
    if (NULL != ref_ && NNOKM_XADD(ref_, -1) == 1) {
        if(data_)fastFree(data_);
    }
    ref_        = 0;
    u_          = 0;
    v_          = 0;
    a_          = 0;
    b_          = 0;
    dtype_      = FLOAT;
    ltype_      = L_UVAB;
    length_     = 0;
    data_       = NULL;
}
Tensor& Tensor::operator=(const Tensor& t) {
    if (this == &t) return *this;
    t.addref_();
    release_();
    ref_        = t.ref_;
    u_          = t.u_;
    v_          = t.v_;
    a_          = t.a_;
    b_          = t.b_;
    dtype_      = t.dtype_;
    ltype_      = t.ltype_;
    data_       = t.data_;
    length_     = t.length_;
    return *this;
}
void Tensor::fillRand(float min, float max, bool round) {
    srand(time(NULL));
    const int N = 99999;
    int size = u_*v_*a_*b_;
    float* p = (float*)data_;
    if(round){
        for(int i = 0; i < size; i++) {
            *p++ = (int)(rand() % (N + 1) / (float)(N + 1) * (max - min) - min);
        }
    }
    else{
        for(int i = 0; i < size; i++) {
            *p++ = rand() % (N + 1) / (float)(N + 1) * (max - min) - min;
        }
    }
}
void Tensor::fill(float v) {
    srand(time(NULL));
    int size = u_*v_*a_*b_;
    float* p = (float*)data_;
    for(int i = 0; i < size; i++) {
        *p++ = v;
    }
}
bool Tensor::valid() { 
    size_t unit = get_unit_from_data_type(dtype_);
    return (u_*v_*a_*b_*unit<=length_)&&(u_*v_*a_*b_*unit!=0); 
}
const int Tensor::size() const {
    return u_*v_*a_*b_;
}
const size_t Tensor::length() const {
    size_t unit = get_unit_from_data_type(dtype_);
    return u_*v_*a_*b_*unit; 
}


#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
