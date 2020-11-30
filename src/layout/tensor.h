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

#ifndef __MAPNN_TENSOR_H__
#define __MAPNN_TENSOR_H__

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "type.h"

namespace mapnn {
class LUVAB;
class Tensor {
protected:
    int* ref_ = NULL;
    int u_ = 0, v_ = 0, a_ = 0, b_ = 0;
    DataType dtype_ = UNDEFINED;
    LayoutType ltype_ = L_UVAB;
    void* data_ = NULL;
    size_t length_ = 0;
    void addref_()const;
    void release_();
    void create_(int u, int v, int a, int b, DataType type);

public:
    Tensor();
    Tensor(const Tensor& t);
    Tensor(const Tensor& t, const void* data);
    Tensor(int u, int v, int a, int b, DataType type, const void* data);
    Tensor(int v, int a, int b, DataType type, const void* data);
    Tensor(int u, int v, int a, int b, DataType type);
    Tensor(int v, int a, int b, DataType type);
    virtual ~Tensor();

public:
    DataType type();
    DataType type()const;
    LayoutType layout();
    LayoutType layout()const;
    int u();
    int v();
    int a();
    int b();
    int u() const;
    int v() const;
    int a() const;
    int b() const;
    float* data();
    float* data() const;
    Tensor& operator=(const Tensor& t);
    float& operator[](int i);
    const float& operator[](int i) const;
    Tensor clone() const;
    void fillRand(float min=0., float max=1., bool round=false);
    void fill(float v);
    const int size() const;
    const size_t length() const;
    void setShape(int u, int v, int a, int b);
    void setLayout(LayoutType layout);
    bool valid() const;
    bool empty() const;

};

inline void Tensor::setShape(int u, int v, int a, int b) { u_=u;v_=v;a_=a;b_=b;dtype_=FLOAT;}
inline void Tensor::setLayout(LayoutType type)   { ltype_ = type; }
inline DataType Tensor::type()                   { return dtype_;}
inline DataType Tensor::type() const             { return dtype_;}
inline LayoutType Tensor::layout()               { return ltype_;}
inline LayoutType Tensor::layout()const          { return ltype_;}
inline int Tensor::u()                           { return u_; }
inline int Tensor::v()                           { return v_; }
inline int Tensor::a()                           { return a_; }
inline int Tensor::b()                           { return b_; }
inline int Tensor::u() const                     { return u_; }
inline int Tensor::v() const                     { return v_; }
inline int Tensor::a() const                     { return a_; }
inline int Tensor::b() const                     { return b_; }
inline float* Tensor::data()                     {return (float*)data_;}
inline float* Tensor::data() const               {return (float*)data_;}
inline float& Tensor::operator[](int i)          { return ((float*)data_)[i]; }
inline const float& Tensor::operator[](int i)const { return ((const float*)data_)[i]; }
}
#endif // __MAPNN_TENSOR_H__
