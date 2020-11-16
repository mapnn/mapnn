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

#ifndef __MAPNN_LUVAB_H__
#define __MAPNN_LUVAB_H__

#include "type.h"
class LUVAB{
private:
    void create(const Tensor& t);
protected:
    Tensor* ft = NULL;
    void* operator new(size_t) = delete;
    void* operator new[](size_t) = delete;
    void operator delete(void*) = delete;
    void operator delete[](void*) = delete;
    LUVAB& operator=(const LUVAB* l) = delete; 
    LUVAB& operator=(const LUVAB* l)const = delete; 
public:
    float* data;
    int u, v, a, b, ab, vab, uvab;
    LayoutType layout = L_UVAB;
    LUVAB(Tensor& t);
    LUVAB(const Tensor& t);
    ~LUVAB();
};
inline LUVAB::LUVAB(Tensor& t) { 
    create(t);
    ft=&t;
}

inline LUVAB::LUVAB(const Tensor& t){
    create(t);
}

inline LUVAB::~LUVAB() { 
    if(ft!=NULL) {
        if(u*v*a*b == 0) printf("[ERROR] empty layout. \n");
        ft->setShape(u, v, a, b);
        ft->setLayout(layout);
    } 
}
inline void LUVAB::create(const Tensor& t) {
    data    = t.data();
    u       = t.u();
    v       = t.v();
    a       = t.a();
    b       = t.b();
    ab      = a*b;
    vab     = v*ab;
    uvab     = u*vab;
}
#endif // __MAPNN_LUVAB_H__
