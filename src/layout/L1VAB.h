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

#ifndef __MAPNN_L1VAB_H__
#define __MAPNN_L1VAB_H__

#include "LUVAB.h"
namespace mapnn {
class L1VAB : protected LUVAB{
private:
    void create(const Tensor& t);
    void check();
public:
    float* data;
    int u, v, a;
    int va, uva;
    L1VAB(Tensor& t);
    L1VAB(const Tensor& t);
    ~L1VAB();
};
inline L1VAB::L1VAB(Tensor& t): LUVAB(t) {
    create(t);
}
inline L1VAB::L1VAB(const Tensor& t): LUVAB(t) {
    create(t);
    check();
}
inline L1VAB::~L1VAB() { 
    LUVAB::u = u;
    LUVAB::v = v;
    LUVAB::a = a;
    LUVAB::b = 1;
    LUVAB::layout = L_UVA1;
}
inline void L1VAB::create(const Tensor& t) {
    data    = t.data();
    u       = LUVAB::u;
    v       = LUVAB::v;
    a       = LUVAB::a;
    va      = v*a;
    uva     = u*va;
}
inline void L1VAB::check() {
    if(layout == L_UVA1) return;
    if(layout == L_UVAB && LUVAB::b == 1) return;
    LOGE("[ERROR] layout: L_UVA1\n");
}
}
#endif // __MAPNN_L1VAB_H__
