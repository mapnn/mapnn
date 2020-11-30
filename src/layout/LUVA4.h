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

#ifndef __MAPNN_LUVA4_H__
#define __MAPNN_LUVA4_H__

#include "LUVAB.h"
namespace mapnn {
class LUVA4 : protected LUVAB{
private:
    void create(const Tensor& t);
    void check();
public:
    float* data;
    int u, v, a4;
    int va4;
    LUVA4(Tensor& t);
    LUVA4(const Tensor& t);
    ~LUVA4();
};
inline LUVA4::LUVA4(Tensor& t): LUVAB(t){
    create(t);
}
inline LUVA4::LUVA4(const Tensor& t): LUVAB(t) {
    create(t);
    check();
}
inline LUVA4::~LUVA4() { 
    LUVAB::u = u;
    LUVAB::v = v;
    LUVAB::a = a4/4;
    LUVAB::b = 4;
    LUVAB::layout = L_UVA4;
}
inline void LUVA4::create(const Tensor& t) {
    data    = t.data();
    u       = LUVAB::u;
    v       = LUVAB::v;
    a4      = a*4;
    va4     = v*a4;
}
inline void LUVA4::check() {
    if(layout == L_UVA4) return;
    if(layout == L_UVAB && LUVAB::b == 4) return;
    LOGE("[ERROR] layout: LUVA4\n");
}
}
#endif // __MAPNN_LUVA4_H__
