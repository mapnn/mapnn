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

#ifndef __MAPNN_L1CHW_H__
#define __MAPNN_L1CHW_H__

#include "LUVAB.h"
class L1CHW : protected LUVAB{
private:
    void create(const Tensor& t);
public:
    float* data;
    int c, h, w;
    int hw, chw;
    bool check();
    L1CHW(Tensor& t);
    L1CHW(const Tensor& t);
    ~L1CHW();
public:
    static bool check(const Tensor& t);
};
inline L1CHW::L1CHW(Tensor& t): LUVAB(t) {
    create(t);
}
inline L1CHW::L1CHW(const Tensor& t): LUVAB(t) {
    create(t);
    if(!check()) printf("[ERROR] layout: L_1CHW\n");
}
inline L1CHW::~L1CHW() { 
    LUVAB::u = 1;
    LUVAB::v = c;
    LUVAB::a = h;
    LUVAB::b = w;
    LUVAB::layout = L_1CHW;
}
inline void L1CHW::create(const Tensor& t) {
    data    = t.data();
    c       = LUVAB::v;
    h       = LUVAB::a;
    w       = LUVAB::b;
    hw      = h*w;
    chw     = c*hw;
}
inline bool L1CHW::check() {
    if(layout == L_NCHW && LUVAB::u == 1) return true;
    if(layout == L_1CHW) return true;
    if(layout == L_UVAB) return true;
    return false;
}
inline bool L1CHW::check(const Tensor& t) {
    L1CHW l(t);
    return l.check();
}
#endif // __MAPNN_L1CHW_H__
