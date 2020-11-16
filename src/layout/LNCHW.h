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

#ifndef __MAPNN_LNCHW_H__
#define __MAPNN_LNCHW_H__

#include "LUVAB.h"
class LNCHW : protected LUVAB{
private:
    void create(const Tensor& t);
public:
    float* data;
    int n, c, h, w;
    int hw, chw, nchw;
    bool check();
    LNCHW(Tensor& t);
    LNCHW(const Tensor& t);
    ~LNCHW();
public:
    static bool check(const Tensor& t);
};
inline LNCHW::LNCHW(Tensor& t): LUVAB(t) {
    create(t);
}
inline LNCHW::LNCHW(const Tensor& t): LUVAB(t) {
    create(t);
    if(!check()) printf("[ERROR] layout: L_1CHW\n");
}
inline LNCHW::~LNCHW() { 
    LUVAB::u = n;
    LUVAB::v = c;
    LUVAB::a = h;
    LUVAB::b = w;
    LUVAB::layout = L_NCHW;
}
inline void LNCHW::create(const Tensor& t) {
    data    = t.data();
    n       = LUVAB::u;
    c       = LUVAB::v;
    h       = LUVAB::a;
    w       = LUVAB::b;
    hw      = h*w;
    chw     = c*hw;
    nchw    = n*chw;
}
inline bool LNCHW::check() {
    if(layout == L_1CHW) return true;
    if(layout == L_UVAB) return true;
    return false;
}
inline bool LNCHW::check(const Tensor& t) {
    LNCHW l(t);
    return l.check();
}
#endif // __MAPNN_LNCHW_H__
