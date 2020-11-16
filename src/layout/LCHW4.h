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

#ifndef __MAPNN_LCHW4_H__
#define __MAPNN_LCHW4_H__

#include "LUVAB.h"
class LCHW4 : protected LUVAB{
private:
    void create(const Tensor& t);
public:
    float* data;
    int c, h, w4, hw4, chw4;
    bool check();
    LCHW4(Tensor& t);
    LCHW4(const Tensor& t);
    ~LCHW4();
public:
    static bool check(const Tensor& t);
};
inline LCHW4::LCHW4(Tensor& t): LUVAB(t) {
    create(t);
}
inline LCHW4::LCHW4(const Tensor& t): LUVAB(t) {
    create(t);
    if(!check()) printf("[ERROR] layout: LCHW4 %d(%d %d %d %d)\n", layout, c, h , w4/4, 4);
}
inline LCHW4::~LCHW4() { 
    LUVAB::u = c;
    LUVAB::v = h;
    LUVAB::a = w4/4;
    LUVAB::b = 4;
    LUVAB::layout = L_CHW4;
}
inline void LCHW4::create(const Tensor& t) {
    data    = t.data();
    c       = LUVAB::u;
    h       = LUVAB::v;
    w4      = LUVAB::a*4;
    hw4     = h*w4;
    chw4    = c*hw4;
}
inline bool LCHW4::check() {
    if(layout == L_CHW4) return true;
    if(layout == L_UVAB) return true;
    return false;
}
inline bool LCHW4::check(const Tensor& t) {
    LCHW4 l(t);
    return l.check();
}
#endif // __MAPNN_LCHW4_H__
