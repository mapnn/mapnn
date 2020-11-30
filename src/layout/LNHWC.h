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

#ifndef __MAPNN_LNHWC_H__
#define __MAPNN_LNHWC_H__

#include "LUVAB.h"
namespace mapnn {
class LNHWC : protected LUVAB{
private:
    void create(const Tensor& t);
public:
    float* data;
    int n, c, h, w;
    int wc, hwc, nhwc;
    bool check();
    LNHWC(Tensor& t);
    LNHWC(const Tensor& t);
    ~LNHWC();
public:
    static bool check(const Tensor& t);
};
inline LNHWC::LNHWC(Tensor& t): LUVAB(t) {
    create(t);
}
inline LNHWC::LNHWC(const Tensor& t): LUVAB(t) {
    create(t);
    if(!check()) LOGE("[ERROR] layout: L_1CHW\n");
}
inline LNHWC::~LNHWC() { 
    LUVAB::u = n;
    LUVAB::v = h;
    LUVAB::a = w;
    LUVAB::b = c;
    LUVAB::layout = L_NHWC;
}
inline void LNHWC::create(const Tensor& t) {
    data    = t.data();
    n       = LUVAB::u;
    h       = LUVAB::v;
    w       = LUVAB::a;
    c       = LUVAB::b;
    wc      = w*c;
    hwc     = h*wc;
    nhwc    = n*hwc;
}
inline bool LNHWC::check() {
    if(layout == L_UVAB) return true;
    return false;
}
inline bool LNHWC::check(const Tensor& t) {
    LNHWC l(t);
    return l.check();
}
}
#endif // __MAPNN_LNHWC_H__
