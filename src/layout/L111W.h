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

#ifndef __MAPNN_L111W_H__
#define __MAPNN_L111W_H__
#include "LUVAB.h"
class L111W : protected LUVAB{
private:
    void create(const Tensor& t);
public:
    float* data;
    int w;
    L111W(Tensor& t);
    L111W(const Tensor& t);
    ~L111W() = default;
    bool empty() const;
    float operator[](int n) const;
};
inline L111W::L111W(const Tensor& t): LUVAB(t) {
    create(t);
}

inline bool L111W::empty() const{
    return w == 0;
}
inline float L111W::operator[](int n) const{
    if(data == NULL) return 0;
    return data[n];
}
inline void L111W::create(const Tensor& t) {
    data    = t.data();
    w       = LUVAB::u*LUVAB::v*LUVAB::a*LUVAB::b;
}
#endif // __MAPNN_L111W_H__
