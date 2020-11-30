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

#ifndef __MAPNN_TRANSPOSE_H__
#define __MAPNN_TRANSPOSE_H__

#include "operator.h"

namespace mapnn {
class Transpose {
public:
    enum OP_TYPE {
        NTO         = 0,
        CTO         = 1,
        HTO         = 2,
        WTO         = 3,
    };
public:
    int n, c, h, w;
    Transpose(const Operator& op);
};
inline Transpose::Transpose(const Operator& op){
    n = op[NTO].i;
    c = op[CTO].i;
    h = op[HTO].i;
    w = op[WTO].i;
}
}
#endif // __MAPNN_TRANSPOSE_H__
