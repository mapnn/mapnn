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

#ifndef __MAPNN_PRELU_H__
#define __MAPNN_PRELU_H__
#include "operator.h"

namespace mapnn {
class Reduction {
public:
    enum OP_MODE{
        MEAN        = 0,
        MAX         = 1,
        MIN         = 2,
        SUM         = 3,
    };
    enum OP_TYPE {
        MODE        = 0,
        KEEPDIMS    = 1,
        REDUCE_N    = 2,
        REDUCE_C    = 3,
        REDUCE_H    = 4,
        REDUCE_W    = 5,
    };
public:
    int mode;
    int n, c, h ,w, keepdims;
    Reduction(const Operator& op);
};
inline Reduction::Reduction(const Operator& op){
    mode        = op[MODE].i;
    keepdims    = op[KEEPDIMS].i;
    n           = op[REDUCE_N].i;
    c           = op[REDUCE_C].i;
    h           = op[REDUCE_H].i;
    w           = op[REDUCE_W].i;
}
}
#endif // __MAPNN_PRELU_H__
