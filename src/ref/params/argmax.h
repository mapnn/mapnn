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

#ifndef __MAPNN_ARGMAX_H__
#define __MAPNN_ARGMAX_H__

#include "operator.h"

class ArgMax {
public:
    enum OP_TYPE {
        OUT_MAX_VAL = 0,
        TOP_K       = 1,
        AXIS        = 2,
    };
public:
    int out_max_val = 0;
    int top_k = 1;
    int axis;
    ArgMax(const Operator& op);
};
inline ArgMax::ArgMax(const Operator& op){
    out_max_val = op[OUT_MAX_VAL].i;
    top_k = op[TOP_K].i;
    axis = op[AXIS].i;
}
#endif // __MAPNN_ARGMAX_H__
