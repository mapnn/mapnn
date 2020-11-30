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

#ifndef __MAPNN_POWER_H__
#define __MAPNN_POWER_H__

#include "operator.h"

namespace mapnn {
class Power {
public:
    enum OP_TYPE {
        POWER       = 0,
        SCALE       = 1,
        SHIFT       = 2,
    };
public:
    float power=1.f, scale=1.f, shift=0.f;
    Power(const Operator& op);
};
inline Power::Power(const Operator& op){
    power = op[POWER].f;
    scale = op[SCALE].f;
    shift = op[SHIFT].f;
}
}
#endif // __MAPNN_POWER_H__
