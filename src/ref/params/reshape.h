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

#ifndef __MAPNN_RESHAPE_H__
#define __MAPNN_RESHAPE_H__

#include "operator.h"

namespace mapnn {
class Reshape {
public:
    enum OP_TYPE {
        CHANNEL     = 0,
        HEIGHT      = 1,
        WIDTH       = 2,
    };
public:
    int channel, height, width;
    Reshape(const Operator& op);
};
inline Reshape::Reshape(const Operator& op) {
    channel = op[Reshape::CHANNEL].i;
    height = op[Reshape::HEIGHT].i;
    width = op[Reshape::WIDTH].i;
}
}
#endif // __MAPNN_RESHAPE_H__
