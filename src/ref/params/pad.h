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

#ifndef __MAPNN_PAD_H__
#define __MAPNN_PAD_H__

#include "operator.h"

class Pad {
public:
    enum OP_MODE {
        CONSTANT,
        REFLECT,
        EDGE,
    };
    enum OP_TYPE {
        HPAD0       = 0,
        WPAD0       = 1,
        HPAD1       = 2,
        WPAD1       = 3,
        MODE        = 4,
        VALUE       = 5,
    };
public:
    int mode = Pad::CONSTANT;
    int hpad0, wpad0, hpad1, wpad1;
    float value = 0;
    Pad(const Operator& op);
};
inline Pad::Pad(const Operator& op){
    wpad0 = op[WPAD0].i;
    hpad0 = op[HPAD0].i;
    wpad1 = op[WPAD1].i;
    hpad1 = op[HPAD1].i;
    mode  = op[MODE].i;
    value = op[VALUE].f;
}
#endif // __MAPNN_PAD_H__
