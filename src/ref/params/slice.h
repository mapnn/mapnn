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

#ifndef __MAPNN_SLICE_H__
#define __MAPNN_SLICE_H__

#include "operator.h"

class Slice {
public:
    enum OP_TYPE {
        AXIS            = 0,
        BEGIN           = 1,
        END             = 2,
        INDEX           = 3,
        MAX             = 4,
    };
public:
    int axis = 1, begin = 0, end = 0;
    int max = 0, index = 0;
    Slice(const Operator& op);
};
inline Slice::Slice(const Operator& op){
    axis  = op[AXIS].i;
    begin = op[BEGIN].i;
    end   = op[END].i;
    index = op[INDEX].i;
    max   = op[MAX].i;
}
#endif // __MAPNN_SLICE_H__
