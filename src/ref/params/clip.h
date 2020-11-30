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

#ifndef __MAPNN_CLIP_H__
#define __MAPNN_CLIP_H__

#include "operator.h"

namespace mapnn {
class Clip {
public:
    enum OP_TYPE {
        MAX       = 0,
        MIN       = 1,
    };
public:
    float max, min;
    Clip(const Operator& op);
};
inline Clip::Clip(const Operator& op){
    max = op[MAX].f;
    min = op[MIN].f;
}
}
#endif // __MAPNN_CLIP_H__
