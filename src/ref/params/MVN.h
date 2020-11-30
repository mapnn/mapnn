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
#ifndef __MAPNN_MVN_H__
#define __MAPNN_MVN_H__

#include "operator.h"

namespace mapnn {
class MVN {
public:
    enum OP_TYPE {
        NORMALIZE_VARIANCE = 0,
        ACROSS_CHANNELS    = 1,
    };
public:
    bool normalize_variance, across_channels;
    MVN(const Operator& op);
};
inline MVN::MVN(const Operator& op){
    normalize_variance = op[NORMALIZE_VARIANCE].i != 0;
    across_channels    = op[ACROSS_CHANNELS].i != 0;
}
}
#endif // __MAPNN_MVN_H__
