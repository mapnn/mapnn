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
#ifndef __MAPNN_LRN_H__
#define __MAPNN_LRN_H__

#include "operator.h"

class LRN {
public:
    enum NormRegion {
        ACROSS_CHANNELS = 0,
        WITHIN_CHANNEL = 1,
    };
    enum OP_TYPE {
        LOCAL_SIZE  = 0,
        NORM_REGION = 1,
        ALPHA       = 2,
        BETA        = 3,
        BIAS        = 4,
    };
public:
    float alpha, beta, bias;
    int local_size = 1;
    int norm_region = ACROSS_CHANNELS;
    LRN(const Operator& op);
};
inline LRN::LRN(const Operator& op){
    alpha = op[ALPHA].f;
    beta = op[BETA].f;
    bias = op[BIAS].f;
    local_size  = op[LOCAL_SIZE].i;
    norm_region = op[NORM_REGION].i;

}
#endif // __MAPNN_LRN_H__
