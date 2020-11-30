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

#ifndef __MAPNN_CROP_H__
#define __MAPNN_CROP_H__

#include "operator.h"

namespace mapnn {
class Crop {
public:
    enum OP_TYPE {
        CCROP0       = 0,
        HCROP0       = 2,
        WCROP0       = 3,
        CCROP1       = 1,
        HCROP1       = 4,
        WCROP1       = 5,
        axis         = 6,
        offset0      = 6,
        offset1      = 7,
        offset2      = 8,
    };
public:
    int ccrop0, hcrop0, wcrop0;
    int ccrop1, hcrop1, wcrop1;
    Crop(const Operator& op);
};
inline Crop::Crop(const Operator& op){
    wcrop0 = op[WCROP0].i;
    hcrop0 = op[HCROP0].i;
    ccrop0 = op[CCROP0].i;
    wcrop1 = op[WCROP1].i;
    hcrop1 = op[HCROP1].i;
    ccrop1 = op[CCROP1].i;
}
}
#endif // __MAPNN_CROP_H__
