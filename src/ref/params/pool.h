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

#ifndef __MAPNN_POOL_H__
#define __MAPNN_POOL_H__

#include "operator.h"

class Pool{
public:
    enum PAD_MODE {
        CEIL,
        FLOOR,
        SAME,
    };
    enum OP_TYPE {
        HPAD0       = 0,
        WPAD0       = 1,
        HPAD1       = 2,
        WPAD1       = 3,
        HKERNEL     = 4,
        WKERNEL     = 5,
        HSTRIDE     = 6,
        WSTRIDE     = 7,
        PADMODE     = 8,
        COUNT_PAD   = 9,
    };
public:
    int hpad0, wpad0;
    int hpad1, wpad1;
    int hstride, wstride;
    int hkernel, wkernel;
    int pad_mode = CEIL;
    bool count_pad = true;
    Pool(const Operator& op);
};
inline Pool::Pool(const Operator& op) {
    wpad0 = op[WPAD0].i;
    hpad0 = op[HPAD0].i;
    wpad1 = op[WPAD1].i;
    hpad1 = op[HPAD1].i;
    wkernel = op[WKERNEL].i;
    hkernel = op[HKERNEL].i;
    wstride = op[WSTRIDE].i;
    hstride = op[HSTRIDE].i;
    pad_mode = op[PADMODE].i;
    count_pad = op[COUNT_PAD].i != 0;
}
#endif // __MAPNN_POOL_H__
