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
#ifndef __MAPNN_CONV_H__
#define __MAPNN_CONV_H__

#include "operator.h"

namespace mapnn {
class Conv {
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
        HSTRIDE     = 4,
        WSTRIDE     = 5,
        HDILATION   = 6,
        WDILATION   = 7,
        OUTCH       = 8,
        INCH        = 9,
        HKERNEL     = 10,
        WKERNEL     = 11,
        GROUP       = 12,
        PADMODE     = 13,
    };
public:
    int hpad0, wpad0;
    int hpad1, wpad1;
    int hstride, wstride;
    int hdilation, wdilation;
    int hkernel, wkernel;
    int outch, inch;
    int g;
    int pad_mode;
public:
    Conv(const Operator& op);
};
inline Conv::Conv(const Operator& op) {
    wpad0 = op[WPAD0].i;
    hpad0 = op[HPAD0].i;
    wpad1 = op[WPAD1].i;
    hpad1 = op[HPAD1].i;
    outch = op[OUTCH].i;
    inch = op[INCH].i;
    wkernel = op[WKERNEL].i;
    hkernel = op[HKERNEL].i;
    wdilation = op[WDILATION].i;
    hdilation = op[HDILATION].i;
    wstride = op[WSTRIDE].i;
    hstride = op[HSTRIDE].i;
    g = op[GROUP].i;
    pad_mode = op[PADMODE].i;
}
}
#endif // __MAPNN_CONV_H__
