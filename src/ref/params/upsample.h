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

#ifndef __MAPNN_UPSAMPLE_H__
#define __MAPNN_UPSAMPLE_H__

#include "operator.h"

namespace mapnn {
class Upsample {
public:
    enum UPSAMPLE_TYPE{
        NEAREST         = 1,
        BILINEAR        = 2,
        BICUBIC         = 3,
    };
    enum OP_TYPE {
        HEIGHT          = 1,
        WIDTH           = 2,
        HEIGHT_SCALE    = 3,
        WIDTH_SCALE     = 4,
        UPSAMPLE_MODE     = 5,
        PAD_MODE        = 6,
    };
public:
    int height = 0, width = 0;
    float height_scale = 0.f, width_scale = 0.f;
    int resize_mode = 1;
    int pad_mode   = 1;
    Upsample(const Operator& op);
};
inline Upsample::Upsample(const Operator& op) {
    height          = op[Upsample::HEIGHT].i;
    width           = op[Upsample::WIDTH].i;
    height_scale    = op[Upsample::HEIGHT_SCALE].i;
    width_scale     = op[Upsample::WIDTH_SCALE].i;
    resize_mode     = op[Upsample::UPSAMPLE_MODE].i;
    pad_mode        = op[Upsample::PAD_MODE].i;
}
}
#endif // __MAPNN_UPSAMPLE_H__
