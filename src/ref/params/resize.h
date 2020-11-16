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

#ifndef __MAPNN_RESIZE_H__
#define __MAPNN_RESIZE_H__

#include "operator.h"

class Reshape {
public:
    enum OP_TYPE {
        HEIGHT          = 1,
        WIDTH           = 2,
        HEIGHT_SCALE    = 3,
        WIDTH_SCALE     = 4,
        RESIZE_MODE     = 5,
        PAD_MODE        = 6,
        INTERP_MODE     = 8,
    };
public:
    int height = 0, width = 0;
    float height_scale = 0.f, width_scale = 0.f;
    int resize_mode = 1;
    int interp_mode = 1;
    int pad_model   = 1;
    Reshape(const Operator& op);
};
inline Reshape::Reshape(const Operator& op) {
    height          = op[Reshape::HEIGHT].i;
    width           = op[Reshape::WIDTH].i;
    height_scale    = op[Reshape::HEIGHT_SCALE].i;
    width_scale     = op[Reshape::WIDTH_SCALE].i;
    resize_mode     = op[Reshape::RESIZE_MODE].i;
    pad_mode        = op[Reshape::PAD_MODE].i;
    interp_mode     = op[Reshape::INTERP_MODE].i;
}
#endif // __MAPNN_RESIZE_H__
