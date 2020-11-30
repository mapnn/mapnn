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

namespace mapnn {
class Resize {
public:
    enum RESIZE_MODE{
        NEAREST         = 1,
        LINEAR          = 2,
        CUBIC           = 3,
    };
    enum TRANSFORMATION_MODE{
        HALF_PIXEL          = 1,
        PYTORCH_HALF_PIXEL  = 2,
        ALIGN_CORNERS       = 3,
        ASYMMETRIC          = 4,
        TF_CROP_AND_RESIZE  = 5,
    };
    enum OP_TYPE {
        TRANSFORMATION_MODE     = 1,
        CUBIC_COEFF_A           = 2,
        EXCLUDE_OUTSIDE         = 3,
        EXTRAPOLATION_VALUE     = 4,
        MODE                    = 5,
        NEAREST_MODE            = 6,
    };
public:
    int coordinate_transformation_mode = 0;
    float cubic_coeff_a = -0.75f;
    int exclude_outside = 0;
    float extrapolation_value = 0.f;
    int mode = 0;
    int nearest_mode = 0;
    Resize(const Operator& op);
};
inline Resize::Resize(const Operator& op) {
    coordinate_transformation_mode  = op[TRANSFORMATION_MODE].i;
    cubic_coeff_a                   = op[CUBIC_COEFF_A].f;
    exclude_outside                 = op[EXCLUDE_OUTSIDE].i;
    extrapolation_value             = op[EXTRAPOLATION_VALUE].f;
    mode                            = op[MODE].i;
    nearest_mode                    = op[NEAREST_MODE].i;
}
}
#endif // __MAPNN_RESIZE_H__
