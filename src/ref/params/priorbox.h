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

#ifndef __MAPNN_PRIORBOX_H__
#define __MAPNN_PRIORBOX_H__
#include "operator.h"

namespace mapnn {
class Priorbox {
public:
    enum OP_TYPE {
        VARIANCES0          = 0,
        VARIANCES1          = 1,
        VARIANCES2          = 2,
        VARIANCES3          = 3,
        FLIP                = 4,
        CLIP                = 5, 
        IMAGE_WIDTH         = 6,
        IMAGE_HEIGHT        = 7,
        STEP_WIDTH          = 8,
        STEP_HEIGHT         = 9,
        OFFSET              = 10,
        STEP_MMDETECTION    = 11,
        CENTER_MMDETECTION  = 12,
    };

public:
    float variances0 = 0.1f;
    float variances1 = 0.1f;
    float variances2 = 0.2f;
    float variances3 = 0.2f;
    int flip = 1;
    int clip = 0;
    int image_width  = 0;
    int image_height = 0;
    int step_width   = -1; 
    int step_height  = -1; 
    int offset = 0;
    int step_mmdetection = 0;
    int center_mmdetection = 0;
    Priorbox(const Operator& op);
};
inline Priorbox::Priorbox(const Operator& op){
    variances0              = op[VARIANCES0].f;
    variances1              = op[VARIANCES1].f;
    variances2              = op[VARIANCES2].f;
    variances3              = op[VARIANCES3].f;
    flip                    = op[FLIP].i;
    clip                    = op[CLIP].i;
    image_width             = op[IMAGE_WIDTH].i;
    image_height            = op[IMAGE_HEIGHT].i;
    step_width              = op[STEP_WIDTH].i;
    step_height             = op[STEP_HEIGHT].i;
    offset                  = op[OFFSET].i;
    step_mmdetection        = op[STEP_MMDETECTION].i;
    center_mmdetection      = op[CENTER_MMDETECTION].i;
}
}
#endif // __MAPNN_PRIORBOX_H__
