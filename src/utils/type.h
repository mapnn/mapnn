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

#ifndef __MAPNN_TYPE_H__
#define __MAPNN_TYPE_H__

typedef enum DataType {
    UNDEFINED   = 0,
    FLOAT       = 1,  // float
    UINT8       = 2,  // uint8_t
    INT8        = 3,  // int8_t
    UINT16      = 4,  // uint16_t
    INT16       = 5,  // int16_t
    INT32       = 6,  // int32_t
    INT64       = 7,  // int64_t
    STRING      = 8,  // string
    BOOL        = 9,  // bool
    FLOAT16     = 10, // IEEE754 half-precision floating-point format (16 bits wide).
    DOUBLE      = 11,
    UINT32      = 12,
    UINT64      = 13,
    COMPLEX64   = 14, // complex with float32 real and imaginary components
    COMPLEX128  = 15, // complex with float64 real and imaginary components
    BFLOAT16    = 16,
} DataType;

typedef enum LayoutType {
    L_UVAB,
    L_UVA4,
    L_UVA1,
    L_NCHW,
    L_NHWC,
    L_111W,
    L_1CHW,
    L_CHW4,
}LayoutType;

typedef enum MapStage {
    STAGE_NULL,
    STAGE_KERNEL,
    STAGE_FUSION,
    STAGE_OPTIMAL,
}MapStage;

typedef union Parameter{
    float f; 
    int i;
}Parameter;

#endif // __MAPNN_TYPE_H__
