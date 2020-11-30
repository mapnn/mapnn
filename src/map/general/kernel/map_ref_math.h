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

#include "map.h"

DECLARE_KERNEL_MAP(map_ref_math);

namespace mapnn {
inline bool map_ref_math::request(Operator& op) {
    return op.type == OpType_Add    ||
           op.type == OpType_Abs    ||
           op.type == OpType_Acos   ||
           op.type == OpType_Acosh  ||
           op.type == OpType_Asin   ||
           op.type == OpType_Asinh  ||
           op.type == OpType_Atan   ||
           op.type == OpType_Atanh  ||
           op.type == OpType_Ceil   ||
           op.type == OpType_Cos    ||
           op.type == OpType_Cosh   ||
           op.type == OpType_Exp    ||
           op.type == OpType_Floor  ||
           op.type == OpType_Log    ||
           op.type == OpType_Sin    ||
           op.type == OpType_Sinh   ||
           op.type == OpType_Sqrt   ||
           op.type == OpType_Tan    ||
           op.type == OpType_Tanh   ||
           op.type == OpType_Neg;
}
inline bool map_ref_math::run(Graph* graph, Node* node) {
    node->setKernel(new RefMath());
    return true;
}
}
