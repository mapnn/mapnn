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

#ifndef __MAPNN_OPERATOR_H__
#define __MAPNN_OPERATOR_H__

#define MAX_PARAMETER 20
#include "op_type_generated.h"
#include "type.h"

namespace mapnn {
class Operator{
private: 
    Parameter p[MAX_PARAMETER] = {{0}};
public:
    OpType type = OpType_Unkown;
    MapStage stage = STAGE_NULL;
    int in = 0, ic=1, ih=1, iw=1;
    int on = 0, oc=1, oh=1, ow=1;
    Operator(OpType t);
    Operator(const Operator& op);
    Operator(const Operator& op, OpType t);
    Operator() = default;
    Operator& operator=(const Operator& op);
    virtual ~Operator() = default;
    Parameter& operator[](int n);
    const Parameter& operator[](int n)const;
};

inline Operator& Operator::operator=(const Operator& op){
    if (this == &op) return *this;
    this->type = op.type;
    this->stage = op.stage;
    this->in = op.in;
    this->ic = op.ic;
    this->ih = op.ih;
    this->iw = op.iw;
    this->on = op.on;
    this->oc = op.oc;
    this->oh = op.oh;
    this->ow = op.ow;
    memcpy(p, op.p, MAX_PARAMETER*sizeof(Parameter));
    return *this;
}
inline Operator::Operator(const Operator& op) { *this = op; }
inline Operator::Operator(const Operator& op, OpType t) { *this = op;type=t;}
inline Parameter& Operator::operator[](int n){ return p[n]; }
inline const Parameter& Operator::operator[](int n)const{ return p[n]; }
inline Operator::Operator(OpType t):type(t){}
}
#endif // __MAPNN_OPERATOR_H__
