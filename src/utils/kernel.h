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

#ifndef __MAPNN_KERNEL_H__
#define __MAPNN_KERNEL_H__

#include <vector>
#include "tensor.h"
#include "operator.h"
#include "LUVAB.h"
#include "LUVA4.h"
#include "L1VAB.h"
#include "L1CHW.h"
#include "LNCHW.h"
#include "LCHW4.h"
#include "L111W.h"
#include "L111W_s64.h"

typedef std::vector<Tensor> Tensors;

class Kernel {
public:
    Kernel() = default;
    ~Kernel() = default;
    virtual void init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) = 0;
    virtual void run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) = 0;
};

#define DECLARE_KERNEL(name)                                                        \
    class name : public Kernel{                                                     \
    public:                                                                         \
        virtual ~name() = default;                                                  \
        virtual void init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op)override;   \
        virtual void run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op)override;    \
    };

#define DECLARE_KERNEL_BASE(name, base)                                             \
    class name : public base{                                                       \
    public:                                                                         \
        virtual ~name() = default;                                                  \
        virtual void run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op)override;    \
    };


#define DECLARE_KERNEL_WITH_MATH(name, math)                                        \
    class name : public Kernel {                                                    \
        void init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op)override{           \
            out.copyShape(ins[0]);                                                  \
        }                                                                           \
        void run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {                   \
            L1CHW input(ins[0]);                                                    \
            L1CHW out(out);                                                         \
            const float* ptr = input.data;                                          \
            float* outptr = out.data;                                               \
            for(int c = 0; c < out.c; c++) {                                        \
                for(int h = 0; h < out.h; h++) {                                    \
                    for(int w = 0; w < out.w; w++) {                                \
                        *outptr++ = math(*ptr++);                                   \
                    }                                                               \
                }                                                                   \
            }                                                                       \
        }                                                                           \
    };

#endif //__MAPNN_KERNEL_H__
