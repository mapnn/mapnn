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

#ifndef __MAPNN_TEST_H__
#define __MAPNN_TEST_H__

#pragma once

#include "operator.h"
#include "conv.h"
#include "node.h"
#include "graph.h"
#include "bctime.h"

#include "hypothesis_test.h"

class test {
protected:
    struct Perf {
        const char* name;
        float ref_time;
        float opt_time;
        float test;
    };
public: 
    std::vector<test::Perf> run_test(Graph* graph, Tensor input_tensor, int cycle);
    virtual int run() = 0;
};
#endif // __MAPNN_TEST_H__
