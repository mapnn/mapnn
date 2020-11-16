/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2019, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <sys/time.h>

#include "wino_trans_ker.h"
#include "wino_trans_inp.h"
#include "wino_sgemm.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

namespace TEngine {

namespace conv_2d_wino {

static void wino_trans_inp_kernel(const int i, const int tid, const void* step, int input_c, int cin_64, int cin_step,
                          const float* input, float* trans_inp, int block_w, int in_hw, int inw, int block_hw)
{
    int my_step = (( int* )step)[0];

    for(int idx = tid; idx < cin_64; idx += my_step)
    {
        int cin_start = idx * cin_step;
        int cin_end = cin_start + cin_step;
        cin_end = cin_end > input_c ? input_c : cin_end;

        tran_input_all(input, trans_inp, input_c, cin_start, cin_end, block_hw,block_w, in_hw, inw);
    }
}


}    // namespace conv_2d_wino
}    // namespace TEngine
