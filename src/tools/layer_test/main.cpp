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

#include <unistd.h>

#include "conv_test.h"
#include "groupconv_test.h"
#include "depthwise_test.h"
#include "bctime.h"

void print_test_help() {
    printf(
           "Usage: layer_test layer [options]\n"
           "layer:\n" 
           "       conv\n"
           "       groupconv\n"
           "       depthwise\n"
           );
}

int main(int argc, char** argv) {
    if(argc <= 1) {
        print_test_help();
        return 0;
    }
    if(strcmp(argv[1], "conv") == 0) {
        conv_test t(argc, argv);
        return t.run();
    }
    else if(strcmp(argv[1], "groupconv") == 0) {
        groupconv_test t(argc, argv);
        return t.run();
    }
    else if(strcmp(argv[1], "depthwise") == 0) {
        depthwise_test t(argc, argv);
        return t.run();
    }
    else {
        print_test_help();
        return 0;
    }
}


