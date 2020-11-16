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

#include "layer_debug.h"
#include <unistd.h>

class depthwise_debug : public layer_debug{
private:
    bool enable = true; 
    int kernel = 3;
    int stride = 1;
    int outch = 3;
    int inch = 4;
    int height = 4;
    int width = 4;
public:
    depthwise_debug(int argc, char** argv);
    int run()override;
};

depthwise_debug::depthwise_debug(int argc, char** argv) {
    int opt;
    const char *optstring = "k:s:c:S:h";

    while ((opt = getopt(argc-1, argv+1, optstring)) != -1) {
        switch(opt) {
            case 'k': 
                kernel = atoi(optarg);
                break;
            case 's': 
                stride = atoi(optarg);
                break;
            case 'S': 
                outch  = atoi(strtok(optarg,"," ));
                inch   = atoi(strtok(NULL,"," ));
                height = atoi(strtok(NULL,"," ));
                width  = atoi(strtok(NULL,"," ));
                break;
            case 'h': 
            default:
               printf(
                       "Usage: layer_debug %s -k 3 -s 3 -c 1 -g 2\n"
                       "Option:\n" 
                       "       -k kernel INT\n"
                       "       -s stride INT\n"
                       "       -S Shape o,i,h,h INT,INT,INT,INT\n"
                       "       -h this help\n"
                     , argv[1]);
               enable = false;
        }
    }
}

int depthwise_debug::run() {
    if(!enable) return -1;
    bool needprint = (width<10&&height<10);
    Operator op(OpType_Conv);
    op[Conv::OUTCH].i = outch;
    op[Conv::INCH].i  = inch;
    op[Conv::WKERNEL].i = kernel;
    op[Conv::HKERNEL].i = kernel;
    op[Conv::WDILATION].i = 1;
    op[Conv::HDILATION].i = 1;
    op[Conv::WSTRIDE].i = stride;
    op[Conv::HSTRIDE].i = stride;
    op[Conv::GROUP].i = inch;
    Tensor input_tensor(inch, height, width, FLOAT);
    Tensor weight_tensor(1, 1, inch*outch*kernel*kernel, FLOAT);
    Tensor bias_tensor(1, 1, outch, FLOAT);
    input_tensor.fillRand(0, 4, true);
    weight_tensor.fillRand(0, 4, true);
    bias_tensor.fillRand(0, 4, true);
    //input_tensor.fill(1);
    //weight_tensor.fill(1);
    //bias_tensor.fill(0);
    Graph* graph = new Graph();
    Node* node = graph->createNode("node_op", op); // TODO link node
    Node* weight = graph->createNode("node_weight", weight_tensor);
    Node* bias = graph->createNode("node_bias", bias_tensor);
    graph->link(weight, node);
    graph->link(bias, node);
    run_debug(graph, input_tensor, !needprint);
    delete graph;
    return 0; 
}


