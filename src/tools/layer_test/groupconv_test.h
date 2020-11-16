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

#include <string>
#include "test.h"
#include "bctime.h"

class groupconv_test : public test {
private:
    std::string output = "perf_groupconv.txt";
    bool enable = true; 
    int cycle = 1;
    int kernel = 3;
    int stride = 1;
    int group = 1;
    int gap = 0;
public:
    groupconv_test(int argc, char** argv);
    int run()override;
};

groupconv_test::groupconv_test(int argc, char** argv) {
    int opt;
    const char *optstring = "k:s:c:g:h:o";

    while ((opt = getopt(argc-1, argv+1, optstring)) != -1) {
        switch(opt) {
            case 'k': 
                kernel = atoi(optarg);
                break;
            case 's': 
                stride = atoi(optarg);
                break;
            case 'c': 
                cycle = atoi(optarg);
                break;
            case 'g': 
                group = atoi(optarg);
                break;
            case 'o':
               output = optarg; 
               break;
            case 'h': 
            default:
               printf(
                       "Usage: layer_test %s -k 3 -s 3 -c 1 -g 2\n"
                       "Option:\n" 
                       "       -k kernel INT\n"
                       "       -s stride INT\n"
                       "       -c cycle  INT\n"
                       "       -g group  INT\n"
                       "       -o output INT\n"
                       "       -h this help\n"
                     , argv[1]);
               enable = false;
        }
    }
}

int groupconv_test::run() {
    if(!enable) return -1;
    if(group <= 2) return -1;
    FILE* fp = fopen(output.c_str(), "w");
    for(int inh = 8; inh < 256; inh+=2) {
        printf("%d\n", inh);
        int inw = inh;
        for(int inch= 4; inch < 256; inch+=2) {
            for(int outch= 4; outch < 256; outch+=2) {
                if(inch%group!=0 || outch%group!=0) continue;
                if(2*inh*inh+inch*inch+outch*outch > 20000) continue;
                if(2*inh*inh+inch*inch+outch*outch < 100) continue;
                Operator op(OpType_Conv);
                op[Conv::OUTCH].i = outch;
                op[Conv::INCH].i  = inch;
                op[Conv::WKERNEL].i = kernel;
                op[Conv::HKERNEL].i = kernel;
                op[Conv::WDILATION].i = 1;
                op[Conv::HDILATION].i = 1;
                op[Conv::WSTRIDE].i = stride;
                op[Conv::HSTRIDE].i = stride;
                op[Conv::GROUP].i = group;
                Tensor input_tensor(inch, inh, inw, FLOAT);
                Tensor weight_tensor(1, 1, inch*outch*kernel*kernel, FLOAT);
                Tensor bias_tensor(1, 1, outch, FLOAT);
                input_tensor.fillRand();
                weight_tensor.fillRand();
                bias_tensor.fillRand();
                Graph* graph = new Graph();
                Node* node = graph->createNode("node_op", op); // TODO link node
                Node* weight = graph->createNode("node_weight", weight_tensor);
                Node* bias = graph->createNode("node_bias", bias_tensor);
                graph->link(weight, node);
                graph->link(bias, node);
                std::vector<Perf> perfs = run_test(graph, input_tensor, cycle);
                for(size_t p = 0; p < perfs.size(); p++) {
                    fprintf(fp,"%d,%d,%d,%d,%lu,%s,%6.5f,%6.5f,%6.5f\n", 
                            kernel, stride, inh, inch, perfs.size(),
                            perfs[p].name, perfs[p].ref_time, perfs[p].opt_time, perfs[p].test);
                }

                delete graph;
            }
        }
    }
    fclose(fp);
    return 0; 
}


