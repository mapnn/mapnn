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
#include "kernel_stage.h"
#include "optimal_stage.h"
#include "hypothesis_test.h"

void layer_debug::print(const char* name, Tensor t) {
    printf("## %s\n", name);
    float* p = t.data();
    if(t.b() == 1) {
        for(int c = 0; c < t.u(); c++) {
            printf("c = %d\n", c);
            for(int h = 0; h < t.v(); h++) {
                for(int w = 0; w < t.a(); w++) {
                    printf("%6.1f ", *p++); 
                }
                printf("\n");
            }
        }
    }
    else {
        for(int c = 0; c < t.v(); c++) {
            printf("c = %d\n", c);
            for(int h = 0; h < t.a(); h++) {
                for(int w = 0; w < t.b(); w++) {
                    printf("%6.1f ", *p++); 
                }
                printf("\n");
            }
        }
    }
}

void layer_debug::run_debug(Graph* graph, Tensor input_tensor, bool test) {
    KernelStage ks;
    graph->mapping(ks.maps);
    graph->inferShape(input_tensor, true);
    graph->allocMemory();
    Tensor base, dest;
    {
        graph->inferSideRoad();
        graph->inferMainRoad(input_tensor);
        Node* node = *(graph->bOutput());
        Tensor out = node->getTensor();
        base = out.clone();
        if(!test)print("base", out);
    }
    graph->freeMemory();

    OptimalStage os;
    for(size_t i = 0; i < os.maps.size(); i++) {
        const char* mname = os.maps[i]->name();
        Graph* g = graph->clone();
        {
            int ret = g->mapping(os.maps[i]);
            if(0 != ret) {
                delete g;
                continue;
            }
            //PadStage ps;
            //g->mapping(ps.maps);
            //g->inferShape(input_tensor, false);
        }
        g->show((std::string(mname)+".dot").c_str());

        g->allocMemory();
        {
            g->inferSideRoad();
            g->inferMainRoad(input_tensor);
            Node* node = *(g->bOutput());
            Tensor out = node->getTensor();
            if(!test)print(mname, out);
            if(test){
                float a = chiTest(base, out);
                printf("%-50s => %f\n", mname, a);
            }
            out.fill(0);
        }
        g->freeMemory();
        delete g;

    }
}
