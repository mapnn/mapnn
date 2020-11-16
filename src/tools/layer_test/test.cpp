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

#include "test.h"
#include "kernel_stage.h"
#include "optimal_stage.h"
std::vector<test::Perf> test::run_test(Graph* graph, Tensor input_tensor, int cycle) {
    std::vector<Perf> perfs;

    KernelStage ks;
    graph->mapping(ks.maps);
    graph->inferShape(input_tensor, true);
    graph->allocMemory();
    Tensor ori;
    float ori_time = 0;
    {
        graph->inferSideRoad();
        for(int i = 0; i < cycle; i++) {
            BCTime tr;
            graph->inferMainRoad(input_tensor);
            ori_time += tr.get();
        }
        Node* node = *(graph->bOutput());
        Tensor out = node->getTensor();
        ori = out.clone();
    }
    graph->freeMemory();

    OptimalStage os;
    for(size_t i = 0; i < os.maps.size(); i++) {
        Tensor dst;
        float dst_time = 0;
        const char* mname = os.maps[i]->name();
        Graph* g = graph->clone();
        {
            BCTime tr;
            int ret = g->mapping(os.maps[i]);
            if(0 != ret) { delete g; continue; }
        }
        g->show((std::string(mname)+".dot").c_str());

        g->allocMemory();
        {
            g->inferSideRoad();
            for(int j = 0; j < cycle; j++) {
                BCTime tr;
                g->inferMainRoad(input_tensor);
                dst_time += tr.get();
            }
            Node* node = *(g->bOutput());
            Tensor out = node->getTensor();

            dst = out.clone();
        }
        g->freeMemory();
        Perf perf;
        perf.name = mname;
        perf.ref_time = ori_time/cycle;
        perf.opt_time = dst_time/cycle;
        perf.test = chiTest(ori, dst);
        perfs.push_back(perf);

        delete g;

    }
    return perfs;
}
