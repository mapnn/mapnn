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

#ifndef __MAPNN_HYPOTHESIS_TEST_H__
#define __MAPNN_HYPOTHESIS_TEST_H__
#include "macro.h"
namespace mapnn {
MAPNN_UNUSED static float chiTest(Tensor& ori, Tensor& dst) {
    if(ori.size() != dst.size()) return 1e19;
    double s1 = 0.f, s2 = 0.f;
    for(int i = 0; i < ori.size(); i++) {
        s1 += ori[i];
        s2 += dst[i];
    }
    double m1 = s1 / ori.size();
    double m2 = s2 / ori.size();
    double m  = (s1 + s2) / 2 / ori.size();
    double SST = 0.f, SSE = 0.f, SSA = 0.f;
    for(int i = 0; i < ori.size(); i++) {
        SST += (ori[i]-m)*(ori[i]-m) + (dst[i]-m)*(dst[i]-m);
        SSE += (ori[i]-m1)*(ori[i]-m1) + (dst[i]-m2)*(dst[i]-m2);
        SSA += (m1-m)*(m1-m) + (m2-m)*(m2-m);
    }
    double MSA = SSA;
    double MSE = SSE / (ori.size() - 2);
    if(MSE < 1e-9) return 0;
    double F = MSA / MSE;
    return F;
}

MAPNN_UNUSED static float kappaTest(Tensor& ori, Tensor& dst) {
    long long bin1[100] = {0}; 
    long long bin2[100] = {0}; 
    float max = -1e9, min = 1e9;
    for(int i = 0; i < ori.size(); i++) {
        float v1 = ori[i];
        float v2 = dst[i];
        if(max < v1) max = v1;
        if(min > v1) min = v1;
        if(max < v2) max = v2;
        if(min > v2) min = v2;
    }
    float step = (max - min) / 50;
    for(int i = 0; i < ori.size(); i++) {
        float v1 = ori[i];
        float v2 = dst[i];
        for(int n = 0; n < 50; n++) {
            if(min + n*step < v1 && v1 < min + (n+1)*step) {
                bin1[n]++;
                break;
            }
        }
        for(int n = 0; n < 50; n++) {
            if(min + n*step < v2 && v2 < min + (n+1)*step) {
                bin2[n]++;
                break;
            }
        }
    }

    float chi = 0;
    for(int i =0; i < 50; i++) {
        if(bin1[i]< 1e-9) continue;
        float scale = bin1[i] < 1e-5?1:1./bin1[i];
        chi += (float)(bin1[i] - bin2[i]) * (bin1[i] - bin2[i])*scale;
    }

    return chi;
}
}
#endif // __MAPNN_HYPOTHESIS_TEST_H__
