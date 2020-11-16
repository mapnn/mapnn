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

#include "map.h"

DECLARE_KERNEL_MAP(map_pad_deconv);

inline bool map_pad_deconv::request(Operator& op) {
    return op.type == OpType_ConvTranspose;
}
inline bool map_pad_deconv::run(Graph* graph, Node* node) {
    std::string nodename = node->name();
    Operator op = node->getOp();
    if(op[Conv::WPAD0].i != 0 ||
            op[Conv::HPAD0].i != 0 ||
            op[Conv::WPAD1].i != 0 ||
            op[Conv::HPAD1].i != 0) {
        Operator crop(OpType_Crop);
        crop[Crop::WCROP0].i = op[Conv::WPAD0].i;
        crop[Crop::HCROP0].i = op[Conv::HPAD0].i;
        crop[Crop::WCROP1].i = op[Conv::WPAD1].i;
        crop[Crop::HCROP1].i = op[Conv::HPAD1].i;
        std::string crop_name = node->name();
        crop_name+="_crop";
        Node* ncrop = graph->createNode(crop_name.c_str(), crop);
        node->sik_insert(ncrop);
    }
    node->setKernel(new RefConvTranspose());
    return true;

}
