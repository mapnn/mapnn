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

#include <float.h>
#include <stdio.h>
#include <limits.h>
#include <unistd.h>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize.h>

#include "net.h"
#include "bctime.h"

using namespace mapnn;

void printHelp() {
    printf(
            "Usage : modeltest -m model -i image [Option]\n"
            "Option:\n" 
            "       -m model\n"
            "       -i input image\n"
            "       -o output tensor default to first tensor.\n"
            "       -s shape of net, such as: 3,224,224 default to model shape. \n"
            "       -C cycle of inference, default to 1.\n"
            "       -a average of image, default to  0,0,0\n"
            "       -n norms of image, default to 1,1,1\n"
            "       -h this help\n"
            );
}

int main(int argc, char** argv) {

    int opt;
    const char *optstring = "m:d:i:a:n:s:o:C:h";

    std::string img;
    std::string model;
    std::string output_name;
    float er=0, eg=0, eb=0;
    float vr=1, vg=1, vb=1;
    int cycle = 1;
    int C=0, H=0, W=0;

    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch(opt) {
            case 'm': 
                model = optarg;
                break;
            case 'i': 
                img = optarg;
                break;
            case 'o': 
                output_name = optarg;
                break;
            case 'C': 
                cycle = atoi(optarg);
                break;
            case 's':
                sscanf(optarg, "%d,%d,%d", &C, &H, &W);
                break;
            case 'a': 
                sscanf(optarg, "%f,%f,%f", &er, &eg, &eb);
                break;
            case 'n': 
                sscanf(optarg, "%f,%f,%f", &vr, &vg, &vb);
                break;
            case 'h': 
            default:
                printHelp();
                exit(0);
        }
    }
    if(model.empty() || img.empty()) {
        printHelp();
        exit(-1);
    }

    std::string filename = model;
    std::string nfix = filename.substr(filename.find_last_of('.') + 1);
    std::string name = filename.substr(0, filename.find_last_of('.'));

    Net* net = NULL;
    { 
        BCTime tr("new net");
        net = new Net();
    }

    { 
        bool ret = 0;
        BCTime tr("load net");
        if(nfix == "onnx") {
            ret = net->load(model.c_str());
        }
        else if(nfix == "prototxt") {
            ret = net->load(model.c_str(), (name+".caffemodel").c_str());
        }
        if(!ret) exit(-1);
    }

    { 
        BCTime tr("prepare net");
        net->prepare();
    }

    if(!C||!H||!W) {
        C = net->channel();        
        H = net->height();        
        W = net->width();        
    }
    printf("XXXXXXXXXXXXXXXXXX %d %d %d\n", C, H, W);


    int width, height, channel;
    unsigned char* input_data = stbi_load(img.c_str(), &width, &height, &channel, 3);
    unsigned char* output_data = (unsigned char*)malloc(C* H* W);
    stbir_resize_uint8(input_data, width, height, 0,
            output_data, W, H, 0, C);
    float* float_data = new float[C*H*W];
    const float means[3] = {er, eg, eb};
    const float norms[3] = {vr, vg, vb};

    float* p_float = float_data;
    for(int c = 0; c < C; c++) {
        for(int h = 0; h < H; h++) {
            for(int w = 0; w < W; w++) {
                *p_float++ = norms[c]*(1.f*output_data[h*C*W + w*C+ c]-means[c]);
            }
        }
    }

    {
        FILE* fp = fopen("log.txt", "w");
        fprintf(fp, "%d %d %d\n", C, H, W);
        float* p = float_data;
        for(int c = 0; c < C; c++) {
            for(int h = 0; h < H; h++) {
                for(int w = 0; w < W; w++) {
                    fprintf(fp, "(%d %d %d) %f\n", c, h ,w, *p++);
                }
            }
        }
        fclose(fp);
    }


    for(int i = 0; i < cycle; i++) {
        net->inference(float_data, C, H, W);
    }
    Tensor output;
    if(!output_name.empty()) {
        output = net->getTensor(output_name.c_str());
    }
    else {
        output = net->getTensor();
    }

    float* data = output.data();
    int c_strip = output.v() * output.a();
    int h_strip = output.a();
    printf("%d %d %d %d\n", output.u(), output.v(), output.a(), output.b());
    for(int c = 0; c < output.u(); c++) {
        if(c > 3) continue;
        printf("c = %d \n",c);
        for(int h = 0; h < output.v(); h++) {
            if(h > 8) continue;
            for(int w = 0; w < output.a(); w++) {
                if(w > 15) continue;
                printf("%f ", data[c*c_strip+h*h_strip+w]);
            }
            printf("\n");
        }
    }

    return 0;
}
