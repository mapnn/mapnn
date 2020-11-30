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

void classification(std::string model, std::string img, 
        float mr, float mg, float mb, float vr, float vg, float vb, int cycle);

void printHelp() {
    printf(
            "Usage: modeltest -d dir -i image\n"
            "     : modeltest -m model -i image [Option]\n"
            "Option:\n" 
            "       -e aveg [0,0,0]\n"
            "       -v norms [1,1,1]\n"
            "       -C cycle\n"
            "       -h this help\n"
            );
}

int main(int argc, char** argv) {
    int opt;
    const char *optstring = "m:d:i:e:v:C:h";

    std::string dir;
    std::string input;
    std::string model;
    float er=0, eg=0, eb=0;
    float vr=1, vg=1, vb=1;
    int cycle = 1;

    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch(opt) {
            case 'm': 
                model = optarg;
                break;
            case 'd': 
                dir = optarg;
                break;
            case 'i': 
                input = optarg;
                break;
            case 'C': 
                cycle = atoi(optarg);
                break;
            case 'e': 
                sscanf(optarg, "%f,%f,%f", &er, &eg, &eb);
                break;
            case 'v': 
                sscanf(optarg, "%f,%f,%f", &vr, &vg, &vb);
                break;
            case 'h': 
            default:
                printHelp();
                return 0;
        }
    }
    if((dir.empty() || model.empty()) && input.empty() ) {
        printHelp();
        return -1;
    }

    if(!dir.empty()) {
            classification(dir+"/caffe/mobilenet.prototxt", input,
                    123.7, 116.3, 103.5, 0.01712, 0.01751, 0.01538, cycle);
            classification(dir+"/caffe/mobilenet_v2.prototxt", input,
                    123.7, 116.3, 103.5, 0.01712, 0.01751, 0.01538, cycle);
            classification(dir+"/caffe/resnet50v1.prototxt", input,
                    0.485, 0.456, 0.406, 0.229, 0.224, 0.225, cycle);
            classification(dir+"/caffe/googlenet.prototxt", input,
                    127.5, 127.5, 127.5, 1./128, 1./128, 1./128, cycle);
            classification(dir+"/caffe/inceptionV3.prototxt", input,
                    127.5, 127.5, 127.5, 1./128, 1./128, 1./128, cycle);
            classification(dir+"/caffe/inception_v4.prototxt", input,
                    127.5, 127.5, 127.5, 1./128, 1./128, 1./128, cycle);
            classification(dir+"/caffe/vgg16.prototxt", input,
                    0.485, 0.456, 0.406, 0.229, 0.224, 0.225, cycle);
            classification(dir+"/caffe/alexnet.prototxt", input,
                    0.485, 0.456, 0.406, 0.229, 0.224, 0.225, cycle);
            classification(dir+"/onnx/squeezenet1.1.onnx", input,
                    123.7, 116.3, 103.5, 0.01712, 0.01751, 0.01538, cycle);
            classification(dir+"/onnx/mobilenetv2-1.0.onnx", input,
                    123.7, 116.3, 103.5, 0.01712, 0.01751, 0.01538, cycle);
            classification(dir+"/onnx/resnet18v1.onnx", input,
                    123.7, 116.3, 103.5, 0.01712, 0.01751, 0.01538, cycle);
            classification(dir+"/onnx/resnet34v1.onnx", input,
                    123.7, 116.3, 103.5, 0.01712, 0.01751, 0.01538, cycle);
            classification(dir+"/onnx/resnet50v1.onnx", input,
                    123.7, 116.3, 103.5, 0.01712, 0.01751, 0.01538, cycle);
            classification(dir+"/onnx/resnet101v1.onnx", input,
                    123.7, 116.3, 103.5, 0.01712, 0.01751, 0.01538, cycle);
            classification(dir+"/onnx/resnet152v2.onnx", input,
                    123.7, 116.3, 103.5, 0.01712, 0.01751, 0.01538, cycle);
            //classification(dir+"/onnx/inception_v1.onnx", input,  // TODO:error
            //        123.7, 116.3, 103.5, 0.01712, 0.01751, 0.01538);
            classification(dir+"/onnx/alexnet.onnx", input,
                    0.f, 0.f, 0.f, 1.f, 1.f, 1.f, cycle);
            classification(dir+"/onnx/vgg19.onnx", input,
                    0.f, 0.f, 0.f, 1.f, 1.f, 1.f, cycle);
    }
    else if(!model.empty()) {
            classification(model, input, er, eg, eb, vr, vg, vb, cycle);
    }

    return 0;
}

void classification(std::string model, std::string img, float mr, float mg, float mb, float vr, float vg, float vb, int cycle = 1) {
    printf("********** %20s ********\n", model.c_str());
    int width;
    int height;
    int channel;
    unsigned char* input_data = stbi_load(img.c_str(), &width, &height, &channel, 3);

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

    float* float_data = new float[3*224*224];
    { 
        BCTime tr("prepare net");
        net->prepare(3, 224, 224);
        for(int i = 0; i < net->getTensorNum(); i++) {
            printf("output[%d]: %s\n", i, net->getTensorName(i));
        }
    }

    const float means[3] = {mr, mg, mb};
    const float norms[3] = {vr, vg, vb};

    float* p_float = float_data;
    for(int c = 0; c < 3; c++) {
        for(int h = 0; h < 224; h++) {
            for(int w = 0; w < 224; w++) {
                *p_float= norms[c]*(1.f*input_data[h*3*224 + w*3+ c]-means[c]);
                //if(c<1&&h<5&&w<10)printf("%6.2f ", *p_float);
                p_float++;
            }
            //if(c<1&&h<5)printf("\n");
        }
    }

    for(int i = 0; i < cycle; i++) {
        net->inference(float_data, 3, 224, 224);
    }


    {
        Tensor& output = net->getTensor("resnetv15_dense0_fwd");

        float* data = output.data();
        int max_i1 = 0, max_i2 = 0, max_i3 = 0;
        float max_f1 = 0, max_f2 = 0, max_f3 = 0;
        for(int k = 0; k < output.size(); k++) {
            if(*data > max_f1) {
                max_f1 = *data;
                max_i1 = k;
            }
            data++;
        }
        data = output.data();
        for(int k = 0; k < output.size(); k++) {
            if(*data > max_f2 && k != max_i1) {
                max_f2 = *data;
                max_i2 = k;
            }
            data++;
        }
        data = output.data();
        for(int k = 0; k < output.size(); k++) {
            if(*data > max_f3 && k != max_i1 && k != max_i2) {
                max_f3 = *data;
                max_i3 = k;
            }
            data++;
        }
        printf("(1) : %d %f\n", max_i1+1, max_f1);
        printf("(2) : %d %f\n", max_i2+1, max_f2);
        printf("(3) : %d %f\n", max_i3+1, max_f3);


    }
    if(net) delete net;
    if(float_data) delete[] float_data;
    stbi_image_free(input_data);
}
