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

#include "log.h"
#include "reference.h"
namespace mapnn {
void RefTranspose::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    LNCHW input(ins[0]); 
    LNCHW output(out); 
    Transpose transpose(op);
    if(transpose.n==0&&transpose.c==0&&transpose.h==1&&transpose.w==2) {
        output.n = input.n;
        output.c = input.c;
        output.h = input.h;
        output.w = input.w;
    }
    else if(transpose.n==0&&transpose.c==0&&transpose.h==2&&transpose.w==1) {
        output.n = input.n;
        output.c = input.c;
        output.h = input.w;
        output.w = input.h;
    }
    else if(transpose.n==0&&transpose.c==1&&transpose.h==0&&transpose.w==2) {
        output.n = input.n;
        output.c = input.h;
        output.h = input.c;
        output.w = input.w;
    }
    else if(transpose.n==0&&transpose.c==1&&transpose.h==2&&transpose.w==0) {
        output.n = input.n;
        output.c = input.w;
        output.h = input.c;
        output.w = input.h;
    }
    else if(transpose.n==0&&transpose.c==2&&transpose.h==1&&transpose.w==0) {
        output.n = input.n;
        output.c = input.w;
        output.h = input.h;
        output.w = input.c;
    }
    else if(transpose.n==0&&transpose.c==2&&transpose.h==0&&transpose.w==1) {
        output.n = input.n;
        output.c = input.w;
        output.h = input.c;
        output.w = input.h;
    }
    else if(transpose.n==1&&transpose.c==3&&transpose.h==4&&transpose.w==2) {
        output.n = input.n;
        output.c = input.h;
        output.h = input.w;
        output.w = input.c;
    }
    else if(transpose.n==2&&transpose.c==1&&transpose.h==3&&transpose.w==4) {
        output.n = input.c;
        output.c = input.n;
        output.h = input.h;
        output.w = input.w;
    }
    else {
        LOGE("error %d %d %d %d\n", transpose.n, transpose.c, transpose.h, transpose.w);
    }
}
void RefTranspose::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Transpose transpose(op);
    LNCHW input(ins[0]); 
    LNCHW output(out); 

    if(transpose.n==0&&transpose.c==0 && transpose.h==1 && transpose.w==2) {
        const float* inptr = input.data;
        float* outptr = output.data;
        memcpy(outptr, inptr, input.chw*4);
    }
    else if(transpose.n==0&&transpose.c==0 && transpose.h==2 && transpose.w==1) {
        for(int oc = 0; oc < output.c; oc++) {
            float* outptr = output.data + output.hw*(oc);
            const float* inptr = input.data + input.hw*(oc);
            for(int oh = 0; oh < output.h; oh++) {
                for(int ow = 0; ow < output.w; ow++) {
                    *outptr++ = inptr[ow*output.h+oh];
                }
            }
        }
    }
    else if(transpose.n==0&&transpose.c==1 && transpose.h==0 && transpose.w==2) {
        for(int oc = 0; oc < output.c; oc++) {
            float* outptr = output.data + output.hw*(oc);
            for(int oh = 0; oh < output.h; oh++) {
                const float* inptr = input.data + input.hw*(oh);
                for(int ow = 0; ow < input.w; ow++) {
                    *outptr++ = inptr[oc*input.w+ow];
                }
            }
        }
    }
    else if(transpose.n==0&&transpose.c==1 && transpose.h==2 && transpose.w==0) {
        for(int oc = 0; oc < output.c; oc++) {
            float* outptr = output.data + output.hw*(oc);
            for(int oh = 0; oh < output.h; oh++) {
                const float* inptr = input.data + input.hw*(oh);
                for(int ow = 0; ow < input.w; ow++) {
                    *outptr++ = inptr[ow*input.c+oc];
                }
            }
        }
    }
    else if(transpose.n==0&&transpose.c==2 && transpose.h==1 && transpose.w==0) {
        for(int oc = 0; oc < output.c; oc++) {
            float* outptr = output.data + output.hw*(oc);
            for(int oh = 0; oh < output.h; oh++) {
                for(int ow = 0; ow < input.w; ow++) {
                    const float* inptr = input.data + input.hw*(ow);
                    *outptr++ = inptr[oh*input.c+oc];
                }
            }
        }
    }
    else if(transpose.n==0&&transpose.c==2 && transpose.h==0 && transpose.w==1) {
        for(int oc = 0; oc < output.c; oc++) {
            float* outptr = output.data + output.hw*(oc);
            for(int oh = 0; oh < output.h; oh++) {
                for(int ow = 0; ow < input.w; ow++) {
                    const float* inptr = input.data + input.hw*(ow);
                    *outptr++ = inptr[oc*input.h+oh];
                }
            }
        }
    }
    else if(transpose.n==1&&transpose.c==3 && transpose.h==4 && transpose.w==2) {
        for(int on = 0; on < output.n; on++) {
            for(int oh = 0; oh < output.h; oh++) {
                for(int ow = 0; ow < output.w; ow++) {
                    for(int oc = 0; oc < output.c; oc++) {
                        float* outptr = output.data + output.chw*on+output.hw*oh+output.w*ow+oc;
                        const float* inptr = input.data + input.chw*on+output.hw*oc+output.w*oh+ow;
                        *outptr = *inptr;
                    }
                }
            }
        }
    }
    else if(transpose.n==2&&transpose.c==1 && transpose.h==3 && transpose.w==4) {
        for(int on = 0; on < output.n; on++) {
            for(int oc = 0; oc < output.c; oc++) {
                float* outptr = output.data + output.hw*(on*output.c+oc);
                const float* inptr = input.data + input.hw*(oc*input.c+on);
                memcpy(outptr, inptr, input.hw*sizeof(float));
            }
        }
    }
    else {
        LOGE("transpose error\n");
    }
}
}
