
//
//  Matrix.hpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Matrix_hpp
#define Matrix_hpp

#include <stdio.h>
#include <memory>
namespace MNN {
namespace Math {
class Matrix {
public:
    static void multi(float* c, const float* a, const float* b, int h, int k, int w);
};
} // namespace Math
} // namespace MNN

#endif /* Matrix_hpp */
