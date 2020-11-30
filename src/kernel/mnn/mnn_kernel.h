#pragma once

#include "reference.h"
// x86
#ifdef MCNN_USE_SSE
#include <immintrin.h>
#define MNN_USE_SSE
#elif defined(MCNN_USE_NEON32) || defined(MCNN_USE_NEON64)
#include <arm_neon.h>
#define MNN_USE_NEON
#else
#error("none")
#endif

DECLARE_KERNEL(mnn_ConvolutionDepthwise3x3)
DECLARE_KERNEL(mnn_ConvolutionTiledExecutorBasic)
DECLARE_KERNEL(mnn_ConvolutionTiledExecutorBasic2)
DECLARE_KERNEL(mnn_ConvolutionWinogradF23)
DECLARE_KERNEL(mnn_ConvolutionWinogradF63)
DECLARE_KERNEL(mnn_convolution3x3_gemm)
DECLARE_KERNEL(mnn_depthwise3x3Weight)
DECLARE_KERNEL(mnn_reorderWeight)
DECLARE_KERNEL(mnn_transformWeightF23)
DECLARE_KERNEL(mnn_transformWeightF63)
