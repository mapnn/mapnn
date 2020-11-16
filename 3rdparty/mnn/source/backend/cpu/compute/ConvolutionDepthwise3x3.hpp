

#ifndef ConvolutionDepthwise3x3_hpp
#define ConvolutionDepthwise3x3_hpp
namespace MNN {
void ConvolutionDepthwise3x3(const float* input_data, int inc4, int inh, int inw,
                             const float* weight_data, int wc, int wh, int ww,
                             float* temp_data, int tn, int tv, int ta, int tb,
                             float* output_data, int outc4, int outh, int outw);
} // namespace MNN

#endif /* ConvolutionDepthwise3x3_hpp */
