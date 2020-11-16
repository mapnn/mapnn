cmake_minimum_required(VERSION 3.12)

mapnn_config_map_begin(OptimalStage)

mapnn_enable_map(arm/ncnn/map_conv1x1s1_sgemm_pack4_neon.h       OFF)
mapnn_enable_map(arm/ncnn/map_conv1x1s1_sgemm_pack4to1_neon.h    OFF)#op.oc%4!=0) error
mapnn_enable_map(arm/ncnn/map_conv1x1s1_sgemm_neon.h             ON)
mapnn_enable_map(arm/ncnn/map_conv1x1s1_neon.h                   ON)

mapnn_enable_map(arm/ncnn/map_conv1x1s2_pack4_neon.h             OFF)
mapnn_enable_map(arm/ncnn/map_conv1x1s2_sgemm_pack4to1_neon.h    OFF)
mapnn_enable_map(arm/ncnn/map_conv1x1s2_neon.h                   ON)

mapnn_enable_map(arm/ncnn/map_conv2x2s1_neon.h                   ON)

mapnn_enable_map(arm32/tengine/map_tengine_conv_2d_wino.h        ON)
mapnn_enable_map(arm/ncnn/map_conv3x3s1_pack1to4_neon.h          OFF) #op.ic%4!=0 error
mapnn_enable_map(arm/ncnn/map_conv3x3s1_winograd64_pack4_neon.h  OFF) #op.ic>=16&&op.oc>=16 error
mapnn_enable_map(arm/ncnn/map_conv3x3s1_winograd64_neon5.h       op.ic>=16&&op.oc>=16)
mapnn_enable_map(arm/ncnn/map_conv3x3s1_winograd64_neon4.h       op.ic>=16&&op.oc>=16)
mapnn_enable_map(arm/ncnn/map_conv3x3s1_neon.h                   ON)


mapnn_enable_map(arm/ncnn/map_conv3x3s2_pack1to4_neon.h          OFF) # op.iw>=4
mapnn_enable_map(arm/ncnn/map_conv3x3s2_pack4_neon.h             OFF) # op.iw>=4
mapnn_enable_map(arm/ncnn/map_conv3x3s2_neon.h                   ON)
mapnn_enable_map(cpu/mnn/map_ConvolutionWinogradF23.h            ON)
mapnn_enable_map(cpu/mnn/map_ConvolutionWinogradF63.h            ON)
mapnn_enable_map(cpu/mnn/map_ConvolutionTiledExecutorBasic.h     ON)
mapnn_enable_map(cpu/mnn/map_ConvolutionTiledExecutorBasic2.h    ON)
mapnn_enable_map(arm/ncnn/map_conv3x3s2_packed_neon.h            OFF)

mapnn_enable_map(arm/ncnn/map_conv4x4s4_neon.h                   ON)


mapnn_enable_map(arm/ncnn/map_conv5x5s1_pack4_neon.h             OFF)
mapnn_enable_map(arm/ncnn/map_conv5x5s1_neon.h                   ON)

mapnn_enable_map(arm/ncnn/map_conv5x5s2_pack4_neon.h             OFF)
mapnn_enable_map(arm/ncnn/map_conv5x5s2_neon.h                   ON)

mapnn_enable_map(arm/ncnn/map_conv7x7s1_neon.h                   ON)

mapnn_enable_map(arm/ncnn/map_conv7x7s2_pack1to4_neon.h          OFF)
mapnn_enable_map(arm/ncnn/map_conv7x7s2_neon.h                   ON)

mapnn_enable_map(arm32/tengine/map_tengine_conv_2d_dw.h          ON)
mapnn_enable_map(arm32/tengine/map_tengine_conv_2d_dw_3x3.h      OFF)

mapnn_enable_map(arm/ncnn/map_convdw3x3s1_pack4_neon.h           OFF)
mapnn_enable_map(arm/ncnn/map_convdw3x3s1_neon.h                 ON)
mapnn_enable_map(cpu/mnn/map_ConvolutionDepthwise3x3.h           ON)

mapnn_enable_map(arm/ncnn/map_convdw3x3s2_pack4_neon.h           OFF)
mapnn_enable_map(arm/ncnn/map_convdw3x3s2_neon.h                 ON)

mapnn_enable_map(arm/ncnn/map_convdw5x5s1_pack4_neon.h           OFF)
mapnn_enable_map(arm/ncnn/map_convdw5x5s1_neon.h                 ON)

mapnn_enable_map(arm/ncnn/map_convdw5x5s2_pack4_neon.h           OFF)
mapnn_enable_map(arm/ncnn/map_convdw5x5s2_neon.h                 ON)

mapnn_enable_map(arm/ncnn/map_conv_im2col_sgemm_neon.h           ON)
mapnn_enable_map(arm/ncnn/map_pooling2x2s2_max_neon.h            ON)
mapnn_enable_map(arm/ncnn/map_pooling3x3s2_max_neon.h            ON)
mapnn_enable_map(arm/ncnn/map_eltwise_add_neon.h                 ON)

mapnn_enable_map(cpu/mnn/map_convolution3x3_gemm.h               OFF)

mapnn_config_map_end(optimal_stage.h)
