cmake_minimum_required(VERSION 3.12)

mapnn_config_map_begin(OptimalStage)

mapnn_enable_map(arm/ncnn/map_conv1x1s1_sgemm_pack4to1_neon.h     op.oc%4!=0) 
mapnn_enable_map(arm/ncnn/map_conv1x1s1_sgemm_neon.h              ON) 
mapnn_enable_map(arm/ncnn/map_conv1x1s1_neon.h                    ON) 
mapnn_enable_map(arm/ncnn/map_conv1x1s1_sgemm_pack4_neon.h        ON) 

mapnn_enable_map(arm/ncnn/map_conv1x1s2_neon.h                    op.oc<16&&op.ic<16) 
mapnn_enable_map(arm/ncnn/map_conv1x1s2_pack4_neon.h              OFF)  # low percision
mapnn_enable_map(arm/ncnn/map_conv1x1s2_sgemm_pack4to1_neon.h     OFF)  # low percision

mapnn_enable_map(arm/ncnn/map_conv2x2s1_neon.h                    ON) 

mapnn_enable_map(arm64/tengine/map_tengine_conv_2d_wino.h         op.ic>=60&&op.oc>=60) 
mapnn_enable_map(cpu/mnn/map_ConvolutionWinogradF63.h             op.ic>=16&&op.oc>=16) 
mapnn_enable_map(arm64/tengine/map_tengine_conv_2d_wino_1.h       op.ic<16&&op.oc<16) 
#cnn_enable_map(arm/ncnn/map_conv3x3s1_pack1to4_neon.h         op.ic%4!=0)  # TODO: error!?
mapnn_enable_map(arm/ncnn/map_conv3x3s1_winograd64_pack4_neon.h   op.ic>=16&&op.oc>=16) 
mapnn_enable_map(arm/ncnn/map_conv3x3s1_winograd64_neon5.h        op.ic>=16&&op.oc>=16) 
mapnn_enable_map(arm/ncnn/map_conv3x3s1_winograd64_neon4.h        op.ic>=16&&op.oc>=16) 
mapnn_enable_map(arm/ncnn/map_conv3x3s1_neon.h                    ON) 

mapnn_enable_map(arm/ncnn/map_conv3x3s2_pack1to4_neon.h           op.iw>=16) 
mapnn_enable_map(arm/ncnn/map_conv3x3s2_pack4_neon.h              op.iw>=4&&op.iw<=16) 
mapnn_enable_map(arm/ncnn/map_conv3x3s2_neon.h                    op.ic>=60&&op.oc>=60) 
mapnn_enable_map(cpu/mnn/map_ConvolutionWinogradF23.h             ON) 
mapnn_enable_map(cpu/mnn/map_ConvolutionTiledExecutorBasic.h      ON) 
mapnn_enable_map(cpu/mnn/map_ConvolutionTiledExecutorBasic2.h     ON) 
#cnn_enable_map(arm/ncnn/map_conv3x3s2_packed_neon.h           ON)  # TODO: error!?

mapnn_enable_map(arm/ncnn/map_conv4x4s4_neon.h                    ON) 

mapnn_enable_map(arm64/tengine/map_tengine_conv_2d_dw_k5s1.h      ON) 
mapnn_enable_map(arm/ncnn/map_conv5x5s1_pack4_neon.h              ON) 
mapnn_enable_map(arm/ncnn/map_conv5x5s1_neon.h                    ON) 

#cnn_enable_map(arm64/tengine/map_tengine_conv_2d_dw_k5s2.h    OFF)  # TODO: error!?
mapnn_enable_map(arm/ncnn/map_conv5x5s2_pack4_neon.h              ON) 
mapnn_enable_map(arm/ncnn/map_conv5x5s2_neon.h                    ON) 

#cnn_enable_map(arm64/tengine/map_tengine_conv_2d_dw_k7s1.h    OFF)  # TODO: error!?
mapnn_enable_map(arm/ncnn/map_conv7x7s1_neon.h                    ON) 

#cnn_enable_map(arm64/tengine/map_tengine_conv_2d_dw_k7s2.h    OFF)  # TODO: error!?
mapnn_enable_map(arm/ncnn/map_conv7x7s2_pack1to4_neon.h           ON) 
mapnn_enable_map(arm/ncnn/map_conv7x7s2_neon.h                    ON) 

mapnn_enable_map(arm/ncnn/map_convdw3x3s1_neon.h                  ON) 
mapnn_enable_map(cpu/mnn/map_ConvolutionDepthwise3x3.h            ON) 
mapnn_enable_map(arm/ncnn/map_convdw3x3s1_pack4_neon.h            OFF)   # TODO: error
mapnn_enable_map(arm/ncnn/map_convdw3x3s2_pack4_neon.h            OFF)   # TODO: error
mapnn_enable_map(arm/ncnn/map_convdw3x3s2_neon.h                  ON) 
mapnn_enable_map(arm64/tengine/map_tengine_conv_2d_dw_3x3.h       OFF)  # TODO: low percise for some shape

#cnn_enable_map(arm/ncnn/map_convdw5x5s1_pack4_neon.h          ON) 
mapnn_enable_map(arm/ncnn/map_convdw5x5s1_neon.h                  ON) 

#cnn_enable_map(arm/ncnn/map_convdw5x5s2_pack4_neon.h          ON) 
mapnn_enable_map(arm/ncnn/map_convdw5x5s2_neon.h                  ON) 

mapnn_enable_map(arm64/tengine/map_conv_2d_direct_3x3_dilation.h  ON) 
mapnn_enable_map(arm64/tengine/map_tengine_conv_2d_dw.h           ON) 

#cnn_enable_map(arm64/tengine/map_tengine_conv_fast_gemm.h     ON) 
mapnn_enable_map(arm64/tengine/map_tengine_conv_fast.h            ON) 
mapnn_enable_map(arm/ncnn/map_conv_im2col_sgemm_neon.h            ON) 

mapnn_enable_map(arm/ncnn/map_pooling3x3s2_max_neon.h             ON) 
mapnn_enable_map(arm/ncnn/map_pooling2x2s2_max_neon.h             ON) 
mapnn_enable_map(arm/ncnn/map_eltwise_add_neon.h                  ON) 

mapnn_config_map_end(optimal_stage.h)
