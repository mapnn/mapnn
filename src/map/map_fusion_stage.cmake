cmake_minimum_required(VERSION 3.12)

mapnn_config_map_begin(FusionStage)

mapnn_enable_map(general/fusion/map_conv1x1s1_to_fc.h  ON)
mapnn_enable_map(general/fusion/map_conv_bn_scale.h    ON)
mapnn_enable_map(general/fusion/map_conv_bn.h          ON)
mapnn_enable_map(general/fusion/map_group_conv.h       ON)

mapnn_config_map_end(fusion_stage.h)
