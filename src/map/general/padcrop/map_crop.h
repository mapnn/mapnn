#include "map.h"

DECLARE_KERNEL_MAP(map_crop);

inline bool map_crop::request(Operator& op) {
    return op.type == OpType_Crop;
}
inline bool map_crop::run(Graph* graph, Node* node) {
    node->setKernel(new RefCrop());
    return true;
}
