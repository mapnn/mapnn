#include "map.h"

DECLARE_KERNEL_MAP(map_pad);

inline bool map_pad::request(Operator& op) {
    return op.type == OpType_Pad;
}
inline bool map_pad::run(Graph* graph, Node* node) {
    node->setKernel(new RefPad());
    return true;
}
