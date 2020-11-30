#include "map.h"
#include "mnn_kernel.h"

DECLARE_OPTIMAL_MAP(map_convolution3x3_gemm);

inline bool map_convolution3x3_gemm::request(Operator& op) {
    return op.type == OpType_Conv    &&
        op[Conv::WKERNEL].i == 3     &&
        op[Conv::HKERNEL].i == 3     &&
        op[Conv::WSTRIDE].i == 1     &&
        op[Conv::HSTRIDE].i == 1     &&
        op[Conv::WDILATION].i == 1   &&
        op[Conv::HDILATION].i == 1   &&
        op[Conv::GROUP].i == 1;
}
inline bool map_convolution3x3_gemm::run(Graph* graph, Node* node) {
    node->setKernel(new mnn_convolution3x3_gemm());
    return true;
}
