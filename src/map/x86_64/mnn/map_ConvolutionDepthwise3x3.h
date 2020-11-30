#include "map.h"
#include "mnn_kernel.h"

DECLARE_OPTIMAL_MAP(map_ConvolutionDepthwise3x3);

inline bool map_ConvolutionDepthwise3x3::request(Operator& op) {
    return op.type == OpType_Conv    &&
        op[Conv::WKERNEL].i == 3     &&
        op[Conv::HKERNEL].i == 3     &&
        op[Conv::WSTRIDE].i == 1     &&
        op[Conv::HSTRIDE].i == 1     &&
        op[Conv::WDILATION].i == 1   &&
        op[Conv::HDILATION].i == 1   &&
        op[Conv::GROUP].i == op.oc   &&
        op[Conv::GROUP].i == op.ic   &&
        op.oc % 4 == 0;
}
inline bool map_ConvolutionDepthwise3x3::run(Graph* graph, Node* node) {
    std::string nodename = node->name();
    Operator op = node->getOp();
    Operator convert(op, OpType_Convert);
    Node* w_pack = graph->createNode(nodename+"_w_", op);
    Node* i_pack = graph->createNode(nodename+"_i_", convert);
    Node* o_pack = graph->createNode(nodename+"_o_", convert);
    node->setKernel(new mnn_ConvolutionDepthwise3x3());
    w_pack->setKernel(new mnn_depthwise3x3Weight());
    i_pack->setKernel(new RefCHW1ToCHW4());
    o_pack->setKernel(new RefCHW4ToCHW1());
    node->src_insert(i_pack);
    node->cst_insert(w_pack);
    node->sik_insert(o_pack);
    return true;
}
