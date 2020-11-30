#include "map.h"
#include "mnn_kernel.h"

DECLARE_OPTIMAL_MAP(map_ConvolutionTiledExecutorBasic);

inline bool map_ConvolutionTiledExecutorBasic::request(Operator& op) {
    return op.type == OpType_Conv    &&
        op[Conv::WKERNEL].i == 3     &&
        op[Conv::HKERNEL].i == 3     &&
        op[Conv::WSTRIDE].i == 1     &&
        op[Conv::HSTRIDE].i == 1     &&
        op[Conv::WDILATION].i == 1   &&
        op[Conv::HDILATION].i == 1   &&
        op[Conv::GROUP].i == 1;
}
inline bool map_ConvolutionTiledExecutorBasic::run(Graph* graph, Node* node) {
    std::string nodename = node->name();
    Operator op = node->getOp();
    node->setKernel(new mnn_ConvolutionTiledExecutorBasic());
    Node* w_pack = graph->createNode(nodename+"_w_", op);
    Node* i_pack = graph->createNode(nodename+"_i4_", op);
    Node* o_pack = graph->createNode(nodename+"_o1_", op);
    w_pack->setKernel(new mnn_reorderWeight());
    i_pack->setKernel(new RefCHW1ToCHW4());
    o_pack->setKernel(new RefCHW4ToCHW1());
    node->src_insert(i_pack);
    node->cst_insert(w_pack);
    node->sik_insert(o_pack);
    return true;
}
