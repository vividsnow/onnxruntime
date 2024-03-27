// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <charconv>
#include <vector>
#include <utility>

#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "orttraining/core/optimizer/memory_optimizer/transformer_specific.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/tensorprotoutils.h"

#include "core/common/string_utils.h"

namespace onnxruntime::optimizer::memory_optimizer {

namespace {

std::tuple<bool, const Node*, const Node*> IsResidualNodeArg(const GraphViewer& graph_viewer, const NodeArg* node_arg) {
  auto consumers = graph_viewer.GetConsumerNodes(node_arg->Name());
  if (2 != consumers.size()) {
    return std::make_tuple(false, nullptr, nullptr);
  }

  // Find the Add node from the consumer list.
  const Node* add_node = nullptr;
  const Node* other_node = nullptr;
  for (const auto* consumer : consumers) {
    // At this point, there should not be any recompute node, so we don't need check the node existence in node_index_to_its_order_in_topological_sort_map.
    if (consumer->OpType() == "Add") {
      add_node = consumer;
    } else {
      other_node = consumer;
    }
  }

  return std::make_tuple(add_node != nullptr && other_node != nullptr, add_node, other_node);
}
}  // namespace

/*
    One classical layer includes 1). input layer norm, 2). attention, 3). residual add (input layer norm input + attention output),
    4). post attention layer norm feedforward, and 5). residual add (post attention layer norm input + feedforward out).

    The pattern graph looks like below for each transformer layer (taking the example of MistralDecoderLayer):
                            |
                        Embedding
                            |
                            |
      ----------------------|
      |                     |
      |                     |
      |        SimplifiedLayerNormalization (layer boudary node)
      |                     |
      |                     |
      |               MistralAttention
      |                     |
      |                     |
      |____________________Add
                            |
      ----------------------|
      |                     |
      |                     |
      |         SimplifiedLayerNormalization
      |                     |
      |                     |
      |            MultipleLayerPerception
      |                     |
      |                     |
      |____________________Add
                            |
                        (new layer)
      ----------------------|
      |                     |
      |                     |
      |         SimplifiedLayerNormalization (layer boudary node E we found earlier)
                                  ....

  Be noted: we need shift a bit around the layer boundary node S and E, as the layer boundary node S and E are not the real boundary nodes now.
  Specifically, we shift two nodes (PostBackwardFunction and ORTZeROOffloadPreForwardFunction) before S and E as the real boundary nodes.
*/
void FindLayerBoundaryLayerNormNodes(
    const GraphViewer& graph_viewer,
    const logging::Logger&,
    const InlinedHashMap<NodeIndex, ptrdiff_t>&
        node_index_to_its_order_in_topological_sort_map,
    const ptrdiff_t& yield_op_order_in_topological_sort,
    InlinedHashVector<const Node*>& layer_boundary_ln_nodes) {
  // Loop all nodes to find LayerNormalization nodes.
  // For each LayerNormalization node, keep checking its output nodes,
  // until find a node that is Softmax or BiasSoftmax or another LayerNormalization.
  // If the found node is Softmax or BiasSoftmax, the LayerNormalization node as ATTENTION.
  // If the found node is another LayerNormalization, the LayerNormalization node as MLP.
  const InlinedHashSet<std::string_view> softmax_ops{
      "Softmax",
      "BiasSoftmax",
  };
  const InlinedHashSet<std::string_view> layernorm_ops{
      "LayerNormalization",
      "SkipLayerNormalization",
      "SimplifiedLayerNormalization",
      "SkipSimplifiedLayerNormalization",
  };

  layer_boundary_ln_nodes.clear();
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED);
  for (auto node_index : node_topology_list) {
    auto& node = *graph_viewer.GetNode(node_index);

    if (layernorm_ops.find(node.OpType()) == layernorm_ops.end()) {
      continue;
    }

    const NodeArg* input_arg = node.InputDefs()[0];

    // IsResidualNodeArg checks input_arg
    auto [is_residual_node_arg, add_node, other_node] = IsResidualNodeArg(graph_viewer, input_arg);
    if (!is_residual_node_arg) {
      continue;
    }

    ptrdiff_t attention_residual_add_node_order = node_index_to_its_order_in_topological_sort_map.at(add_node->Index());
    ptrdiff_t attention_residual_ln_node_order = node_index_to_its_order_in_topological_sort_map.at(other_node->Index());
    if (attention_residual_add_node_order >= yield_op_order_in_topological_sort ||
        attention_residual_ln_node_order >= yield_op_order_in_topological_sort ||
        layernorm_ops.find(other_node->OpType()) == layernorm_ops.end()) {
      continue;
    }

    // IsResidualNodeArg checks add_node->OutputDefs()[0]
    auto [is_residual_node_arg_2, add_node_2, other_node_2] = IsResidualNodeArg(graph_viewer, add_node->OutputDefs()[0]);
    if (!is_residual_node_arg_2) {
      continue;
    }

    ptrdiff_t attention_residual_add_node_order_2 = node_index_to_its_order_in_topological_sort_map.at(add_node_2->Index());
    ptrdiff_t attention_residual_ln_node_order_2 = node_index_to_its_order_in_topological_sort_map.at(other_node_2->Index());
    if (attention_residual_add_node_order_2 >= yield_op_order_in_topological_sort ||
        attention_residual_ln_node_order_2 >= yield_op_order_in_topological_sort ||
        layernorm_ops.find(other_node_2->OpType()) == layernorm_ops.end()) {
      continue;
    }

    // Search all forward nodes that is before `add_node` in topo order, unless we find a softmax or nodes_to_check is empty.
    std::deque<const Node*> nodes_to_check;
    std::set<const Node*> visited_nodes;
    for (auto node_it = node.OutputNodesBegin(); node_it != node.OutputNodesEnd(); ++node_it) {
      // Ignore those nodes after YieldOp.
      auto order = node_index_to_its_order_in_topological_sort_map.at(node_it->Index());
      if (order < yield_op_order_in_topological_sort && order < attention_residual_add_node_order) {
        nodes_to_check.push_back(&(*node_it));
      }
    }

    while (!nodes_to_check.empty()) {
      const Node* next_node = nodes_to_check.front();
      nodes_to_check.pop_front();

      if (visited_nodes.find(next_node) != visited_nodes.end()) {
        continue;
      }

      visited_nodes.insert(next_node);
      if (softmax_ops.find(next_node->OpType()) != softmax_ops.end()) {
        layer_boundary_ln_nodes.push_back(&node);
        break;
      }

      for (auto node_it = next_node->OutputNodesBegin(); node_it != next_node->OutputNodesEnd(); ++node_it) {
        // Stop if the node is after next Layernorm node in execution order.
        auto order = node_index_to_its_order_in_topological_sort_map.at(node_it->Index());
        if (order < yield_op_order_in_topological_sort && order < attention_residual_add_node_order) {
          nodes_to_check.push_back(&(*node_it));
        }
      }
    }
  }
}

}  // namespace onnxruntime::optimizer::memory_optimizer
