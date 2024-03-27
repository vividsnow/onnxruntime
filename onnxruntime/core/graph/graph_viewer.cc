// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_viewer.h"
#include "core/graph/indexed_sub_graph.h"

namespace onnxruntime {

bool NodeCompare::operator()(const Node* n1, const Node* n2) const {
  return n1->Index() < n2->Index();
}

#if !defined(ORT_MINIMAL_BUILD)
struct PriorityNodeCompare {
  PriorityNodeCompare(
      const InlinedHashMap<std::string, const Node*>& fw_name_to_node_map,
      const InlinedHashMap<std::string, std::string>& bw_recompute_to_fw_node_map,
      const InlinedHashMap<std::string, float>& node_name_to_timestamp_map)
      : fw_name_to_node_map_(fw_name_to_node_map),
        bw_recompute_to_fw_node_map_(bw_recompute_to_fw_node_map),
        node_name_to_timestamp_map_(node_name_to_timestamp_map) {}

  inline bool IsHighPri(const Node* n) const {
    // local statics so we can compare std::strings in the checks
    static constexpr std::string_view shape_op("Shape");
    static constexpr std::string_view size_op("Size");

    const auto& op_type = n->OpType();
    return op_type == shape_op || op_type == size_op;
  }

  // Used for std::priority_queue
  // If return false, n1 will be output first
  // If return true, n2 will be output first
  bool operator()(const Node* n1, const Node* n2) const {
    // nodes in global high priority list will be output first
    const bool isN1HighPri = IsHighPri(n1);
    const bool isN2HighPri = IsHighPri(n2);
    if (isN1HighPri != isN2HighPri) {
      return isN2HighPri;
    }

    // nodes with lower priority value will be output first
    const auto n1_priority = n1->Priority();
    const auto n2_priority = n2->Priority();
    if (n1_priority != n2_priority) {
      return n1_priority > n2_priority;
    }

#ifdef ENABLE_TRAINING
    const std::string __critical_execution_order = "__critical_execution_order";
    // nodes of forward pass will be output first
    auto n1_attrs = n1->GetAttributes();
    auto n2_attrs = n2->GetAttributes();
    int64_t n1_is_forward = (n1_attrs.find(__critical_execution_order) != n1_attrs.cend())
                                ? static_cast<int64_t>(n1_attrs.at(__critical_execution_order).i())
                                : -1;
    int64_t n2_is_forward = (n2_attrs.find(__critical_execution_order) != n2_attrs.cend())
                                ? static_cast<int64_t>(n2_attrs.at(__critical_execution_order).i())
                                : -1;
    if (n1_is_forward != -1 && n2_is_forward != -1) {
      return n2_is_forward < n1_is_forward;
    }

    // Check whether node name ends with "_recompute"
    // if (bw_recompute_to_fw_node_map_.count(n1->Name()) && bw_recompute_to_fw_node_map_.count(n2->Name())) {
    //   const auto& fw_n1_pair = fw_name_to_node_map_.at(bw_recompute_to_fw_node_map_.at(n1->Name()));
    //   const auto& fw_n2_pair = fw_name_to_node_map_.at(bw_recompute_to_fw_node_map_.at(n2->Name()));
    //   return node_name_to_timestamp_map_.at(fw_n1_pair->Name()) < node_name_to_timestamp_map_.at(fw_n2_pair->Name());
    //   // auto t = PriorityNodeCompare(fw_name_to_node_map_, bw_recompute_to_fw_node_map_, node_name_to_timestamp_map_);
    //   // ORT_ENFORCE(fw_n1 != nullptr, "Node ", n1->Name(), " not found in bw_name_to_node_map");
    //   // ORT_ENFORCE(fw_n2 != nullptr, "Node ", n2->Name(), " not found in bw_name_to_node_map");
    //   // return !t(fw_n1_pair, fw_n2_pair);  // the earlier executed in fw, the latest we expect it run in bw.
    // }

#endif

    // otherwise, nodes with lower index will be output first
    // return n1->Index() > n2->Index();
    if (node_name_to_timestamp_map_.count(n1->Name()) > 0 && node_name_to_timestamp_map_.count(n2->Name()) > 0) {
      return node_name_to_timestamp_map_.at(n1->Name()) > node_name_to_timestamp_map_.at(n2->Name());
    }

    // std::cout << "Node " << n1->Name() << " or " << n2->Name() << " not found in node_name_to_timestamp_map" << std::endl;
    if (node_name_to_timestamp_map_.count(n1->Name()) <= 0) {
      std::cout << "Node " << n1->Name() << " not found in node_name_to_timestamp_map" << std::endl;
    }

    if (node_name_to_timestamp_map_.count(n2->Name()) <= 0) {
      std::cout << "Node " << n2->Name() << " not found in node_name_to_timestamp_map" << std::endl;
    }
    return n1->Index() > n2->Index();
  }

  const InlinedHashMap<std::string, const Node*>& fw_name_to_node_map_;
  const InlinedHashMap<std::string, std::string>& bw_recompute_to_fw_node_map_;
  const InlinedHashMap<std::string, float>& node_name_to_timestamp_map_;
};
#endif

GraphViewer::GraphViewer(const Graph& graph)
    : GraphViewer(graph, nullptr) {
}

GraphViewer::GraphViewer(const Graph& graph, const IndexedSubGraph& filter_info)
    : GraphViewer(graph, &filter_info) {
}

GraphViewer::GraphViewer(const Graph& graph, const IndexedSubGraph* filter_info)
    : graph_{&graph},
      // we can setup the filter here if needed. filtered_node_indices_ will have been populated by the time it's used
      graph_nodes_{graph_->FilteredNodes(
          filter_info ? [this](NodeIndex idx) { return filtered_node_indices_.count(idx) == 0; }
                      : ConstGraphNodes::NodeFilterFunc(nullptr))},
      filter_info_{filter_info} {
  std::vector<const Node*> leaf_nodes;
#ifdef ENABLE_TRAINING
  // Keep the info of shape and size nodes and their parents so that after topological sort, we can move them
  // right after their parents. This is to make sure the shape and size nodes are executed right after their parents
  // so it's possible the input tensor memory can be released as soon as possible. This is especially important
  // for non-CPU devices or for training case where some gradient graphs use only shape/size of tensors from forward.
  InlinedHashSet<NodeIndex> shape_size_nodes;
  InlinedHashMap<NodeIndex, InlinedVector<NodeIndex>> shape_size_parents;
#endif
  for (auto& node : graph_->Nodes()) {
    // This is a leaf node (without any output node)
    if (node.OutputNodesBegin() == node.OutputNodesEnd()) {
      leaf_nodes.push_back(&node);
    }
    // This is a root node (without any input node)
    if (node.InputEdgesBegin() == node.InputEdgesEnd()) {
      root_nodes_.push_back(node.Index());
    }
#ifdef ENABLE_TRAINING
    if ((node.OpType() == "Shape" || node.OpType() == "Size") && node.InputEdgesBegin() != node.InputEdgesEnd()) {
      shape_size_nodes.insert(node.Index());
      NodeIndex parent = node.InputNodesBegin()->Index();
      if (shape_size_parents.find(parent) == shape_size_parents.end()) {
        shape_size_parents[parent] = InlinedVector<NodeIndex>{node.Index()};
      } else {
        shape_size_parents[parent].push_back(node.Index());
      }
    }
#endif
  }

  graph.ReverseDFSFrom(
      leaf_nodes,
      nullptr,
      [this](const Node* n) {
        nodes_in_topological_order_.push_back(n->Index());
      },
      NodeCompare());
#ifdef ENABLE_TRAINING
  auto original = std::move(nodes_in_topological_order_);
  nodes_in_topological_order_.reserve(original.size());
  InlinedHashSet<NodeIndex> visited;
  for (auto& node : original) {
    if (visited.find(node) != visited.end()) {
      continue;
    }
    nodes_in_topological_order_.push_back(node);
    visited.insert(node);
    if (shape_size_parents.find(node) != shape_size_parents.end()) {
      for (auto& following_node : shape_size_parents[node]) {
        nodes_in_topological_order_.push_back(following_node);
        visited.insert(following_node);
      }
    }
  }
#endif
#if !defined(ORT_MINIMAL_BUILD)

  InlinedHashMap<const Node*, size_t> node_to_execution_order_map;
  for (size_t i = 0; i < nodes_in_topological_order_.size(); ++i) {
    node_to_execution_order_map.insert({graph_->GetNode(nodes_in_topological_order_[i]), i});
  }

  // graph.PriorityBasedReverseDFSFrom(
  //     leaf_nodes,
  //     [this](const Node* n) {
  //       std::cout << "Add Node: " << n->Name() << " Priority: " << n->Priority() << std::endl;
  //       nodes_in_topological_order_with_priority_.push_back(n->Index());
  //     },
  //     PriorityNodeCompare());

  // std::cout << "size of nodes_in_topological_order_with_priority_: " << nodes_in_topological_order_with_priority_.size() << std::endl;
  // std::reverse(nodes_in_topological_order_with_priority_.begin(), nodes_in_topological_order_with_priority_.end());
  InlinedHashMap<std::string, const Node*> fw_name_to_node_map;
  InlinedHashMap<std::string, std::string> bw_recompute_to_fw_node_map;

  InlinedHashMap<std::string, const Node*> name_to_node_map1;
  auto ends_with = [](std::string_view name1, std::string_view ending) {
    if (name1.size() < ending.size()) {
      return false;
    }

    return name1.compare(name1.size() - ending.size(), ending.size(), ending) == 0;
  };
  constexpr std::string_view recompute_suffix = "_recompute";

  for (const auto& n1 : graph.Nodes()) {
    name_to_node_map1.insert({n1.Name(), &n1});
  }

  for (const auto& n1 : graph.Nodes()) {
    // Check whether node name ends with "_recompute"
    if (ends_with(n1.Name(), recompute_suffix)) {
      auto name1_without_recompute = n1.Name().substr(0, n1.Name().size() - recompute_suffix.size());
      fw_name_to_node_map.insert({name1_without_recompute, name_to_node_map1.at(name1_without_recompute)});
      // bw_name_to_node_map.insert({n1.Name(), std::make_pair<const Node*, float>(name_to_node_map1.at(name1_without_recompute), 0.0f)});
      bw_recompute_to_fw_node_map.insert({n1.Name(), name1_without_recompute});
    }
  }

  InlinedHashMap<std::string, float> node_name_to_timestamp_map;
  graph.KahnsTopologicalSort(
      [this](const Node* n) {
        // m.insert({n->Name(), n});  // write
        nodes_in_topological_order_with_priority_.push_back(n->Index());
      },
      PriorityNodeCompare(fw_name_to_node_map, bw_recompute_to_fw_node_map, node_name_to_timestamp_map),
      node_name_to_timestamp_map);
#endif

  if (filter_info_) {
    // validate. if something is off here it's a bug in our code
    for (NodeIndex idx : filter_info->nodes) {
      ORT_ENFORCE(graph_->GetNode(idx) != nullptr, "IndexedSubGraph contains values not present in the Graph");
    }

    // create set of node indexes as we need quick lookups and don't care about the order
    filtered_node_indices_ = FilteredNodeSet(filter_info->nodes.cbegin(),
                                             filter_info->nodes.cend());

    const auto& metadef = filter_info->GetMetaDef();

    filtered_node_inputs_.reserve(metadef->inputs.size());
    filtered_node_inputs_including_initializers_.reserve(metadef->inputs.size());

    for (const auto& input : metadef->inputs) {
      const auto* nodearg = graph.GetNodeArg(input);
      ORT_ENFORCE(nodearg, "Mismatch between Graph and IndexedSubGraph. Input not found:", input);
      filtered_node_inputs_including_initializers_.push_back(nodearg);
      if (!graph.IsInitializedTensor(input)) {
        filtered_node_inputs_.push_back(nodearg);
      }
    }

    for (const auto& output : metadef->outputs) {
      const auto* nodearg = graph.GetNodeArg(output);
      ORT_ENFORCE(nodearg, "Mismatch between Graph and IndexedSubGraph. Output not found:", output);
      filtered_node_outputs_.push_back(nodearg);
    }

    // filter nodes in topo order to just the nodes in filter_info_
    auto orig_order = std::move(nodes_in_topological_order_);
    nodes_in_topological_order_.reserve(filter_info->nodes.size());
    std::copy_if(orig_order.cbegin(), orig_order.cend(), std::back_inserter(nodes_in_topological_order_),
                 [this](NodeIndex idx) { return filtered_node_indices_.count(idx) != 0; });

    // Filter the initializers also
    // Get the names of all the inputs and implicit inputs of all the nodes in this subgraph
    for (const auto node_idx : filtered_node_indices_) {
      const auto* node = GetNode(node_idx);
      ORT_ENFORCE(node, "Mismatch between Graph and IndexedSubGraph. Node not found: ", node_idx);
      const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
      for (const auto* node_input : node->InputDefs()) {
        if (graph.GetInitializedTensor(node_input->Name(), tensor)) {
          filtered_initializers_.insert({node_input->Name(), tensor});
        }
      }

      // The implicit inputs for subgraphs (if any)
      for (const auto* node_input : node->ImplicitInputDefs()) {
        if (graph.GetInitializedTensor(node_input->Name(), tensor)) {
          filtered_initializers_.insert({node_input->Name(), tensor});
        }
      }
    }

#if !defined(ORT_MINIMAL_BUILD)
    auto orig_priority_order = std::move(nodes_in_topological_order_with_priority_);
    nodes_in_topological_order_with_priority_.reserve(filter_info->nodes.size());
    std::copy_if(orig_priority_order.cbegin(), orig_priority_order.cend(),
                 std::back_inserter(nodes_in_topological_order_with_priority_),
                 [this](NodeIndex idx) { return filtered_node_indices_.count(idx) != 0; });
#endif
  }
}

// Graph name.
const std::string& GraphViewer::Name() const noexcept {
  return (filter_info_ == nullptr) ? graph_->Name()
                                   : filter_info_->GetMetaDef()->name;
}

const std::string& GraphViewer::Description() const noexcept {
  // filter_info_ doesn't have description so return 'name' instead of nothing
  // and to disambiguate between the full graph's description
  return (filter_info_ == nullptr) ? graph_->Description()
                                   : filter_info_->GetMetaDef()->name;
}

bool GraphViewer::GetInitializedTensor(const std::string& tensor_name,
                                       const ONNX_NAMESPACE::TensorProto*& value) const {
  value = nullptr;

  // if we are using filtered subgraph, the initializer has to be part of the subgraph
  if (filter_info_ != nullptr && filtered_initializers_.find(tensor_name) == filtered_initializers_.cend())
    return false;

  return graph_->GetInitializedTensor(tensor_name, value);
}

bool GraphViewer::CanOverrideInitializer() const noexcept {
  return graph_->CanOverrideInitializer();
}

// Graph inputs excluding initializers.
const std::vector<const NodeArg*>& GraphViewer::GetInputs() const noexcept {
  return (filter_info_ == nullptr) ? graph_->GetInputs()
                                   : filtered_node_inputs_;
}
// Graph inputs including initializers. Contains no nullptr values.
// This will match the number and order of inputs from the GraphProto.
const std::vector<const NodeArg*>& GraphViewer::GetInputsIncludingInitializers() const noexcept {
  return (filter_info_ == nullptr) ? graph_->GetInputsIncludingInitializers()
                                   : filtered_node_inputs_including_initializers_;
}

// Graph outputs. Should have no nullptr values.
const std::vector<const NodeArg*>& GraphViewer::GetOutputs() const noexcept {
  return (filter_info_ == nullptr) ? graph_->GetOutputs()
                                   : filtered_node_outputs_;
}

bool GraphViewer::NodeProducesGraphOutput(const Node& node) const {
  const auto& outputs = GetOutputs();
  auto end_outputs = outputs.cend();
  for (auto output_def : node.OutputDefs()) {
    if (std::find(outputs.cbegin(), end_outputs, output_def) != end_outputs) {
      return true;
    }
  }
  return false;
}

// Get graph value infos.
const std::unordered_set<const NodeArg*>& GraphViewer::GetValueInfo() const noexcept {
  return graph_->GetValueInfo();
}

// Get const Node given specific node index. May return nullptr if node as been freed.
const Node* GraphViewer::GetNode(NodeIndex node_index) const {
  if (filter_info_ && filtered_node_indices_.count(node_index) == 0) {
    return nullptr;
  }

  return graph_->GetNode(node_index);
}

const ConstGraphNodes& GraphViewer::Nodes() const noexcept {
  return graph_nodes_;
}

int GraphViewer::NumberOfNodes() const noexcept {
  return (filter_info_ == nullptr) ? graph_->NumberOfNodes()
                                   : gsl::narrow_cast<int>(filter_info_->nodes.size());
}

int GraphViewer::MaxNodeIndex() const noexcept {
  return graph_->MaxNodeIndex();
}

const std::vector<NodeIndex>& GraphViewer::GetNodesInTopologicalOrder(ExecutionOrder order) const {
  switch (order) {
    case ExecutionOrder::DEFAULT:
      return nodes_in_topological_order_;
#if !defined(ORT_MINIMAL_BUILD)
    case ExecutionOrder::PRIORITY_BASED:
      return nodes_in_topological_order_with_priority_;
#endif
    default:
      ORT_THROW("Invalid ExecutionOrder");
  }
}

const std::vector<NodeIndex>& GraphViewer::GetRootNodes() const {
  // TODO: See if we need to calculate the root_nodes_ of the filtered graph.
  // GetRootNodes is only used by parallel executor currently, and isn't relevant to the usage of a filtered graph.
  ORT_ENFORCE(filter_info_ == nullptr, "Not supported with filtered graph.");

  return root_nodes_;
}

const InitializedTensorSet& GraphViewer::GetAllInitializedTensors() const noexcept {
  return (filter_info_ == nullptr)
             ? graph_->GetAllInitializedTensors()
             : filtered_initializers_;
}

const NodeArg* GraphViewer::GetNodeArg(const std::string& name) const {
  return graph_->GetNodeArg(name);
}

bool GraphViewer::IsSubgraph() const {
  return graph_->IsSubgraph();
}

bool GraphViewer::IsConstantInitializer(const std::string& name, bool check_outer_scope) const {
  return GetConstantInitializer(name, check_outer_scope) != nullptr;
}

bool GraphViewer::IsInitializedTensor(const std::string& name) const {
  return graph_->IsInitializedTensor(name);
}

const ONNX_NAMESPACE::TensorProto* GraphViewer::GetConstantInitializer(const std::string& initializer_name,
                                                                       bool check_outer_scope) const {
  return graph_->GetConstantInitializer(initializer_name, check_outer_scope);
}

#if !defined(ORT_MINIMAL_BUILD)
const std::unordered_set<std::string>& GraphViewer::GetOuterScopeNodeArgNames() const noexcept {
  return graph_->GetOuterScopeNodeArgNames();
}
#endif

}  // namespace onnxruntime
