// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"
#include "onnx/defs/data_type_utils.h"

#include "QnnOpDef.h"  // From QNN SDK: contains QNN constants (e.g., op names, param values).

namespace onnxruntime {
namespace qnn {

Status ConvertOpBuilder::AddConvertToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& dq_node_unit,
                                                  const NodeUnit& q_node_unit,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  std::vector<std::string> input_names;

  // Process the input from the DQ node
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, dq_node_unit.Inputs()[0], logger, input_names));

  // Process the output from the Q node. Override the QNN operator type to "Convert".
  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, q_node_unit, std::move(input_names), {},
                                     logger, do_op_validation, QNN_OP_CONVERT));
  return Status::OK();
}

void CreateConvertOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ConvertOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
