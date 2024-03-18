// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext} from '../types';

import {AttentionAttrs, AttentionMaskType, AttentionParameters, AttentionQkvFormat, computeAttentionProbs, computeVxAttentionScore} from './attention';
import {maybeTransposeToBNSHAndAddBias} from './multi-head-attentiion';
import {createTileProgramInfo} from './tile';
import {createTransposeProgramInfo, TransposeAttributes} from './transpose';

export const validateInputs = (inputs: readonly TensorView[], attributes: AttentionAttrs): AttentionParameters => {
  const query = inputs[0];
  const key = inputs[1];
  const value = inputs[2];
  const pastKey = inputs[3];
  const pastValue = inputs[4];

  // Abbreviation and Meanings:
  //   B:    batch_size
  //   S:    sequence_length (input sequence length of query)
  //   P:    past_sequence_length (past sequence length of key or value)
  //   L:    kv_sequence_length (input sequence length of key or value)
  //   M:    max_sequence_length
  //   T:    total_sequence_length = past_sequence_length + kv_sequence_length
  //   N:    num_heads
  //   H:    head size for Q and K, aka q_head_size or k_head_size or qk_head_size
  //   H_v:  v_head_size
  //   D_i:  input hidden size
  //   D:    hidden size for Q and K (D = N * H), aka q_hidden_size or k_hidden_size or qk_hidden_size
  //   D_v:  v_hidden_size = num_heads * v_head_size

  //     past_key                   : (B, N, S*, H)
  //     past_value                 : (B, N, S*, H)
  // When no packing for q/k/v:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, D) or (B, N, S*, H)
  //     value            (V)       : (B, L, D_v) or (B, N, S*, H)
  // When packed kv is used:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, N, 2, H)
  //     value            (V)       : None
  // When packed qkv is used:
  //     query            (Q)       : (B, L, N, 3, H) or (B, S, 3*D)
  //     key              (K)       : None
  //     value            (V)       : None

  if (query.dims.length !== 3 && query.dims.length !== 5) {
    throw new Error('Input query is expected to have 3 or 5 dimensions');
  }

  const dmmhaPacking = false;
  const batchSize = query.dims[0];
  const sequenceLength = query.dims[1];
  const hiddenSize = query.dims.length === 3 ? (dmmhaPacking ? query.dims[2] / 3 : query.dims[2]) :
                                               attributes.numHeads * query.dims[4];
  let kvSequenceLength = sequenceLength;

  let pastSequenceLength = 0;
  let maxSequenceLength = 0;
  const headSize = Math.floor(hiddenSize / attributes.numHeads);
  if (pastKey && pastValue) {
    if (pastKey.dims.length !== 4) {
      throw new Error('Input "past_key" is expected to have 4 dimensions');
    }
    if (pastValue.dims.length !== 4) {
      throw new Error('Input "past_value" is expected to have 4 dimensions');
    }
    pastSequenceLength = pastKey.dims[2];
    maxSequenceLength = pastKey.dims[2];
  } else if (pastKey || pastValue) {
    throw new Error('Input "past_key" and "past_value" shall be both present or both absent');
  }

  let qkvFormat: AttentionQkvFormat;
  if (key) {
    if (query.dims.length !== 3) {
      throw new Error('Input "query" is expected to have 3 dimensions when key is given');
    }
    if (key.dims.length < 3 || key.dims.length > 5) {
      throw new Error('Input "key" is expected to have 3, 4, or 5 dimensions');
    }
    if (query.dims[0] !== key.dims[0]) {
      throw new Error('Input "query" and "key" shall have same dim 0 (batch size)');
    }

    if (key.dims.length === 3) {
      if (query.dims[2] % key.dims[2] !== 0) {
        throw new Error('Dimension 2 of "query" should be a multiple of "key"');
      }
      qkvFormat = AttentionQkvFormat.qkvBSNH;
      kvSequenceLength = key.dims[1];
    } else if (key.dims.length === 5) {
      if (key.dims[2] !== attributes.numHeads || key.dims[3] !== 2 || key.dims[4] !== headSize) {
        throw new Error('Expect "key" shape (batch_size, kv_sequence_length, num_heads, 2, head_size) for packed kv');
      }
      if (value) {
        throw new Error('Expect "value" be none when "key" has packed kv format.');
      }
      qkvFormat = AttentionQkvFormat.qKvBSNHxBSN2H;
      kvSequenceLength = key.dims[1];
    } else {  // key_dims.size() == 4 (cross-attention with past_key)
      if (key.dims[1] !== attributes.numHeads || key.dims[3] !== headSize) {
        throw new Error('Expect "key" shape (batch_size, num_heads, kv_sequence_length, head_size) for past_key');
      }

      qkvFormat = AttentionQkvFormat.unknown;
      kvSequenceLength = key.dims[2];
    }
  } else {  // packed QKV
    if (query.dims.length !== 3 && query.dims.length !== 5) {
      throw new Error('Input "query" is expected to have 3 or 5 dimensions when key is empty');
    }
    if (query.dims.length === 5 && (query.dims[2] !== attributes.numHeads || query.dims[3] !== 3)) {
      throw new Error('Expect "query" shape (batch_size, kv_sequence_length, num_heads, 3, head_size) for packed kv');
    }

    qkvFormat = AttentionQkvFormat.qkvBSN3H;
  }

  const maskType: AttentionMaskType = AttentionMaskType.none;
  let passPastInKv = false;
  let vHiddenSize = hiddenSize;
  if (value) {
    if (value.dims.length !== 3 && value.dims.length !== 4) {
      throw new Error('Input "value" is expected to have 3 or 4 dimensions');
    }

    if (query.dims[0] !== value.dims[0]) {
      throw new Error('Input "query" and "value" shall have same dim 0 (batch_size)');
    }

    if (value.dims.length === 3) {
      if (kvSequenceLength !== value.dims[1]) {
        throw new Error('Input "key" and "value" shall have the same dim 1 (kv_sequence_length)');
      }
      vHiddenSize = value.dims[2];
    } else {
      if (kvSequenceLength !== value.dims[2]) {
        throw new Error('Input "past_key" and "past_value" shall have the same dim 2 (kv_sequence_length)');
      }
      vHiddenSize = value.dims[1] * value.dims[3];
      passPastInKv = true;
    }
  }

  const totalSequenceLength = pastSequenceLength + kvSequenceLength;
  const broadcastResPosBias = false;

  if (pastKey) {
    throw new Error('pastKey is not supported');
  }
  if (pastValue) {
    throw new Error('pastValue is not supported');
  }

  const kvNumHeads = attributes.kvNumHeads;
  return {
    batchSize,
    sequenceLength,
    pastSequenceLength,
    kvSequenceLength,
    totalSequenceLength,
    maxSequenceLength,
    inputHiddenSize: 0,
    hiddenSize,
    vHiddenSize,
    headSize,
    vHeadSize: Math.floor(vHiddenSize / kvNumHeads!),
    numHeads: attributes.numHeads,
    kvNumHeads,
    isUnidirectional: false,
    pastPresentShareBuffer: false,
    maskFilterValue: attributes.maskFilterValue,
    maskType,
    scale: attributes.scale,
    broadcastResPosBias,
    passPastInKv,
    qkvFormat,
  };
};

export const applyAttention =
    (context: ComputeContext, q: TensorView, k: TensorView, v: TensorView, _maskIndex: TensorView|undefined,
     _past: TensorView|undefined, _pastKey: TensorView|undefined, _pastValue: TensorView|undefined,
     relativePositionBias: TensorView|undefined, parameters: AttentionParameters, nReps: number|undefined) => {
      const probs = computeAttentionProbs(context, q, k, relativePositionBias, parameters, 1.0);

      computeVxAttentionScore(context, probs, v, parameters, nReps);
    };

export const parseGroupQueryAttentionAttributes = (attributes: AttentionAttrs): AttentionAttrs =>
    createAttributeWithCacheKey({...attributes});

const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: [0, 2, 1, 3]});

const maybeExpandAndTransposeToBNSH =
    (context: ComputeContext, batchSize: number, numHeads: number, sequenceLength: number, headSize: number,
     input: TensorView, nReps: number) => {
      let reshapedInput = input;
      if (input.dims.length === 3) {
        reshapedInput = input.reshape([batchSize, sequenceLength, numHeads, headSize]);
      }
      if (nReps !== 1) {
        reshapedInput = context.compute(
            createTileProgramInfo([reshapedInput], [1, 1, 1, nReps]), {inputs: [reshapedInput], outputs: [-1]})[0];

        reshapedInput = reshapedInput.reshape([batchSize, sequenceLength, numHeads * nReps, headSize]);
      }

      return context.compute(
          createTransposeProgramInfo(reshapedInput, weightTransposeAttribute.perm),
          {inputs: [reshapedInput], outputs: [-1]})[0];
    };

export const groupQueryAttention = (context: ComputeContext, attributes: AttentionAttrs): void => {
  const params = validateInputs(context.inputs, attributes);

  if (context.inputs[0].dims.length === 5) {
    throw new Error('Packed QKV is not implemented');
  }

  if (context.inputs[1]?.dims.length === 5) {
    throw new Error('Packed KV is not implemented');
  }

  const Q = maybeTransposeToBNSHAndAddBias(
      context, params.batchSize, params.numHeads, params.sequenceLength, params.headSize, context.inputs[0],
      context.inputs[3], 0);

  const nReps = Math.floor(attributes.numHeads / params.kvNumHeads!);
  const K = maybeExpandAndTransposeToBNSH(
      context, params.batchSize, params.kvNumHeads!, params.kvSequenceLength, params.vHeadSize, context.inputs[1],
      nReps);

  const V = maybeExpandAndTransposeToBNSH(
      context, params.batchSize, params.kvNumHeads!, params.kvSequenceLength, params.vHeadSize, context.inputs[2],
      nReps);
  applyAttention(context, Q, K, V, context.inputs[4], undefined, undefined, undefined, undefined, params, nReps);
};
