# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np


def apply_pos_noise():
    pass


def apply_mask_dropout(encoder_padding_mask):
    """ A B C D E --> B D E
    drop encoder_out randomly
    refactor: pooler? dropout? random_mask? corrupt? noise?
    Reference：
      - https://github.com/pytorch/fairseq/blob/master/tests/test_inference_dropout.py
    """
    pooled_mask = None
    return pooled_mask


def apply_teacher_forcing_dropout():
    """ mask
    randomly zeroes some of the elements of the input tensor with probability
    """
    pass


def apply_entity_mask(src_len, src_entity, mask_prob=0.1, ignore_index=None, max_len=512):
    """
    :param src_len:
    :param src_entity:
    :param mask_prob:
    :param excludes:
    :param max_len: src-entity
    :return:
    """
    if len(src_entity) == 0:
        return []

    # 1. excludes
    valid_idx = []
    for idx in range(len(src_entity)):
        ent_id = set(range(src_entity[idx, 0], src_entity[idx, 1]))
        if ent_id & ignore_index:
            continue
        if src_entity[idx, 1] > max_len - 2:
            continue
        valid_idx.append(idx)
    valid_entity = src_entity[valid_idx, :]
    valid_cnt = len(valid_entity)
    if valid_cnt == 0:
        return []

    # 2. apply entity mask
    entity_size = valid_entity[:, 1] - valid_entity[:, 0]
    mask_token_cnt = (src_len - len(ignore_index)) * mask_prob
    mask_entity_cnt = math.ceil(mask_token_cnt // np.mean(entity_size))
    np.random.shuffle(valid_entity)
    mask_idx = valid_entity[:min(mask_entity_cnt, valid_cnt), :]
    mask_pos = []
    for pos in mask_idx:
        mask_pos += list(range(pos[0], pos[1]))
    mask_pos = sorted(list(set(mask_pos)))
    return mask_pos


def apply_random_mask(src_len, mask_prob=0.1, ignore_index=None):
    """ TODO: complete entity mask with random mask
    """
    candidates = [idx for idx in range(1, src_len - 1) if idx not in ignore_index]  # ignore bos and eos
    mask_token_cnt = math.ceil((src_len - len(ignore_index)) * mask_prob)
    random.shuffle(candidates)
    mask_pos = sorted(candidates[:mask_token_cnt])
    return mask_pos


def apply_span_mask(src_len, block_size=64, mask_prob=0.3):
    """ mask contiguous spans rather than random tokens
    30% clm_mask + 20% mlm_mask。
    - SpanBERT: mask contiguous span.
    - MASS:
    - BART: possioin
    mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
    """
    positions = np.arange(0, src_len)
    masked_pos = []
    for i in range(1, src_len - 1, block_size):
        block = positions[
                i: i + block_size]
        masked_len = int(len(block) * mask_prob)
        masked_len = masked_len if masked_len >= 0 else 1
        masked_block_start = np.random.choice(block[:len(block) - masked_len + 1], 1)[0]
        masked_pos.extend(positions[masked_block_start: masked_block_start + masked_len])
    return masked_pos


def apply_punc_deletion():
    """
    punc reconstruction: BART text_infilling
    motivation: for some noisy corpus with error punc.
    source: remove all punc
    """
    pass


def apply_gap_sentence_mask():
    """ ProphetNet """
    mask_pos = []
    return mask_pos


def apply_bart_mask():
    """ BART """
    pass


def apply_sentent_permutation():
    pass


def apply_entity_permutation():
    pass


def apply_ocr_segment_mask(src_item, mask_prob=0.1, ignore_index=None, max_len=512):
    pass
