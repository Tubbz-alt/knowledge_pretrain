# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import random
import math
import logging
import itertools

from fairseq import utils
from fairseq.data import FairseqDataset, LanguagePairDataset
from .noise_util import apply_span_mask, apply_random_mask, apply_entity_mask
from fairseq.data import data_utils

logger = logging.getLogger(__name__)


def collate(
        samples,
        pad_idx,
        eos_idx,
        left_pad_source=False,
        left_pad_target=False,
        input_feeding=True,
        pad_to_length=None,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].ne(pad_idx).long().sum() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)

    id = torch.LongTensor([s['id'] for s in samples]).index_select(0, sort_order)
    src_tokens = merge('source', left_pad=left_pad_source).index_select(0, sort_order)

    # sentence classification
    cls_target = merge('cls_target', left_pad=left_pad_target).index_select(0, sort_order).view(-1)

    # masked language model
    mlm_target = merge('mlm_target', left_pad=left_pad_target).index_select(0, sort_order)

    # causal language model
    prev_output_tokens = merge('prev_output_tokens', left_pad=left_pad_target).index_select(0, sort_order)
    clm_positions = merge('clm_positions', left_pad=left_pad_target).index_select(0, sort_order)
    clm_target = merge('clm_target', left_pad=left_pad_target).index_select(0, sort_order)

    # sequence tagging
    tag_target = merge('tag_target', left_pad=left_pad_target).index_select(0, sort_order)

    ntokens = src_lengths.sum().item()

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'prev_output_tokens': prev_output_tokens,
            'clm_positions': clm_positions,
        },
        'cls_target': cls_target,
        'mlm_target': mlm_target,
        'clm_target': clm_target,
        'tag_target': tag_target,
    }
    return batch


class KnowledgeLanguagePairDataset(LanguagePairDataset):

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        return cls(dataset, *args, **kwargs)

    def __init__(
            self, src, src_sizes, src_dict,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            meta=None, meta_sizes=None, meta_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True,
            mask_prob=0.15, leave_unmasked_prob=0.1, random_token_prob=0.1,
            mask_whole_words=None,
            block_size=64,
            sub_task=None,
    ):
        super().__init__(src, src_sizes, src_dict,
                         tgt=tgt, tgt_sizes=tgt_sizes, tgt_dict=tgt_dict,
                         left_pad_source=left_pad_source, left_pad_target=left_pad_target,
                         shuffle=shuffle)
        self.meta = meta
        self.meta_sizes = meta_sizes
        self.meta_dict = meta_dict
        self.mask_prob = mask_prob
        assert len(meta_sizes) == len(src_sizes)

        self.sub_task = sub_task
        self.cls_pad = self.src_dict.pad()
        self.block_size = block_size
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.pred_probs = torch.FloatTensor(
            [1 - leave_unmasked_prob - random_token_prob, leave_unmasked_prob, random_token_prob])
        self.debug_size_for_mlm = 0
        self.debug_size_for_tag = 0
        self.debug_size_for_titlegen = 0

    def _parse_ocr_data(self, src_item):
        """
        Args:
            src_item:
        """

        def _get_title_and_content(sep_idx):
            title_pos = []
            content_pos = []
            for i, pos in enumerate(sep_idx):
                last_pos = sep_idx[i - 1] if i > 0 else 1
                pos_range = np.arange(last_pos + 1,
                                      pos) if pos > last_pos + 1 else None
                if i % 2 == 0:
                    title_pos.append(pos_range)
                else:
                    content_pos.append(pos_range)
            if len(content_pos) < len(title_pos):
                content_pos.append(None)
            return title_pos, content_pos

        src_item_np = np.array(src_item)
        sep_idx = np.where(src_item_np == self.src_dict.eos())[0]
        title_positions, content_positions = _get_title_and_content(sep_idx)
        source = src_item[:1]
        clm_target = np.array([], dtype=src_item_np.dtype)
        clm_positions_list = []
        sep_positions_list = []
        for title_position, content_position in zip(title_positions, content_positions):
            if title_position is not None:
                old_len = len(source)
                source = np.append(source, src_item[title_position])
                clm_target = np.append(clm_target, src_item[title_position])
                clm_positions_list = clm_positions_list + list(range(old_len, len(source)))
            if content_position is not None:
                source = np.append(source, src_item[content_position])
            sep_positions_list.append(len(source) - 1)
        sep_positions_list = [v for v in sep_positions_list if v != 0 and v != len(source) - 1]
        source = torch.LongTensor(np.append(source, self.src_dict.eos()))
        clm_target = torch.LongTensor(clm_target)
        return source, clm_target, clm_positions_list, sep_positions_list

    def _get_example_for_boundary_detection(self, index, src_item):
        """ TokenClassification
        Task: sequence tagging
        """
        source, _, _, sep_positions_list = self._parse_ocr_data(src_item)
        tag_target = torch.from_numpy(np.full(len(source), 1))  # 0: pad  1: negative 2: positive
        tag_target[0] = self.cls_pad
        tag_target[-1] = self.cls_pad
        tag_target[sep_positions_list] = 2

        if self.debug_size_for_tag < 2:
            self.debug_size_for_tag += 1
            logger.info('========= index: {} == boundary detection ======='.format(str(index)))
            logger.info('src_raw: ' + ''.join([self.src_dict[ii] for ii in src_item]))
            logger.info('src: ' + ''.join([self.src_dict[ii] for ii in source]))
            logger.info('tag_target: ' + ''.join([str(ii.item()) for ii in tag_target]))

        example = {
            'id': index,
            'source': source,
            'cls_target': torch.LongTensor([self.cls_pad]),
            'mlm_target': torch.from_numpy(np.full(len(source), self.src_dict.pad())),
            'clm_target': torch.from_numpy(np.full(1, self.src_dict.pad())),
            'tag_target': tag_target,
            'prev_output_tokens': torch.from_numpy(np.full(1, 1)),
            'clm_positions': torch.LongTensor([1]),
        }
        return example

    def _create_dummy_clm(self):
        clm_positions = torch.LongTensor([1])
        prev_output_tokens = torch.from_numpy(np.full(1, 1))
        clm_target = torch.from_numpy(np.full(1, self.src_dict.pad()))
        return clm_positions, prev_output_tokens, clm_target

    def _get_example_for_title_generation(self, index, src_item):
        """ title generation
        Task: CLM + MLM
        """
        source, clm_target, clm_positions_list, _ = self._parse_ocr_data(src_item)

        # build data for MLM (random mask)
        mlm_positions = apply_random_mask(len(source), ignore_index=set(clm_positions_list))
        masked_pos = sorted(list(set(clm_positions_list + mlm_positions)))
        mlm_target = torch.from_numpy(np.full(len(source), self.src_dict.pad()))
        mlm_target[mlm_positions] = source[mlm_positions]

        # build data for CLM (mask all title)
        clm_positions = np.array(clm_positions_list)
        prev_output_tokens = source[clm_positions - 1].clone()
        clm_positions = torch.LongTensor(clm_positions) + self.src_dict.pad_index

        if self.debug_size_for_titlegen < 2:
            logger.info('========= index: {} == title generation ======='.format(str(index)))
            logger.info('src_raw: ' + ''.join([self.src_dict[ii] for ii in src_item]))
            logger.info('src: ' + ''.join([self.src_dict[ii] for ii in source]))

        source[masked_pos] = self.replace(source[masked_pos])

        if self.debug_size_for_titlegen < 2:
            self.debug_size_for_titlegen += 1
            logger.info('src_mask: ' + ''.join([self.src_dict[ii] for ii in source]))
            logger.info('clm_pos: ' + ' '.join([str(v) for v in clm_positions_list]))
            logger.info('clm_input: ' + ''.join([self.src_dict[ii] for ii in prev_output_tokens]))
            logger.info('clm_target: ' + ''.join([self.src_dict[ii] for ii in clm_target]))
            logger.info(
                'mlm_target:' + ''.join([self.src_dict[ii] for ii in mlm_target if ii != self.src_dict.pad_index]))

        if prev_output_tokens.numel() == 0:
            clm_positions, prev_output_tokens, clm_target = self._create_dummy_clm()

        example = {
            'id': index,
            'source': source,
            'cls_target': torch.LongTensor([self.cls_pad]),
            'mlm_target': mlm_target,
            'clm_target': clm_target,
            'tag_target': torch.from_numpy(np.full(len(source), self.cls_pad)),
            'prev_output_tokens': prev_output_tokens,
            'clm_positions': clm_positions,
        }
        return example

    def _get_example_for_cls(self, index, src_item, src_meta):
        assert 'cls' in self.sub_task
        src_meta = np.array([int(self.meta_dict[k]) if k != self.meta_dict.unk() else 10000 for k in src_meta])
        src_sz = len(src_item)
        assert len(src_meta) % 2 == 1
        src_label, src_entity = torch.LongTensor(src_meta[:1]), src_meta[1:]
        src_entity = np.array(src_entity.reshape(-1, 2)) + 1  # offset for [CLS]

        mlm_position_list, clm_position_list = [], []
        clm_positions = np.array(clm_position_list)
        masked_pos_list = sorted(list(set(clm_position_list + mlm_position_list)))
        assert len(clm_position_list) + len(mlm_position_list) == len(masked_pos_list)
        assert masked_pos_list[0] > 0
        masked_pos = np.array(masked_pos_list)

        # build data for CLM in Decoder
        prev_output_tokens = src_item[clm_positions - 1].clone()
        clm_target = src_item[clm_positions].clone()
        clm_positions = torch.LongTensor(clm_positions) + self.src_dict.pad_index

        # build data for MLM in Encoder
        mlm_target = torch.from_numpy(np.full(src_sz, self.src_dict.pad()))
        mlm_target[mlm_position_list] = src_item[mlm_position_list]

        if self.debug_size_for_mlm < 2:
            logger.info('========= index: {} ==== MLM and CLM mask ====='.format(str(index)))
            logger.info('src: ' + ''.join([self.src_dict[ii] for ii in src_item]))
            logger.info('src_entity: ' + ' '.join(
                [''.join([self.src_dict[src_item[ii]] if ii < src_sz else '' for ii in range(ent[0], ent[1])]) for ent
                 in src_entity]))

        src_item[masked_pos] = self.replace(src_item[masked_pos])

        if self.debug_size_for_mlm < 2:
            self.debug_size_for_mlm += 1
            logger.info('src_mask: ' + ''.join([self.src_dict[ii] for ii in src_item]))
            logger.info('clm_pos: ' + ' '.join([str(v) for v in clm_position_list]))
            logger.info('clm_input: ' + ''.join([self.src_dict[ii] for ii in prev_output_tokens]))
            logger.info('clm_target: ' + ''.join([self.src_dict[ii] for ii in clm_target]))
            logger.info('mlm_pos: ' + ' '.join([str(v) for v in mlm_position_list]))
            logger.info(
                'mlm_target:' + ''.join([self.src_dict[ii] for ii in mlm_target if ii != self.src_dict.pad_index]))

        if prev_output_tokens.numel() == 0:
            clm_positions, prev_output_tokens, clm_target = self._create_dummy_clm()

        example = {
            'id': index,
            'source': src_item,
            'cls_target': src_label,
            'mlm_target': mlm_target,
            'clm_target': clm_target,
            'tag_target': torch.from_numpy(np.full(len(src_item), self.cls_pad)),
            'prev_output_tokens': prev_output_tokens,
            'clm_positions': clm_positions,
        }
        return example

    def _get_example_for_mlm(self, index, src_item, src_meta):
        assert 'mlm' in self.sub_task
        src_meta = np.array([int(self.meta_dict[k]) if k != self.meta_dict.unk() else 10000 for k in src_meta])
        src_sz = len(src_item)
        assert len(src_meta) % 2 == 1
        src_label, src_entity = torch.LongTensor(src_meta[:1]), src_meta[1:]
        src_entity = np.array(src_entity.reshape(-1, 2)) + 1  # offset for [CLS]
        mlm_position_list, clm_position_list = [], []
        clm_positions = np.array(clm_position_list)

        mlm_positions_1 = apply_entity_mask(src_sz, src_entity,
                                            ignore_index=set(clm_position_list))  # BERT & entity
        mlm_positions_2 = apply_random_mask(src_sz, ignore_index=set(clm_position_list + mlm_positions_1))  # BERT
        mlm_position_list = sorted(list(set(mlm_positions_1 + mlm_positions_2)))
        assert len(mlm_positions_1) + len(mlm_positions_2) == len(mlm_position_list)

        masked_pos_list = sorted(list(set(clm_position_list + mlm_position_list)))
        assert len(clm_position_list) + len(mlm_position_list) == len(masked_pos_list)
        assert masked_pos_list[0] > 0  # no mask in bos
        masked_pos = np.array(masked_pos_list)

        # build data for CLM in Decoder
        prev_output_tokens = src_item[clm_positions - 1].clone()
        clm_target = src_item[clm_positions].clone()
        clm_positions = torch.LongTensor(clm_positions) + self.src_dict.pad_index

        # build data for MLM in Encoder
        mlm_target = torch.from_numpy(np.full(src_sz, self.src_dict.pad()))
        mlm_target[mlm_position_list] = src_item[mlm_position_list]

        if self.debug_size_for_mlm < 2:
            logger.info('========= index: {} ==== MLM and CLM mask ====='.format(str(index)))
            logger.info('src: ' + ''.join([self.src_dict[ii] for ii in src_item]))
            logger.info('src_entity: ' + ' '.join(
                [''.join([self.src_dict[src_item[ii]] if ii < src_sz else '' for ii in range(ent[0], ent[1])]) for ent
                 in src_entity]))

        src_item[masked_pos] = self.replace(src_item[masked_pos])

        if self.debug_size_for_mlm < 2:
            self.debug_size_for_mlm += 1
            logger.info('src_mask: ' + ''.join([self.src_dict[ii] for ii in src_item]))
            logger.info('clm_pos: ' + ' '.join([str(v) for v in clm_position_list]))
            logger.info('clm_input: ' + ''.join([self.src_dict[ii] for ii in prev_output_tokens]))
            logger.info('clm_target: ' + ''.join([self.src_dict[ii] for ii in clm_target]))
            logger.info('mlm_pos: ' + ' '.join([str(v) for v in mlm_position_list]))
            logger.info(
                'mlm_target:' + ''.join([self.src_dict[ii] for ii in mlm_target if ii != self.src_dict.pad_index]))

        if prev_output_tokens.numel() == 0:
            clm_positions, prev_output_tokens, clm_target = self._create_dummy_clm()

        example = {
            'id': index,
            'source': src_item,
            'cls_target': src_label,
            'mlm_target': mlm_target,
            'clm_target': clm_target,
            'tag_target': torch.from_numpy(np.full(len(src_item), self.cls_pad)),
            'prev_output_tokens': prev_output_tokens,
            'clm_positions': clm_positions,
        }
        return example

    def _get_example_for_clm(self, index, src_item, src_meta):
        assert 'clm' in self.sub_task
        src_meta = np.array([int(self.meta_dict[k]) if k != self.meta_dict.unk() else 10000 for k in src_meta])
        src_sz = len(src_item)
        assert len(src_meta) % 2 == 1
        src_label, src_entity = torch.LongTensor(src_meta[:1]), src_meta[1:]
        src_entity = np.array(src_entity.reshape(-1, 2)) + 1
        src_label = torch.LongTensor([self.cls_pad])

        mlm_position_list, clm_position_list = [], []
        clm_position_list = apply_span_mask(src_sz)  # GPT、MASS
        clm_positions = np.array(clm_position_list)
        masked_pos_list = sorted(list(set(clm_position_list + mlm_position_list)))
        assert len(clm_position_list) + len(mlm_position_list) == len(masked_pos_list)
        assert masked_pos_list[0] > 0
        masked_pos = np.array(masked_pos_list)

        # build data for CLM in Decoder
        prev_output_tokens = src_item[clm_positions - 1].clone()
        clm_target = src_item[clm_positions].clone()
        clm_positions = torch.LongTensor(clm_positions) + self.src_dict.pad_index

        # build data for MLM in Encoder
        mlm_target = torch.from_numpy(np.full(src_sz, self.src_dict.pad()))
        mlm_target[mlm_position_list] = src_item[mlm_position_list]

        if self.debug_size_for_mlm < 2:
            logger.info('========= index: {} ==== MLM and CLM mask ====='.format(str(index)))
            logger.info('src: ' + ''.join([self.src_dict[ii] for ii in src_item]))
            logger.info('src_entity: ' + ' '.join(
                [''.join([self.src_dict[src_item[ii]] if ii < src_sz else '' for ii in range(ent[0], ent[1])]) for ent
                 in src_entity]))

        src_item[masked_pos] = self.replace(src_item[masked_pos])

        if self.debug_size_for_mlm < 2:
            self.debug_size_for_mlm += 1
            logger.info('src_mask: ' + ''.join([self.src_dict[ii] for ii in src_item]))
            logger.info('clm_pos: ' + ' '.join([str(v) for v in clm_position_list]))
            logger.info('clm_input: ' + ''.join([self.src_dict[ii] for ii in prev_output_tokens]))
            logger.info('clm_target: ' + ''.join([self.src_dict[ii] for ii in clm_target]))
            logger.info('mlm_pos: ' + ' '.join([str(v) for v in mlm_position_list]))
            logger.info(
                'mlm_target:' + ''.join([self.src_dict[ii] for ii in mlm_target if ii != self.src_dict.pad_index]))

        if prev_output_tokens.numel() == 0:
            clm_positions, prev_output_tokens, clm_target = self._create_dummy_clm()

        example = {
            'id': index,
            'source': src_item,
            'cls_target': src_label,
            'mlm_target': mlm_target,
            'clm_target': clm_target,
            'tag_target': torch.from_numpy(np.full(len(src_item), self.cls_pad)),
            'prev_output_tokens': prev_output_tokens,
            'clm_positions': clm_positions,
        }
        return example

    def _get_example_for_multitask(self, index, src_item, src_meta):
        """ multi-task joint training, including mlm, clm, cls
        https://github.com/pytorch/fairseq/blob/e75cff5f2c1d62f12dc911e0bf420025eb1a4e33/fairseq/data/legacy/masked_lm_dataset.py#L264
        """
        assert 'clm' in self.sub_task or 'mlm' in self.sub_task
        src_meta = np.array([int(self.meta_dict[k]) if k != self.meta_dict.unk() else 10000 for k in src_meta])
        src_sz = len(src_item)
        assert len(src_meta) % 2 == 1
        src_label, src_entity = torch.LongTensor(src_meta[:1]), src_meta[1:]
        src_entity = np.array(src_entity.reshape(-1, 2)) + 1  # offset for [CLS]
        if 'sentcls' not in self.sub_task:
            src_label = torch.LongTensor([self.cls_pad])

        mlm_position_list, clm_position_list = [], []
        if 'clm' in self.sub_task:
            clm_position_list = apply_span_mask(src_sz)
        clm_positions = np.array(clm_position_list)

        if 'mlm' in self.sub_task:
            mlm_positions_1 = apply_entity_mask(src_sz, src_entity,
                                                ignore_index=set(clm_position_list))  # BERT & entity
            mlm_positions_2 = apply_random_mask(src_sz, ignore_index=set(clm_position_list + mlm_positions_1))  # BERT
            mlm_position_list = sorted(list(set(mlm_positions_1 + mlm_positions_2)))
            assert len(mlm_positions_1) + len(mlm_positions_2) == len(mlm_position_list)

        masked_pos_list = sorted(list(set(clm_position_list + mlm_position_list)))
        assert len(clm_position_list) + len(mlm_position_list) == len(masked_pos_list)
        assert masked_pos_list[0] > 0
        masked_pos = np.array(masked_pos_list)

        # build data for CLM in Decoder
        prev_output_tokens = src_item[clm_positions - 1].clone()
        clm_target = src_item[clm_positions].clone()
        clm_positions = torch.LongTensor(clm_positions) + self.src_dict.pad_index

        # build data for MLM in Encoder
        mlm_target = torch.from_numpy(np.full(src_sz, self.src_dict.pad()))
        mlm_target[mlm_position_list] = src_item[mlm_position_list]

        if self.debug_size_for_mlm < 2:
            logger.info('========= index: {} ==== MLM and CLM mask ====='.format(str(index)))
            logger.info('src: ' + ''.join([self.src_dict[ii] for ii in src_item]))
            logger.info('src_entity: ' + ' '.join(
                [''.join([self.src_dict[src_item[ii]] if ii < src_sz else '' for ii in range(ent[0], ent[1])]) for ent
                 in src_entity]))

        src_item[masked_pos] = self.replace(src_item[masked_pos])

        if self.debug_size_for_mlm < 2:
            self.debug_size_for_mlm += 1
            logger.info('src_mask: ' + ''.join([self.src_dict[ii] for ii in src_item]))
            logger.info('clm_pos: ' + ' '.join([str(v) for v in clm_position_list]))
            logger.info('clm_input: ' + ''.join([self.src_dict[ii] for ii in prev_output_tokens]))
            logger.info('clm_target: ' + ''.join([self.src_dict[ii] for ii in clm_target]))
            logger.info('mlm_pos: ' + ' '.join([str(v) for v in mlm_position_list]))
            logger.info(
                'mlm_target:' + ''.join([self.src_dict[ii] for ii in mlm_target if ii != self.src_dict.pad_index]))

        if prev_output_tokens.numel() == 0:
            clm_positions, prev_output_tokens, clm_target = self._create_dummy_clm()

        example = {
            'id': index,
            'source': src_item,
            'cls_target': src_label,
            'mlm_target': mlm_target,
            'clm_target': clm_target,
            'tag_target': torch.from_numpy(np.full(len(src_item), self.cls_pad)),
            'prev_output_tokens': prev_output_tokens,
            'clm_positions': clm_positions,
        }
        return example

    def __getitem__(self, index):
        """
        TODO: dynamic_span_length, dynamic_total_length
        """
        src_item = self.src[index]
        src_meta = self.meta[index]
        sep_sz = (src_item == self.src_dict.eos()).sum()
        if sep_sz > 1:  # ocr data tasks: titlegen segcls, sentcls
            if 'titlegen' in self.sub_task and 'segcls' in self.sub_task:
                task_selector = random.random()
                if task_selector > 0.5:
                    example = self._get_example_for_title_generation(index, src_item)
                else:
                    example = self._get_example_for_boundary_detection(index, src_item)
            elif 'segcls' in self.sub_task:
                example = self._get_example_for_boundary_detection(index, src_item)
            elif 'titlegen' in self.sub_task:
                example = self._get_example_for_boundary_detection(index, src_item)
            else:
                return
            return example
        else:  # product summary data tasks:
            task_selector = random.random()
            if task_selector < 0.4:
                return self._get_example_for_mlm()
            elif task_selector < 0.7:
                return self._get_example_for_clm()
            else:
                return self._get_example_for_cls()

            return self._get_example_for_multitask(index, src_item, src_meta)

    def collater(self, samples):
        return collate(samples, self.src_dict.pad(), self.src_dict.eos())

    def replace(self, x):
        _x_real = x
        _x_rand = _x_real.clone().random_(self.src_dict.nspecial, len(self.src_dict))
        _x_mask = _x_real.clone().fill_(self.src_dict.index('[MASK]'))
        probs = torch.multinomial(self.pred_probs, len(x), replacement=True)
        _x = _x_mask * (probs == 0).long() + \
             _x_real * (probs == 1).long() + \
             _x_rand * (probs == 2).long()
        return _x
