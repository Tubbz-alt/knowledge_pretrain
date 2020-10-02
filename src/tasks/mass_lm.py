# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import logging
import torch

from collections import OrderedDict
from fairseq import utils
from fairseq.data import (
    data_utils,
    PrependTokenDataset,
    LanguagePairDataset,
    TokenBlockDataset,
    AppendTokenDataset,
    data_utils,
    Dictionary,
    IdDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TransformEosDataset,
    TruncatedDictionary,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationTask

from ..data.mask_language_pair_dataset import MaskedLanguagePairDataset
from ..data.bert_dictionary import BertDictionary

logger = logging.getLogger(__name__)


@register_task('masked_s2s')
class MaskedS2STask(FairseqTask):
    """
    Train a sequence-to-sequence task

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--sample-break-mode', default='none',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')
        parser.add_argument('--max-source-positions', default=512, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=512, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', default=False, action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-list', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')
        parser.add_argument('--shuffle', action='store_true',
                            help='shuffle each dataset while training')
        # parser.add_argument('--no-shuffle', action='store_true', default=False)

        # fmt: on

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        # add mask token

    @classmethod
    def setup_task(cls, args, **kwargs):
        """
        https://github.com/pytorch/fairseq/blob/master/fairseq/tasks/masked_lm.py#L78
        """
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = cls.load_dictionary(os.path.join(paths[0], 'dict.txt'))
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )

        # create continuous blocks of tokens.  block_size=511或者512
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.args.sample_break_mode,
        )
        logger.info('loaded {} blocks from: {}'.format(len(dataset), split_path))
        s2s_dataset = MaskedLanguagePairDataset.apply_mask(
            dataset,
            dataset.sizes,
            self.source_dictionary,
            shuffle=True,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
        )
        self.datasets[split] = s2s_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def max_positions(self):
        max_positions = 1024
        if hasattr(self.args, 'max_positions'):
            max_positions = min(max_positions, self.args.max_positions)
        if hasattr(self.args, 'max_source_positions'):
            max_positions = min(max_positions, self.args.max_source_positions)
        if hasattr(self.args, 'max_target_positions'):
            max_positions = min(max_positions, self.args.max_target_positions)
        return (max_positions, max_positions)

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                self.source_dictionary.bos()
                if getattr(self.args, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )
        tgt_dataset = AppendTokenDataset(
            dataset,
            token=self.source_dictionary.pad()
        )
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(src_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False),
            },
            sizes=[np.array(src_lengths)],
        )

    # def inference_step(self, generator, models, sample, prefix_tokens=None):
    #     raise NotImplementedError

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            if constraints is not None:
                raise NotImplementedError("Constrained decoding with the language_modeling task is not supported")

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(bos_token).all():
                    prefix_tokens = prefix_tokens[:, 1:]

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token,
            )


@register_task('translation_mass')
class TranslationMASSTask(TranslationTask):

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)
