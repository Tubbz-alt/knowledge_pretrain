# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion

logger = logging.getLogger(__name__)


@register_criterion('multitask_lm')
class MultitaskLMCriterion(FairseqCriterion):

    def __init__(self, task, tpu, classification_head_name=None, tagging_head_name=None):
        super().__init__(task)
        self.tpu = tpu
        self.classification_head_name = classification_head_name
        self.tagging_head_name = tagging_head_name
        self.tag_weight = torch.tensor([0.1, 0.9]).cuda()

    def check_valid(self, masked_tokens):
        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        # https://github.com/pytorch/fairseq/blob/4c55744ec4cb26749cf2cf8dac89942f26ce4bd2/fairseq/criterions/masked_lm.py#L36
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )
        return masked_tokens

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = {}
        masked_tokens['mlm'] = sample['mlm_target'].ne(self.padding_idx)
        masked_tokens['clm'] = sample['clm_target'].ne(self.padding_idx)
        masked_tokens['cls'] = sample['cls_target'].ne(0)  # 0 as pad
        masked_tokens['tag'] = sample['tag_target'].ne(0)  # 0 as pad

        tags = model.get_tag_targets(sample, None) - 1

        sample_sizes = {k: v.int().sum().item() for k, v in masked_tokens.items()}
        for k, v in sample_sizes.items():
            if v > 0:
                continue
            masked_tokens[k] = None

        net_output = model(**sample['net_input'], masked_tokens=masked_tokens,
                           classification_head_name=self.classification_head_name,
                           tagging_head_name=self.tagging_head_name,
                           tags=tags)
        loss = 0
        ppl_loss = 0  # ppl_loss = mlm_loss + clm_loss
        ncorrect_cls = 0
        ncorrect_tag = 0
        if sample_sizes['mlm'] > 0:
            mlm_loss = self._compute_mlm_loss(model, net_output, sample, masked_tokens['mlm'])
            loss += mlm_loss
            ppl_loss += mlm_loss
        if sample_sizes['clm'] > 0:
            clm_loss = self._compute_clm_loss(model, net_output, sample, masked_tokens['clm'])
            loss += clm_loss
            ppl_loss += clm_loss
        if sample_sizes['cls'] > 0:
            cls_loss, ncorrect_cls = self._compute_cls_loss(model, net_output, sample, masked_tokens['cls'])
            loss += cls_loss
        if sample_sizes['tag'] > 0:
            tag_loss, ncorrect_tag = net_output[1]['tag_out']
            loss += tag_loss
        sample_size = sum(sample_sizes.values())
        logging_output = {
            'loss': loss.data,
            'ppl_loss': ppl_loss.data if isinstance(ppl_loss, torch.Tensor) else ppl_loss,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,  # avg_loss = loss/sample_size
            'sample_size_cls': sample_sizes['cls'],
            'sample_size_ppl': sample_sizes['clm'] + sample_sizes['mlm'],
            'sample_size_tag': sample_sizes['tag'],
            'ncorrect_cls': ncorrect_cls,
            'ncorrect_tag': ncorrect_tag,
        }
        return loss, sample_size, logging_output

    def _compute_mlm_loss(self, model, net_output, sample, masked_tokens):
        logits = net_output[1]['mlm_out']
        targets = model.get_mlm_targets(sample, net_output)[masked_tokens]
        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )
        return loss

    def _compute_clm_loss(self, model, net_output, sample, masked_tokens):
        logits = net_output[1]['clm_out']
        targets = model.get_clm_targets(sample, net_output)[masked_tokens]
        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )
        return loss

    def _compute_cls_loss(self, model, net_output, sample, masked_tokens):
        logits = net_output[1]['cls_out']
        targets = model.get_cls_targets(sample, net_output)[masked_tokens]
        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        loss = F.nll_loss(lprobs, targets - 1, reduction='sum')
        sample['cls_target'] = sample['cls_target'] - 1
        preds = logits.argmax(dim=1)
        ncorrect = (preds == targets - 1).sum()
        return loss, ncorrect

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ppl_loss_sum = sum(log.get('ppl_loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        sample_size_ppl = sum(log.get('sample_size_ppl', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)  # 为什么要 log2？

        if isinstance(ppl_loss_sum, torch.Tensor):
            metrics.log_scalar('ppl_loss', ppl_loss_sum / sample_size_ppl / math.log(2), sample_size_ppl, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['ppl_loss'].avg))

        if len(logging_outputs) > 0 and 'ncorrect_cls' in logging_outputs[0]:
            nsentences = sum(log.get('sample_size_cls', 0) for log in logging_outputs)
            ncorrect_cls = sum(log.get('ncorrect_cls', 0) for log in logging_outputs)
            if nsentences > 0:
                metrics.log_scalar('sent_cls_accuracy', 100.0 * ncorrect_cls / nsentences, nsentences, round=1)

        if len(logging_outputs) > 0 and 'ncorrect_tag' in logging_outputs[0]:
            ntags = sum(log.get('sample_size_tag', 0) for log in logging_outputs)
            ncorrect_tag = sum(log.get('ncorrect_tag', 0) for log in logging_outputs)
            if nsentences > 0:
                metrics.log_scalar('tag_accuracy', 100.0 * ncorrect_tag / ntags, ntags, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
