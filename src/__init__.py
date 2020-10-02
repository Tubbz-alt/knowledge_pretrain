
# MASS
from .models import transformer_mass
from .tasks import mass_lm


# multi-task lm
from .models import transformer_multitask_pretrain
from .tasks import multitask_lm, sequence_tagging, sentence_prediction_bert
from .criterions import composite_criterion, sequence_tagging


# ROBETA


# MPNet
# from .models import mpnet
# from .tasks import masked_permutation_lm
# from .criterions import masked_permutation_criterion


# BART



"""
ProphetNet
"""
# from .models import transformer_ngram
# from .modules import ngram_multihead_attention
# from .criterions import ngram_criterions
