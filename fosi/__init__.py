from .jax_optim import get_ese_fn, fosi, fosi_adam, fosi_momentum, fosi_sgd
from .torch_optim import get_ese_fn_torch, fosi_torch, fosi_adam_torch, fosi_momentum_torch, fosi_sgd_torch

import logging
import sys
logging.basicConfig(stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.NOTSET)

__all__ = ['get_ese_fn', 'fosi', 'fosi_adam', 'fosi_momentum', 'fosi_sgd',
           'get_ese_fn_torch', 'fosi_torch', 'fosi_adam_torch', 'fosi_momentum_torch', 'fosi_sgd_torch']
