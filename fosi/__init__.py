from .extreme_spectrum_estimation import get_ese_fn
from .fosi_optimizer import fosi, fosi_adam, fosi_momentum, fosi_sgd

import logging
import sys
logging.basicConfig(stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.NOTSET)

__all__ = ['get_ese_fn', 'fosi', 'fosi_adam', 'fosi_momentum', 'fosi_sgd']
