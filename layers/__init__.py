# -*- coding: utf-8 -*-

from .aggregator import SumAggregator, ConcatAggregator, NeighAggregator
from .multiAttention import MultiAttention
from .capsule import Caps

Aggregator = {
    'sum': SumAggregator,
    'concat': ConcatAggregator,
    'neigh': NeighAggregator
}
