#!/usr/bin/env python
"""Dataset test environment."""

from icllib.dataset import CSVProtocolDataset
from icllib.dataset.protocol import Kinship

d = '../tests/test_dataset'
tbt = 'disengagement_results.csv'
dataset = CSVProtocolDataset(d, Kinship())
gzd = dataset.get_gazedata('1.gazedata')
data = gzd.data
