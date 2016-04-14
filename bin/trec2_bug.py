#!/usr/bin/env python
"""Script to replicate bug reading TREC2 dataset."""

from icllib.dataset import CSVProtocolDataset
from icllib.dataset.protocol import TREC2
from os.path import join


if __name__ == "__main__":

    # Load dataset into memory
    basedir = '/mnt/bigstuffy1/ICLData'
    ds = CSVProtocolDataset(join(basedir, 'TREC2_7mo_Gaze'), TREC2())

    def _read_gazedata(gzdname):
        print('==== Reading %s' % gzdname)
        return ds.get_gazedata(gzdname)

    dslist = ['Disengagement201-272-221-2.gazedata']

    # Fix was to remove last line of gazedata
    gazedatas = map(_read_gazedata, dslist)
