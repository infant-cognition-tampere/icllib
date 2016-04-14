#!/usr/bin/env python
"""Converts TREC2 dataset to HDF5 format."""

from icllib.dataset import CSVProtocolDataset
from icllib.dataset.protocol import TREC2
from os.path import join


if __name__ == "__main__":

    # Load dataset into memory
    basedir = '/Users/mika/ICL/FPA data'
    ds = CSVProtocolDataset(join(basedir, 'TREC2'), TREC2())

    def _read_gazedata(gzdname):
        print('==== Reading %s' % gzdname)
        return ds.get_gazedata(gzdname)

    dslist = ['Disengagement201-272-221-2.gazedata']
    gazedatas = map(_read_gazedata, dslist)
