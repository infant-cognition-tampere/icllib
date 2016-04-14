#!/usr/bin/env python
"""Converts Kinship dataset to HDF5 format."""

from icllib.dataset import CSVProtocolDataset
from icllib.dataset.protocol import Kinship

from tables import open_file, Filters

from multiprocess import Pool

from os.path import join


if __name__ == "__main__":

    # Create HDF5 file
    filters = Filters(complib='blosc', complevel=9)
    fileh = open_file('kinship.h5', mode='w', filters=filters)
    root = fileh.root
    gazedatagroup = fileh.create_group(root, 'gazedatas')

    # Load dataset into memory
    basedir = '/Users/mika/ICL/FPA data'
    ds = CSVProtocolDataset(join(basedir, 'Kinship'), Kinship())

    p = Pool(8)
    gazedatas = p.imap(ds.get_gazedata, ds.list_gazedatas())

    # Write TBT data into HDF5 file
    fileh.create_table('/',
                       'tbt',
                       expectedrows=ds.tbt.data.shape[0],
                       obj=ds.tbt.data)

    # Write tables from dataset into HDF5 file
    for gzd, gzdname in zip(gazedatas, ds.list_gazedatas()):
        table = fileh.create_table(gazedatagroup,
                                   gzdname,
                                   expectedrows=gzd.data.shape[0],
                                   obj=gzd.data)

    fileh.close()
