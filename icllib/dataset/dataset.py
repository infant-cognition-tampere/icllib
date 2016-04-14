"""Convenience access methods for reading different datasets."""
import numpy as np
from os.path import join
from csv import DictReader

try:
    from functools32 import lru_cache
except ImportError:
    from functools import lru_cache


class DatasetError(Exception):
    """Exception class for Dataset Errors."""

    pass


class Dataset(object):
    """Convenience class for accessing the whole dataset."""

    def __init__(self):
        """Consructor."""
        pass

    def get_gazedata(self, name):
        """Get Gazedata from Dataset."""
        raise DatasetError('Not Implemented')

    def list_gazedatas(self):
        """List available Gazedatas."""
        raise DatasetError('Not Implemented')


class CSVDataset(Dataset):
    """CSVDataset implementation of Dataset interface."""

    def __init__(self, directory, filename, replaceRotatingTrialIDs=False):
        """Constructor.

        Input directory - Directory of dataset
              filename - Name of TBT file in directory
              replaceRotatingTrialIDs - Set True if Trial IDs rotate
        """
        self.directory = directory
        self.tbt = TBTFile(join(directory, filename), replaceRotatingTrialIDs)
        self.replaceRotatingTrialIDs = replaceRotatingTrialIDs

    @lru_cache(maxsize=32)
    def get_gazedata(self, name):
        """Get Gazedata from Dataset."""
        return GazedataFile(join(self.directory, name),
                            self.replaceRotatingTrialIDs)

    def list_gazedatas(self):
        """List available Gazedatas."""
        return np.unique(self.tbt.data['filename'])


class CSVProtocolDataset(CSVDataset):
    """Implementation of CSV datasets which uses protocols."""

    def __init__(self, directory, protocol):
        """Constructor.

        Input directory - Directory of dataset
              filename - Name of TBT file in directory
              protocol - Protocol definition from icllib.dataset.protocol
        """
        super(CSVProtocolDataset, self).__init__(
            directory,
            protocol.get_tbt_filename(),
            protocol.get_trial_ids_rotate())

        self.protocol = protocol

    @lru_cache(maxsize=32)
    def get_gazedata(self, name):
        """Get Gazedata from Dataset."""
        gazedata = super(CSVProtocolDataset, self).get_gazedata(name)

        # TODO: More generic transformation method
        gazedata.data = self.protocol.get_unified_format(gazedata.data)

        return gazedata


class TrialIterator(object):
    """Iterate trials from dataset."""

    def __init__(self, dataset, tbtfilter=None, gazedatafilter=None):
        """Constructor.

        Input dataset - Dataset to iterate through
              tbtfilter - List of lambda functions to use for filtering entries
                          from tbt file.
              gazedatafilter - List of lambda functions to use for filtering
                               gazedata.
        """
        self.dataset = dataset
        self.gazedatafilter = gazedatafilter

        self.tbtmask = np.array([True] * self.dataset.tbt.data.shape[0])

        if tbtfilter is not None:
            for f in tbtfilter:
                self.tbtmask = self.tbtmask & f(self.dataset.tbt.data)

        self.maskedtbt = self.dataset.tbt.data[self.tbtmask]
        self.current = 0

    def __iter__(self):
        """Return the iterable (=self)."""
        return self

    def __next__(self):
        """Return the next in series."""
        if self.current < self.maskedtbt.shape[0]:
            current_tbt_line = self.maskedtbt[self.current]
            gazedata = self.dataset.get_gazedata(
                current_tbt_line['filename']).data

            trialid = current_tbt_line['trialid']
            print('fname: %s, trialid: %s' % (current_tbt_line['filename'],
                  str(trialid)))
            retval = gazedata[np.array(list(map(str, gazedata['trialid']))) ==
                              str(trialid)]

            gzdmask = np.array([True] * retval.shape[0])

            if self.gazedatafilter is not None:
                for f in self.gazedatafilter:
                    gzdmask = gzdmask & f(retval)
                    print(gzdmask)

            self.current += 1
            return retval[gzdmask]
        else:
            raise StopIteration

    def next(self):
        """Return the next in series.

        Compatibility between Python 2 and 3.
        """
        return self.__next__()


def seek_past_first_sep(filelike):
    """Function to seek past first sep= entry in file."""
    import re

    line = filelike.readline().strip()
    if line != 'sep=' and line != 'sep=,':
        match = re.search('sep=,', str(line))
        if match is None:
            filelike.seek(0)
            return

        print("Seeking to: %i" % match.end())
        filelike.seek(match.end())


def sniff_csv_dialect(filelike):
    """Return CSV dialect object for file."""
    from csv import Sniffer

    loc = filelike.tell()

    dialect = Sniffer().sniff(str(filelike.read(4096)),
                              delimiters=[',', ';', '\t'])

    filelike.seek(loc)

    return dialect


def read_csv_names(filelike, dialect, delimiter=None):
    """Read the header names from CSV file."""
    from csv import reader

    delimiter = delimiter if dialect is None else dialect.delimiter
    r = reader(filelike, dialect, delimiter=delimiter)

    return next(r)

    def filter_unwanted(l):
        return filter(lambda x: x != 'sep=' and x != '', l)

    firstline = filter_unwanted(r.next())
    print('firstline_check: %s' % firstline)
    if len(firstline) == 0:
        print("AEE")
        firstline = filter_unwanted(r.next())

    return firstline


def get_common_name(name):
    """Used for replacing names in TBT file."""
    substitutes = \
        {'condition': 'stimulus',
         'aoi border violation before disengagement or1000ms (during nvs)':
             'aoi border violation',
         'trial number': 'trialid'}

    try:
        return substitutes[name]
    except KeyError:
        return name


class TBTFile():
    """Abstraction class for TBT file."""

    def __init__(self, filename, replaceRotatingTrialIDs=False):
        """Constructor.

        Input filename - Path to TBT file
              replaceRotatingTrialIDs - Set True if Trial IDs rotate
        """
        print('Opening %s:' % filename)
        with open(filename, 'rt') as f:
            seek_past_first_sep(f)
            dialect = sniff_csv_dialect(f)

            r = DictReader(f, delimiter=dialect.delimiter)
            data = [{k: _convert_type(v) for k, v in d.items()} for d in r]
            data = {k: [v[k] for v in data] for k in data[0].keys()}

            names = [str(get_common_name(s.lower().strip()))
                     for s in data.keys()]

            print(names)
            self.names = names

            self.data = np.rec.fromarrays(data.values(),
                                          names=','.join(names))

            if replaceRotatingTrialIDs:
                for gzdfn in np.unique(self.data['filename']):
                    selector = self.data['filename'] == gzdfn
                    self.data['trialid'][selector] = \
                        np.arange(1, np.sum(selector) + 1)

                    print(np.arange(1, np.sum(selector) + 1))
                    print(self.data[selector]['trialid'])


def _get_common_gzname(name):
    """Used for reaplcing names in Gazedata file."""
    substitutes = {'diameterpupillefteye': 'pupil_l',
                   'diameterpupilrighteye': 'pupil_r',
                   'lefteyepupildiameter': 'pupil_l',
                   'righteyepupildiameter': 'pupil_r',
                   'trialnumber': 'trialid',
                   'onscreen': 'userdefined_1'}

    try:
        return substitutes[name]
    except KeyError:
        return name


def _convert_type(s):
    """Convert type from string to python types while reading CSV."""
    converters = [int, float, str]

    for c in converters:
        try:
            return c(s)
        except ValueError:
            pass
        except TypeError:
            pass

    return s


class GazedataFile():
    """Abstraction class for Gazedata file."""

    def __init__(self, filename, replaceRotatingTrialIDs=False):
        """Constructor.

        Input filename - Path to Gazedata file
              replaceRotatingTrialIDs - Set True if Trial IDs rotate
        """
        print("Load GZDF: %s" % filename)
        self.filename = filename

        with open(filename, 'rt') as f:
            seek_past_first_sep(f)
            r = DictReader(f, delimiter='\t')
            data = [{k: _convert_type(v) for k, v in d.items()} for d in r]
            data = {k: [v[k] for v in data] for k in data[0].keys()}

            names = [_get_common_gzname(s.lower().strip())
                     for s in data.keys()]

            self.data = np.rec.fromarrays(data.values(),
                                          names=','.join(names))

        self.data['userdefined_1'][self.data['userdefined_1'] ==
                                   'Stimulus'] = 'Face'

        if replaceRotatingTrialIDs:
            selector = np.array(map(str, self.data['trialid'])) != 'None'
            d = np.array(map(int, self.data['trialid'][selector]))

            diff = np.diff(d)
            diff[diff < 0] = 1
            d = np.cumsum(np.concatenate(([1], diff)))

            print(d)
            print(len(d))
            print(len(self.data['trialid'][selector]))
            print(self.data.dtype)

            if self.data.dtype['trialid'] == 'S4':
                d = np.array(map(str, d))

            self.data['trialid'][selector] = d

        print("Loaded GZDF")
