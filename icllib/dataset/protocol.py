"""Definitions for reading different datasets."""
import numpy as np
import numpy.lib.recfunctions


class ProtocolError(Exception):
    """Exception class for Protocol Errors."""

    pass


def _append_column(data, field_name, field_data):
    """Append column into NumPy structured array."""
    return numpy.lib.recfunctions.append_fields(
        data, field_name, field_data, usemask=False, asrecarray=False)


class Protocol(object):
    """Base class for Protocol definitions.

    A way to interpret CSV dataset into a common format is called Protocol.
    """

    def __init__(self):
        """Constructor."""
        self.rotating_trial_ids = False

    def get_gazedata_substitutes(self):
        """Get gazedata column name substitutes."""
        return self.substitutes

    def get_gazedata_provides(self):
        """Get what kind of data is provided by the dataset."""
        return self.provides

    def get_tbt_filename(self):
        """Get the name of TBT file."""
        return self.tbtfile

    def get_trial_ids_rotate(self):
        """Tell if trial ids rotate or not."""
        return self.rotating_trial_ids

    def get_time(self, data):
        """Calculate time from TIME1 and TIME2 columns."""
        return np.float64(data['time1']) + np.float64(data['time2']) / 10**6

    def _replace_field_name(self, name):
        """Do replacement for one field name in gazedata."""
        try:
            s = self.get_gazedata_substitutes()
            s = {k.lower(): v for k, v in s.items()}
            return s[name]
        except KeyError:
            return name.lower()

    def get_unified_format(self, data):
        """Get data in common unified numpy format.

        Does conversion of column names and generates new columns to generate
        numpy array in commonly understandable format.
        """
        # TODO: Improve. How? Add more unified fields.

        # Rename fields
        data.dtype.names = \
            tuple(map(self._replace_field_name, data.dtype.names))

        # Calculate time and append it as a new column to data
        time = self.get_time(data)
        data = _append_column(data, 'unified_time', time)

        return data


class Kinship(Protocol):
    """Protocol for Kinship dataset."""

    substitutes = {
        "XGazePosLeftEye": "left_eye_x_relative",
        "YGazePosLeftEye": "left_eye_y_relative",
        "XGazePosRightEye": "right_eye_x_relative",
        "YGazePosRightEye": "right_eye_y_relative",
        "pupil_l": "left_eye_pupil_mm",
        "pupil_r": "right_eye_pupil_mm"
    }

    provides = substitutes.values()

    tbtfile = "disengagement_results.csv"

    stimulus_names = {
        '1': 'fearful.bmp',
        '2': 'fearful.bmp',
        '3': 'control.bmp',
        '4': 'control.bmp',
        '5': 'happy.bmp',
        '6': 'happy.bmp',
        '7': 'neutral.bmp',
        '8': 'neutral.bmp'
    }

    aoi = {
        'center': [0.3, 0.7, 0.1, 0.9],
        'right': [0.7, 1, 0.1, 0.9],
        'left': [0, 0.3, 0.1, 0.9],
        'facecoord': [0.335, 0.665, 0.185, 0.815],
        'imgright': [0.9, 1, 0.3, 0.7],
        'imgleft': [0, 0.1, 0.3, 0.7]
    }

    # Eye positions in fearful1 and 2 images.
    # TODO: Is there more eye position information?
    # TODO: Find out coordinate system for the points.
    eye_pos_fearful1 = np.array([0.458200253, 0.494150851,
                                 0.537095014, 0.491226276])
    eye_pos_fearful2 = np.array([0.446981273, 0.481064619,
                                 0.54071404,  0.48496308])

    # Use mean for "fearful" eye positions for now
    eye_positions = {
        'fearful': np.mean([eye_pos_fearful1, eye_pos_fearful2], axis=0)
    }
