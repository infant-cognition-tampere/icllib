"""Gazedata Browser based on Bokeh."""
from bokeh.io import curdoc
from pandas import DataFrame
import numpy as np
from bokeh.models.layouts import HBox, VBox
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.widgets.inputs import Select
from bokeh.plotting import Figure
from bokeh.models import ColumnDataSource
from icllib.dataset.protocol import TREC2, _append_column
from icllib.features import distance_between_vector_and_point
from scipy.signal import medfilt
from tables import open_file
from pandas import Series
from functools32 import lru_cache
import scipy.stats as st


fileh = open_file('trec2.h5', 'r')

eye_coords_columns = ['left_eye_x_relative',
                      'left_eye_y_relative',
                      'right_eye_x_relative',
                      'right_eye_y_relative']

data_columns_to_plot = eye_coords_columns + \
                       ['left_eye_pupil_mm',
                        'right_eye_pupil_mm']

additional_columns = ['unified_time',
                      'userdefined_1',
                      'trialid',
                      'validitylefteye',
                      'validityrighteye',
                      'stim']

generated_columns = ['combined_eye_x_relative',
                     'combined_eye_y_relative',
                     'combined_eye_pupil_mm',
                     'distance_from_eyes',
                     'distance_from_left_eye',
                     'distance_from_right_eye']

bokeh_toolset = "box_select,tap,crosshair,pan,reset,resize,save,wheel_zoom"


def generate_combined_eye(left,
                          right,
                          validityleft,
                          validityright):
    """Generate combined eye values from separate values."""
    left_eye_valid = validityleft < 2
    right_eye_valid = validityright < 2
    both_eyes_valid = left_eye_valid == right_eye_valid
    either_eyes_valid = left_eye_valid | right_eye_valid
    only_left_eye_valid = left_eye_valid == ~both_eyes_valid
    only_right_eye_valid = right_eye_valid == ~both_eyes_valid

    result = np.zeros(shape=(either_eyes_valid.shape[0],),
                      dtype=np.float64)
    result[:] = np.nan

    means = np.vstack([left, right]).mean(axis=0)

    result[both_eyes_valid] = means[both_eyes_valid]
    result[only_left_eye_valid] = left[only_left_eye_valid]
    result[only_right_eye_valid] = right[only_right_eye_valid]

    return result


def gen_plot_from_col(x):
    """Generate a figure container."""
    plot = Figure(title_text_font_size="12pt",
                  plot_width=600,
                  plot_height=400,
                  tools=bokeh_toolset,
                  title=x
                  # x_range=[-0.2, 1.2]
                  )

    # Plot the line by the x,y values in the source property
    plot.asterisk('unified_time', x, source=source,
                  line_width=3,
                  line_alpha=0.6,
                  color='#ff0000')

    return plot


def gen_mean_plot_from_col(x):
    """Generate a figure container."""
    plot = Figure(title_text_font_size="12pt",
                  plot_width=600,
                  plot_height=400,
                  tools=bokeh_toolset,
                  title=x
                  # x_range=[-0.2, 1.2]
                  )

    # Plot the line by the x,y values in the source property
    # Shift Green
    # plot.asterisk('unified_time', x + '_lower_ci_shift', source=mean_source,
    #               line_width=3,
    #               line_alpha=0.6,
    #               alpha=0.5,
    #               color='#00ff00',
    #               legend='Shift Lower CI')
    # plot.asterisk('unified_time', x + '_upper_ci_shift', source=mean_source,
    #               line_width=3,
    #               line_alpha=0.6,
    #               alpha=0.5,
    #               color='#00ff00',
    #               legend='Shift Upper CI')
    plot.segment(x0='unified_time',
                 x1='unified_time',
                 y0=x + '_lower_ci' + '_shift',
                 y1=x + '_upper_ci' + '_shift',
                 line_width=3,
                 line_alpha=0.6,
                 alpha=0.5,
                 color='#00ff00',
                 source=mean_source)
    plot.asterisk('unified_time', x + '_shift', source=mean_source,
                  line_width=1.0,
                  line_alpha=0.6,
                  alpha=1.0,
                  color='#009900',
                  legend='Shift')

    # Hold Red
    # plot.asterisk('unified_time', x + '_lower_ci_hold', source=mean_source,
    #               line_width=3,
    #               line_alpha=0.6,
    #               alpha=0.5,
    #               color='#ff0000',
    #               legend='Hold Lower CI')
    # plot.asterisk('unified_time', x + '_upper_ci_hold', source=mean_source,
    #               line_width=3,
    #               line_alpha=0.6,
    #               alpha=0.5,
    #               color='#ff0000',
    #               legend='Hold Upper CI')
    plot.segment(x0='unified_time',
                 x1='unified_time',
                 y0=x + '_lower_ci' + '_hold',
                 y1=x + '_upper_ci' + '_hold',
                 line_width=3,
                 line_alpha=0.6,
                 alpha=0.5,
                 color='#ff0000',
                 source=mean_source)
    plot.asterisk('unified_time', x + '_hold', source=mean_source,
                  line_width=1.0,
                  line_alpha=0.6,
                  alpha=1.0,
                  color='#990000',
                  legend='Hold')

    return plot


def gen_plots():
    """Generate all plots."""
    return map(gen_plot_from_col,
               data_columns_to_plot + generated_columns)


def gen_mean_plots():
    """Generate all plots for mean."""
    return map(gen_mean_plot_from_col,
               data_columns_to_plot + generated_columns)


@lru_cache(maxsize=8)
def get_gazedata(gzdname):
    """Get gazedata from dataset."""
    # Get table from HDF5
    table = fileh.get_node('/gazedatas/', gzdname).read()

    # Select needed columns from table
    table = table[data_columns_to_plot + additional_columns]

    return table


def get_valid_from_tbt():
    """Get valid entries from TBT data."""
    tbt = fileh.get_node('/tbt').read()
    tbt = tbt[tbt['combination'] != -1]

    return tbt


def filter_trials_from_gazedata(gzd):
    """Filter Face period from gazedata."""
    return gzd[gzd['userdefined_1'] == 'Face']


def filter_trial_from_gazedata(gzd, trialid):
    """Filter trial and foreperiod from gazedata."""
    trial = gzd
    trial = filter_trials_from_gazedata(trial)
    trial = trial[np.array(map(str, trial['trialid'])) == trialid]
    assert trial.shape[0] != 0
    trial['unified_time'] = trial['unified_time'] - trial['unified_time'][0]

    return trial


def get_time_discretized(time, data, bincount):
    """Bin data."""
    old_index = time

    # Generate bins from 0 to 1
    new_index = np.linspace(0, 1, bincount)

    # Helper function to resample one field of data
    def reindex(f):
        x = data[f]
        assert type(x[0]) is not np.string_

        if f in eye_coords_columns:
            x[x > 1.5] = np.nan
            x[x < -0.5] = np.nan

        s = Series(x, index=old_index)

        # Fill missing values in data using nearest method
        s_reindexed = s.reindex(new_index, method='nearest')
        s_reindexed[s_reindexed < 0] = np.nan
        s_reindexed[np.isinf(s_reindexed)] = np.nan
        # s_reindexed[s_reindexed == -np.inf] = np.nan
        s_reindexed = s_reindexed.fillna(method='ffill').fillna(method='bfill')

        arr = np.array(s_reindexed)
        arr = medfilt(np.pad(arr, 37, mode='edge'), kernel_size=37)[37:-37]

        return arr

    # Resample every field separately and combine results to one
    # NumPy array
    cols = [reindex(f) for f in data.dtype.names]

    return np.rec.fromarrays(cols, names=data.dtype.names)


def _get_data(gzdname, trialid):
    print('==== _get_data(%s, %s)' % (gzdname, trialid))
    table = get_gazedata(gzdname)
    table = filter_trial_from_gazedata(table, trialid)
    # TODO: Perhaps generate additional column here
    result = get_time_discretized(table['unified_time'],
                                  table[['unified_time'] +
                                        data_columns_to_plot],
                                  300)

    # TODO: Generate additional columns
    combined_eye_x = generate_combined_eye(table['left_eye_x_relative'],
                                           table['right_eye_x_relative'],
                                           table['validitylefteye'],
                                           table['validityrighteye'])
    combined_eye_y = generate_combined_eye(table['left_eye_y_relative'],
                                           table['right_eye_y_relative'],
                                           table['validitylefteye'],
                                           table['validityrighteye'])
    combined_pupil = generate_combined_eye(table['left_eye_pupil_mm'],
                                           table['right_eye_pupil_mm'],
                                           table['validitylefteye'],
                                           table['validityrighteye'])
    stim = np.unique(table['stim'])
    assert stim.shape[0] == 1
    stim = stim[0]

    proto = TREC2()
    try:
        eye_positions = proto.eye_positions[stim]
    except KeyError:
        eye_positions = np.array([0.5, 0.5, 0.5, 0.5])

    eye1_pos = eye_positions[0:1]
    eye2_pos = eye_positions[2:4]

    comb_eye = np.vstack([combined_eye_x, combined_eye_y]).T
    dist_from_eye1 = distance_between_vector_and_point(comb_eye,
                                                       eye1_pos)
    dist_from_eye2 = distance_between_vector_and_point(comb_eye,
                                                       eye2_pos)

    # TODO: Visualize eyes separately

    dist_from_eye = np.min(np.vstack([dist_from_eye1, dist_from_eye2]), axis=0)

    # TODO: Reindex columns
    generated = np.rec.fromarrays([combined_eye_x,
                                   combined_eye_y,
                                   combined_pupil,
                                   dist_from_eye,
                                   dist_from_eye1,
                                   dist_from_eye2],
                                  names=generated_columns)

    generated = get_time_discretized(table['unified_time'], generated, 300)

    for s in generated_columns:
        result = _append_column(result, s, generated[s])

    return result


def get_data(gzdname, trialid):
    """Get ColumnDataSource with requested data."""
    table = _get_data(gzdname, trialid)

    return ColumnDataSource(data=DataFrame(table))


def _calc_mean_for_datas(datas):
    def _mean_field(x):
        values = np.vstack([d[x] for d in datas])

        return np.nanmean(values, axis=0)

    cols = [_mean_field(f) for f in datas[0].dtype.names]

    return np.rec.fromarrays(cols, names=datas[0].dtype.names)


def _calc_confidence_intervals_for_datas(datas, confidence=0.95):
    def _ci_field(x):
        values = np.vstack([d[x] for d in datas])

        return st.t.interval(confidence,
                             len(values) - 1,
                             loc=np.mean(values, axis=0),
                             scale=st.sem(values, axis=0))

    fieldnames = datas[0].dtype.names
    cols = [_ci_field(f) for f in fieldnames]
    cols_lower = {name + '_lower_ci': field[0] for name, field in
                  zip(fieldnames, cols)}
    cols_upper = {name + '_upper_ci': field[1] for name, field in
                  zip(fieldnames, cols)}

    cols_ci = dict()
    cols_ci.update(cols_lower)
    cols_ci.update(cols_upper)

    return np.rec.fromarrays(cols_ci.values(), names=cols_ci.keys())


# def for_each_field(fun, arr):
#     cols = [fun(f) for f in arr.

# TODO: Function to get stimulus from actual gazedata


def _get_data_for_mean_calc():
    # Get all data based on filters (fearful, valid trialds)
    tbt = get_valid_from_tbt()
    tbt = tbt[tbt['stimulus'] == 'fearful.bmp']

    # Group data on classes (HOLD, SHIFT)
    idx_shift = tbt['combination'] < 1000
    idx_hold = ~idx_shift

    tbt_shift = tbt[idx_shift]
    tbt_hold = tbt[idx_hold]

    # Calculate means
    data_shift = map(lambda x: _get_data(*x),
                     zip(tbt_shift['filename'],
                         map(str, tbt_shift['trialid'])))
    data_hold = map(lambda x: _get_data(*x),
                    zip(tbt_hold['filename'],
                        map(str, tbt_hold['trialid'])))

    return data_shift, data_hold


def get_mean_gazedata():
    """Generate mean ColumnDataSource."""
    (data_shift, data_hold) = _get_data_for_mean_calc()

    means_shift = _calc_mean_for_datas(data_shift)
    means_hold = _calc_mean_for_datas(data_hold)
    ci_shift = _calc_confidence_intervals_for_datas(data_shift)
    ci_hold = _calc_confidence_intervals_for_datas(data_hold)

    def append_tag(s, t):
        if s == 'unified_time':
            return s
        else:
            return s + t

    means_shift.dtype.names = \
        map(lambda x: append_tag(x, '_shift'), means_shift.dtype.names)
    means_hold.dtype.names = \
        map(lambda x: append_tag(x, '_hold'), means_hold.dtype.names)
    ci_shift.dtype.names = \
        map(lambda x: append_tag(x, '_shift'), ci_shift.dtype.names)
    ci_hold.dtype.names = \
        map(lambda x: append_tag(x, '_hold'), ci_hold.dtype.names)

    def _append_columns(dest, src):
        dest2 = dest.copy()
        for f in src.dtype.names:
            try:
                dest2 = _append_column(dest2, f, src[f])
            except ValueError:
                pass
        return dest2

    # Combine columns into one numpy recarray
    dest = means_shift.copy()
    dest = _append_columns(dest, means_hold)
    dest = _append_columns(dest, ci_shift)
    dest = _append_columns(dest, ci_hold)

    # Construct pandas DataFrame
    df = DataFrame(dest)

    return ColumnDataSource(data=df)


def list_gazedatas():
    """List gazedatas from TBT."""
    return list(np.unique(get_valid_from_tbt()['filename']))


def list_trialids(gzdname):
    """List trial ids from TBT based on Gazedata name."""
    tbt = get_valid_from_tbt()
    tbt = tbt[tbt['filename'] == gzdname]

    return list(map(str, np.unique(tbt['trialid'])))


# Initial values
gzdname = list_gazedatas()[0]
trialid = '1'  # TODO: Get trials from gazedata


source = get_data(gzdname, trialid)
mean_source = get_mean_gazedata()


def update_plot(attrname, old, new):
    """Callback to update plot when user changes values."""
    gzdname = gazedata_select.value
    trial_select.options = list_trialids(gzdname)
    source.data.update(get_data(gazedata_select.value,
                                trial_select.value).data)


gazedata_select = Select(value=gzdname,
                         title='Gazedata',
                         options=list_gazedatas())
trial_select = Select(value=trialid,
                      title='Trial',
                      options=list_trialids(gzdname))
# plot = gen_plot_from_col('left_eye_x_relative')
plots = VBox(*tuple(gen_plots()))

# Mean Plot
mean_plots = VBox(*tuple(gen_mean_plots()))

# Callbacks
gazedata_select.on_change('value', update_plot)
trial_select.on_change('value', update_plot)

controls = VBox(gazedata_select, trial_select)

# Define Tabs
single_data_hbox = HBox(controls, plots)
single_data_panel = Panel(child=single_data_hbox, title="Single data")

# TODO: Mean view
mean_data_hbox = HBox(mean_plots)
mean_data_panel = Panel(child=mean_data_hbox, title="Mean data")

tabs = Tabs(tabs=[single_data_panel, mean_data_panel])

curdoc().add_root(tabs)
