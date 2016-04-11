"""Gazedata Browser based on Bokeh."""
from bokeh.server.utils.plugins import object_page
from bokeh.server.app import bokeh_app
from pandas import DataFrame
import numpy as np
from bokeh.properties import Instance
from bokeh.models.widgets import HBox, VBox, VBoxForm, CheckboxGroup
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Paragraph
from bokeh.plotting import figure
from bokeh.models import Plot, ColumnDataSource
from icllib.dataset import CSVProtocolDataset
from icllib.dataset.protocol import Kinship
from icllib.features import distance_between_vector_and_point
from os.path import join
from scipy.signal import medfilt

basedir = '/Users/mika/ICL/FPA data'

# dataset_dirs = ['ATT_9mo_Gaze',
#                 'Cry_8mo_Gaze',
#                 'HKI_7mo_Gaze/AED',
#                 'HKI_7mo_Gaze/Control',
#                 'HKI_7mo_Gaze/SSRI',
#                 'Kinship_7mo_Gaze',
#                 'SANFR_6mo_Gaze',
#                 'TREC2_7mo_Gaze']
#
# tbt_files = ['disengagement_results.csv',
#              'disengagement_tbt_18_12_2015.csv',
#              'disengagement_results.csv',
#              'disengagement_results.csv',
#              'disengagement_results.csv',
#              'disengagement_results.csv',
#              'SANFRdisengagement_tbt_20_10_2015.csv',
#              'disengagement_results 7mo 31.10.2013.csv']
#
# dataset_names = ['ATT_9mo_Gaze',
#                  'Cry_8mo_Gaze',
#                  'HKI_7mo_Gaze/AED',
#                  'HKI_7mo_Gaze/Control',
#                  'HKI_7mo_Gaze/SSRI',
#                  'Kinship_7mo_Gaze',
#                  'SANFR_6mo_Gaze',
#                  'TREC2_7mo_Gaze']
#
# rotating_ids = [False,
#         False, # Check again
#         False,
#         False,
#         False,
#         False,
#         True,
#         True]

dataset_dirs = ['Kinship']
tbt_files = [Kinship()]
dataset_names = ['Kinship_7mo_Gaze']
rotating_ids = [False]

# datasets = {k: v for k, v in zip(dataset_names,
#             [CSVDataset(join(basedir, d), tbt, r) for d, tbt, r in
#              zip(dataset_dirs, tbt_files, rotating_ids)])}

datasets = {k: v for k, v in zip(dataset_names,
            [CSVProtocolDataset(join(basedir, d), tbt) for d, tbt, r in
             zip(dataset_dirs, tbt_files, rotating_ids)])}
dataset = datasets['Kinship_7mo_Gaze']


# TODO: Some UI picture so that one could know what is going to be done
# TODO: Bokeh apparently has its own attribute/property system which overrides
#       the one in python
# TODO: Figure better name for this class.
class FPAApp(VBox):
    """Foreperiod Analysis App."""

    extra_generated_classes = [['FPAApp', 'FPAApp', 'VBox']]
    inputs = Instance(VBoxForm)

    para = Instance(Paragraph)

    xgazeposlefteyeplot = Instance(Plot)
    ygazeposlefteyeplot = Instance(Plot)
    hbox1 = Instance(HBox)

    xgazeposrighteyeplot = Instance(Plot)
    ygazeposrighteyeplot = Instance(Plot)
    hbox2 = Instance(HBox)

    leftpupil = Instance(Plot)
    rightpupil = Instance(Plot)
    hbox3 = Instance(HBox)

    dsselect = Instance(Select)
    gzdfselect = Instance(Select)
    trialselect = Instance(Select)
    checkboxes = Instance(CheckboxGroup)
    source = Instance(ColumnDataSource)

    @classmethod
    def create(cls):
        """Create instance of this class."""
        obj = cls()

        obj.dsselect = Select(
            title='Dataset',
            name='dataset',
            options=['Dataset'] + datasets.keys()
        )

        obj.gzdfselect = Select(
            # title='GZDF', name='gzdf', options=dataset.list_trials().keys()
            title='GZDF', name='gzdf', options=['Select Dataset first']
        )

        obj.trialselect = Select(
            # title='Trial', name='trial',
            # options=map(str, dataset.list_trials()[obj.gzdfselect.value])
            # title='Trial', name='trial',
            # options=map(str, dataset.list_trials()['11.gazedata'])
            title='Trial', name='trial', options=['Select GZDF first']
        )

        obj.checkboxes = CheckboxGroup(
                labels=['Fill invalid?', 'Median filter']
        )

        obj.para = Paragraph(
                text="AEEHAHAA"
        )

        obj.inputs = VBoxForm(
            children=[
                obj.dsselect,
                obj.gzdfselect,
                obj.trialselect,
                obj.checkboxes,
                obj.para
            ]
        )

        obj.source = ColumnDataSource(data=DataFrame())

        toolset = "box_select,tap,crosshair,pan,reset,resize,save,wheel_zoom"

        def gen_plot_from_col(x):
            # Generate a figure container
            plot = figure(title_text_font_size="12pt",
                          plot_height=400,
                          plot_width=600,
                          tools=toolset,
                          title=x,
                          # x_range=[-0.2, 1.2]
                          )

            # Plot the line by the x,y values in the source property
            plot.asterisk('tettime', x, source=obj.source,
                          line_width=3,
                          line_alpha=0.6,
                          color='#ff0000')

            return plot

        [obj.xgazeposlefteyeplot,
         obj.ygazeposlefteyeplot,
         obj.xgazeposrighteyeplot,
         obj.ygazeposrighteyeplot,
         obj.leftpupil,
         obj.rightpupil] = map(gen_plot_from_col,
                               ['left_eye_x_relative',
                                'left_eye_y_relative',
                                'right_eye_x_relative',
                                'right_eye_y_relative',
                                'dists',
                                'right_eye_pupil_mm'])

        obj.children.append(obj.inputs)

        obj.hbox1 = HBox(children=[obj.xgazeposlefteyeplot,
                                   obj.xgazeposrighteyeplot])
        obj.hbox2 = HBox(children=[obj.ygazeposlefteyeplot,
                                   obj.ygazeposrighteyeplot])
        obj.hbox3 = HBox(children=[obj.leftpupil, obj.rightpupil])

        obj.children.append(obj.hbox1)
        obj.children.append(obj.hbox2)
        obj.children.append(obj.hbox3)

        return obj

    def setup_events(self):
        """Bind events."""
        super(FPAApp, self).setup_events()

        if not self.gzdfselect:
            return

        self.dsselect.on_change('value', self, 'ds_change')
        self.gzdfselect.on_change('value', self, 'gzdf_change')
        self.trialselect.on_change('value', self, 'trial_change')
        self.checkboxes.on_change('active', self, 'update_data')

    def ds_change(self, obj, attrname, old, new):
        """Callback for Dataset change."""
        dataset = datasets[self.dsselect.value]
        self.gzdfselect.options = list(dataset.list_gazedatas())
        self.gzdfselect.value = self.gzdfselect.options[0]
        self.gzdf_change(obj, attrname, old, new)

    def _shift_hold(self, combination):
        if combination == -1:
            return 'INVALID'
        elif combination < 1000:
            return 'SHIFT'
        else:
            return 'HOLD'

    def gzdf_change(self, obj, attrname, old, new):
        """Callback for GazedataFile change."""
        dataset = datasets[self.dsselect.value]
        tbtrows = dataset.tbt.data[dataset.tbt.data['filename'] ==
                                   self.gzdfselect.value]
        shiftholds = map(self._shift_hold, tbtrows['combination'])
        # opts = map(str, np.unique(dataset.tbt.data[
        #  dataset.tbt.data['filename'] == self.gzdfselect.value]['trialid']))
        # opts = {k: v for k, v in zip(opts, [o.upper() for o in opts])}
        # opts = [{'value': o, 'name': o + 'aaa'} for o in opts]
        opts = [{'name': o + ' ' + s, 'value': o}
                for o, s in zip(map(str, tbtrows['trialid']), shiftholds)]
        self.trialselect.options = opts
        self.trialselect.value = self.trialselect.options[0]['value']
        self.trial_change(obj, attrname, old, new)

    def trial_change(self, obj, attrname, old, new):
        """Callback for trial change."""
        self.update_data(obj, attrname, old, new)

    def _get_relevant_gazedata(self, dataset, name):
        """Function gets relevant data from dataset."""
        # Getting Gazedata
        print("Getting Gazedata name: %s" % name)
        assert name in dataset.list_gazedatas()
        assert self.trialselect.value is not None
        gzdf = dataset.get_gazedata(name)

        # Filtering relevant parts of gazedata
        trial = gzdf.data[np.array(map(str, gzdf.data['trialid'])) ==
                          str(self.trialselect.value)]
        trial = trial[trial['userdefined_1'] == 'Face']
        assert trial.shape[0] != 0
        trial['tettime'] = trial['tettime'] - trial['tettime'][0]
        trial['tettime'] = trial['tettime'] / 1000.0

        return trial

    def update_data(self, obj, attrname, old, new):
        """Update ColumnDataSource."""
        dataset = datasets[self.dsselect.value]

        trial = self._get_relevant_gazedata(dataset, self.gzdfselect.value)

        # Get corresponding row from tbt file
        tbtrow = dataset.tbt.data[dataset.tbt.data['filename'] ==
                                  self.gzdfselect.value]
        tbtrow = tbtrow[np.array(map(str, tbtrow['trialid'])) ==
                        self.trialselect.value]

        # Tell if it was SHIFT or HOLD
        self.para.text = 'SHIFT' if tbtrow['combination'] < 1000 else 'HOLD'
        self.para.text = \
            'INVALID' if tbtrow['combination'] == -1 else self.para.text

        self.para.text += ' (Combination: %f)' % tbtrow['combination']

        # Convert trial (numpy array) into Pandas DataFrame
        df = DataFrame(trial)

        # Do "Fill invalid" if checkbox was selected
        if 0 in self.checkboxes.active:
            df[df == -1] = np.nan
            df = df.fillna(method='ffill').fillna(method='bfill')

        fields_of_interest = ['left_eye_x_relative',
                              'left_eye_y_relative',
                              'right_eye_x_relative',
                              'right_eye_y_relative',
                              'left_eye_pupil_mm',
                              'right_eye_pupil_mm']

        # Do "Median filter" if checkbox was selected
        if 1 in self.checkboxes.active:
            for foi in fields_of_interest:
                # medfilt(np.pad(arr, 37, mode='edge'), kernel_size=37)[37:-37]
                df[foi] = medfilt(np.pad(np.array(df[foi]), 37, mode='edge'),
                                  kernel_size=37)[37:-37]

        # Calculate combined eyes
        # Calculate columns
        proto = Kinship()
        eye_pos = np.vstack([df['left_eye_x_relative'],
                             df['left_eye_y_relative'],
                             df['right_eye_x_relative'],
                             df['right_eye_y_relative']]).T
        ds = distance_between_vector_and_point(eye_pos,
                                               proto.eye_positions['fearful'])
        generated_df = DataFrame(ds, columns=['dists'])

        # Join columns
        df = df.join(generated_df)

        print("DF %s" % df['dists'])

        self.source.data = ColumnDataSource.from_df(df)


# TODO: Change function name
@bokeh_app.route('/fpa/')
@object_page('fpa')
def make_crossfilter():
    """Create Bokeh App."""
    app = FPAApp.create()

    return app
