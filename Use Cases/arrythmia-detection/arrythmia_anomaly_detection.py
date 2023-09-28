# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import pandas as pd
from shapelets.apps import DataApp
from shapelets.apps.widgets import LineChart, View
from matrixprofile import *
from matrixprofile.discords import discords
import datetime
import numpy as np #required by matrixprofile, do not remove
import sys #required by matrixprofile, do not remove

def computeAndPlotAnomalies(ts: pd.Series, window_size: int, top_k: int) -> LineChart:
    # Find discords
    mp = matrixProfile.scrimp_plus_plus(ts.to_numpy(), m=window_size, runtime=1)
    discords_idx = discords(np.append(mp[0],np.zeros(window_size-1)+np.nan), ex_zone=int(window_size/2), k=top_k)
    views = []
    for m in discords_idx:
        views.append(View(start=ts.index[m], end=ts.index[m]+datetime.timedelta(milliseconds=window_size)))
    return LineChart(title='Anomalies', data=ts, views=views)

app = DataApp(name="Anomaly detection",
              version=(1,0),
              description="In this app, Data from the MIT-BIH Arrhythmia Database (mitdb) are analyzed looking for premature ventricular contractions (PVC).")

df = pd.read_csv('mitdb102.csv', header=None, index_col=0, names=['MLII', 'V1'], skiprows=200000, nrows=20000)

df.index = pd.to_datetime(df.index, unit='s')

df = df.resample('1ms').asfreq().ffill()

hl = app.horizontal_layout()
app.place(hl)

vl1 = app.vertical_layout()
vl2 = app.vertical_layout()
hl.place(vl1)
hl.place(vl2)

windowSize = app.number_input(title="Window size value [ms]", value=400)

vl1.place(windowSize, width=6)

top_k = app.slider(title="Desired number of anomalies: ", min_value=1, max_value=20, step=1, value=1)

vl2.place(top_k, width=6)

button = app.button("Execute anomaly-detection")

app.place(button)

line_chart1 = app.line_chart(title='MLII', data=df['MLII'])

app.place(line_chart1)

line_chart1.bind(computeAndPlotAnomalies, df['MLII'], windowSize, top_k, triggers=[button],mute=[windowSize,top_k])

app.register()