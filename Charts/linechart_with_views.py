# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import pandas as pd
from shapelets.apps import DataApp
from shapelets.apps.widgets import View

# Create data app
app = DataApp(name="linechart_with_views")

# Load data from csv
df = pd.read_csv('../Resources/mitdb102.csv', header=None, index_col=0, names=['MLII', 'V1'], skiprows=200000, nrows=20000)

# Set index
df.index = pd.to_datetime(df.index, unit='s')

# Highlight view by setting index start and end of the view to be highlighted
view = View(1000,2000)

# Create linechart and add view
line_chart1 = app.line_chart(title='MLII', data=df['MLII'], views=[view])

# Add linechart to data app
app.place(line_chart1)

# Register data app
app.register()
