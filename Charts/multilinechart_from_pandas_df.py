# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
from vega_datasets import data

# Instantiate DataApp
app = DataApp("multilinechart_from_pandas_df")

# Get sample data frame
source = data.cars()

# Create line chart widget and plot df
lc = app.line_chart(data=source, title="My title")

# Place line chart widget
app.place(lc)

# Register data app
app.register()