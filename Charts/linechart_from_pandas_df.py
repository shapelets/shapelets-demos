# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets as sh
sh.login(user_name='admin',password='admin')

from shapelets.apps import DataApp
import pandas as pd

# Instantiate DataApp
app = DataApp("linechart_from_pandas_df")

# Create pd.DataFrame from list
data = {"series1":[420, 380, 390]}
df = pd.DataFrame(data=data, index=range(len(data["series1"])))

# Create line chart widget and plot df
lc = app.line_chart(data=df, title="My title")

# Place line chart widget
app.place(lc)

# Register data app
app.register()
