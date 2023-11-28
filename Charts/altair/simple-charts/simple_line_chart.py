# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
import numpy as np
import pandas as pd

# Instantiate data app
app = DataApp("Simple line chart", version=(0,1))

x = np.arange(100)
source = pd.DataFrame({
  'x': x,
  'f(x)': np.sin(x / 5)
})

spec = alt.Chart(source).mark_line().encode(
    x='x',
    y='f(x)'
).interactive()

# Create altair chart widget
altair_chart = app.altair_chart(title='Simple line chart', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()