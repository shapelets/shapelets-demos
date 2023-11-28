# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("Simple strip plot", version=(0,1))

source = data.cars()

spec = alt.Chart(source).mark_tick().encode(
    x='Horsepower:Q',
    y='Cylinders:O'
).interactive()

# Create altair chart widget
altair_chart = app.altair_chart(title='Simple strip plot', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()