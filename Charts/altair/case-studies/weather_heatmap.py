# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("Seattle Weather Heatmap", version=(0,1))

text = app.text("""
This example shows the 2010 daily high temperature (F) in Seattle, WA.
""")
app.place(text)

# Since the data is more than 5,000 rows we'll import it from a URL
source = data.seattle_temps.url

spec = alt.Chart(
    source,
    title="2010 Daily High Temperature (F) in Seattle, WA"
).mark_rect().encode(
    x='date(date):O',
    y='month(date):O',
    color=alt.Color('max(temp):Q', scale=alt.Scale(scheme="inferno")),
    tooltip=[
        alt.Tooltip('monthdate(date):T', title='Date'),
        alt.Tooltip('max(temp):Q', title='Max Temp')
    ]
).properties(width=550)

# Create altair chart widget
altair_chart = app.altair_chart(title='Seattle Weather Heatmap', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()