# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("US Population: Wrapped Facet", version=(0,1))

text = app.text("""
This chart visualizes the age distribution of the US population over time,
using a wrapped faceting of the data by decade.
""")
app.place(text)

source = data.population.url

spec = alt.Chart(source).mark_area().encode(
    x='age:O',
    y=alt.Y(
        'sum(people):Q',
        title='Population',
        axis=alt.Axis(format='~s')
    ),
    facet=alt.Facet('year:O', columns=5),
).properties(
    title='US Age Distribution By Year',
    width=90,
    height=80
)

# Create altair chart widget
altair_chart = app.altair_chart(title='US Population: Wrapped Facet', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()