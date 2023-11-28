# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("Locations of US Airports", version=(0,1))

text = app.text("""
This is a layered geographic visualization that shows the positions of US
airports on a background of US states.
""")
app.place(text)

airports = data.airports()
states = alt.topo_feature(data.us_10m.url, feature='states')

# US states background
background = alt.Chart(states).mark_geoshape(
    fill='lightgray',
    stroke='white'
).properties(
    width=1000,
    height=600
).project('albersUsa')

# airport positions on background
points = alt.Chart(airports).mark_circle(
    size=10,
    color='steelblue'
).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    tooltip=['name', 'city', 'state']
)

spec = background + points

# Create altair chart widget
altair_chart = app.altair_chart(title='Locations of US Airports', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()