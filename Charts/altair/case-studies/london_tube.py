# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("London Tube Lines", version=(0,1))

text = app.text("""
This example shows the London tube lines against the background of the borough boundaries.
""")
app.place(text)

boroughs = alt.topo_feature(data.londonBoroughs.url, 'boroughs')
tubelines = alt.topo_feature(data.londonTubeLines.url, 'line')
centroids = data.londonCentroids.url

background = alt.Chart(boroughs).mark_geoshape(
    stroke='white',
    strokeWidth=2
).encode(
    color=alt.value('#eee'),
).properties(
    width=1400,
    height=1000
)

labels = alt.Chart(centroids).mark_text().encode(
    longitude='cx:Q',
    latitude='cy:Q',
    text='bLabel:N',
    size=alt.value(11),
    opacity=alt.value(0.9)
).transform_calculate(
    "bLabel", "indexof (datum.name,' ') > 0  ? substring(datum.name,0,indexof(datum.name, ' ')) : datum.name"
)

line_scale = alt.Scale(domain=["Bakerloo", "Central", "Circle", "District", "DLR",
                               "Hammersmith & City", "Jubilee", "Metropolitan", "Northern",
                               "Piccadilly", "Victoria", "Waterloo & City"],
                       range=["rgb(137,78,36)", "rgb(220,36,30)", "rgb(255,206,0)",
                              "rgb(1,114,41)", "rgb(0,175,173)", "rgb(215,153,175)",
                              "rgb(106,114,120)", "rgb(114,17,84)", "rgb(0,0,0)",
                              "rgb(0,24,168)", "rgb(0,160,226)", "rgb(106,187,170)"])

lines = alt.Chart(tubelines).mark_geoshape(
    filled=False,
    strokeWidth=2
).encode(
    alt.Color(
        'id:N',
        legend=alt.Legend(
            title=None,
            orient='bottom-right',
            offset=0
        ),
        scale=line_scale
    )
)

spec = background + labels + lines

# Create altair chart widget
altair_chart = app.altair_chart(title='London Tube Lines', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()