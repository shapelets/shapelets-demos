# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("Iowa renewable energy boom", version=(0,1))

text = app.text("""
This example is a fully developed stacked chart using the sample dataset of Iowa's electricity sources.
""")
app.place(text)

source = data.iowa_electricity()

spec = alt.Chart(source, title="Iowa's renewable energy boom").mark_area().encode(
    x=alt.X(
        "year:T",
        title="Year"
    ),
    y=alt.Y(
        "net_generation:Q",
        stack="normalize",
        title="Share of net generation",
        axis=alt.Axis(format=".0%"),
    ),
    color=alt.Color(
        "source:N",
        legend=alt.Legend(title="Electricity source"),
    )
)

# Create altair chart widget
altair_chart = app.altair_chart(title='Iowa renewable energy boom', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()