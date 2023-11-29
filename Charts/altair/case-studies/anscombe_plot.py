# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("Anscombe Quartet", version=(0,1))

text = app.text("""
This example shows how to use the column channel to make a trellis plot. Anscombe's Quartet is a famous dataset constructed by Francis Anscombe. Common summary statistics are identical for each subset of the data, despite the subsets having vastly different characteristics.
""")
app.place(text)

source = data.anscombe()

spec = alt.Chart(source).mark_circle().encode(
    alt.X('X', scale=alt.Scale(zero=False)),
    alt.Y('Y', scale=alt.Scale(zero=False)),
).facet(facet='Series', columns=2)

# Create altair chart widget
altair_chart = app.altair_chart(title='Anscombe Quartet', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()