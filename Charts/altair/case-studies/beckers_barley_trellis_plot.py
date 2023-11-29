# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("Becker Barley Trellis Plot", version=(0,1))

text = app.text("""
The example demonstrates the trellis charts created by Richard Becker, William Cleveland and others in the 1990s. Using the visualization technique below they identified an anomoly in a widely used agriculatural dataset, which they termed The Morris Mistake. It became their favored way of showcasing the power of this pioneering plot.
""")
app.place(text)

source = data.barley()

spec = alt.Chart(source, title="The Morris Mistake").mark_point().encode(
    alt.X(
        'yield:Q',
        title="Barley Yield (bushels/acre)",
        scale=alt.Scale(zero=False),
        axis=alt.Axis(grid=False)
    ),
    alt.Y(
        'variety:N',
        title="",
        sort='-x',
        axis=alt.Axis(grid=True)
    ),
    color=alt.Color('year:N', legend=alt.Legend(title="Year")),
    row=alt.Row(
        'site:N',
        title="",
        sort=alt.EncodingSortField(field='yield', op='sum', order='descending'),
    )
).configure_view(stroke="transparent")

# Create altair chart widget
altair_chart = app.altair_chart(title='Becker Barley Trellis Plot', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()