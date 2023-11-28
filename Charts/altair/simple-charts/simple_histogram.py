# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("Simple histogram", version=(0,1))

source = data.movies.url

spec = alt.Chart(source).mark_bar().encode(
    alt.X("IMDB_Rating:Q", bin=True),
    y='count()',
).interactive()

# Create altair chart widget
altair_chart = app.altair_chart(title='Simple histogram', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()