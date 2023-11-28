# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("Top K Items", version=(0,1))

text = app.text("""
This example shows how to use the window and transformation filter to display
the Top items of a long list of items in decreasing order.
Here we sort the top 10 highest ranking movies of IMDB.
""")
app.place(text)

source = data.movies.url

# Top 10 movies by IMBD rating
spec = alt.Chart(
    source,
).mark_bar().encode(
    x=alt.X('Title:N', sort='-y'),
    y=alt.Y('IMDB_Rating:Q'),
    color=alt.Color('IMDB_Rating:Q')
    
).transform_window(
    rank='rank(IMDB_Rating)',
    sort=[alt.SortField('IMDB_Rating', order='descending')]
).transform_filter(
    (alt.datum.rank < 10)
)

# Create altair chart widget
altair_chart = app.altair_chart(title='Top K Items', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()