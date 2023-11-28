# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("Top-K plot with Others", version=(0,1))

text = app.text("""
This example shows how to use aggregate, window, and calculate transfromations
to display the top-k directors by average worldwide gross while grouping the 
remaining directors as All Others.
""")
app.place(text)

source = data.movies.url

spec = alt.Chart(source).mark_bar().encode(
    x=alt.X("aggregate_gross:Q", aggregate="mean", title=None),
    y=alt.Y(
        "ranked_director:N",
        sort=alt.Sort(op="mean", field="aggregate_gross", order="descending"),
        title=None,
    ),
).transform_aggregate(
    aggregate_gross='mean(Worldwide_Gross)',
    groupby=["Director"],
).transform_window(
    rank='row_number()',
    sort=[alt.SortField("aggregate_gross", order="descending")],
).transform_calculate(
    ranked_director="datum.rank < 10 ? datum.Director : 'All Others'"
).properties(
    title="Top Directors by Average Worldwide Gross",
)

# Create altair chart widget
altair_chart = app.altair_chart(title='Top-K plot with Others', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()