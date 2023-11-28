# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data
import pandas as pd

# Instantiate data app
app = DataApp("The U.S. employment crash during the Great Recession", version=(0,1))

text = app.text("""
This example is a fully developed bar chart with negative values using the sample dataset of U.S. employment changes during the Great Recession.
""")
app.place(text)

source = data.us_employment()
presidents = pd.DataFrame([
    {
        "start": "2006-01-01",
        "end": "2009-01-19",
        "president": "Bush"
    },
    {
        "start": "2009-01-20",
        "end": "2015-12-31",
        "president": "Obama"
    }
])

bars = alt.Chart(
    source,
    title="The U.S. employment crash during the Great Recession"
).mark_bar().encode(
    x=alt.X("month:T", title=""),
    y=alt.Y("nonfarm_change:Q", title="Change in non-farm employment (in thousands)"),
    color=alt.condition(
        alt.datum.nonfarm_change > 0,
        alt.value("steelblue"),
        alt.value("orange")
    )
)

rule = alt.Chart(presidents).mark_rule(
    color="black",
    strokeWidth=2
).encode(
    x='end:T'
).transform_filter(alt.datum.president == "Bush")

text = alt.Chart(presidents).mark_text(
    align='left',
    baseline='middle',
    dx=7,
    dy=-135,
    size=11
).encode(
    x='start:T',
    x2='end:T',
    text='president',
    color=alt.value('#000000')
)

spec = (bars + rule + text).properties(width=600)

# Create altair chart widget
altair_chart = app.altair_chart(title='The U.S. employment crash during the Great Recession', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()