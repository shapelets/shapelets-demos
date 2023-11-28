# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("One Dot Per Zipcode", version=(0,1))

text = app.text("""
This example shows a geographical plot with one dot per zipcode.
""")
app.place(text)

# Since the data is more than 5,000 rows we'll import it from a URL
source = data.zipcodes.url

spec = alt.Chart(source).transform_calculate(
    "leading digit", alt.expr.substring(alt.datum.zip_code, 0, 1)
).mark_circle(size=3).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='leading digit:N',
    tooltip='zip_code:N'
).project(
    type='albersUsa'
).properties(
    width=1300,
    height=800
)

# Create altair chart widget
altair_chart = app.altair_chart(title='One Dot Per Zipcode', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()