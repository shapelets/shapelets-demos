# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
import pandas as pd

# Instantiate data app
app = DataApp("Simple bar chart", version=(0,1))

source = pd.DataFrame({
    'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
    'b': [28, 55, 43, 91, 81, 53, 19, 87, 52]
})

spec = alt.Chart(source).mark_bar().encode(
    x='a',
    y='b'
).interactive()

# Create altair chart widget
altair_chart = app.altair_chart(title='The Title', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()