# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("Simple stacked area chart", version=(0,1))

source = data.iowa_electricity()

spec = alt.Chart(source).mark_area().encode(
    x="year:T",
    y="net_generation:Q",
    color="source:N"
)

# Create altair chart widget
altair_chart = app.altair_chart(title='Simple stacked area chart', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()