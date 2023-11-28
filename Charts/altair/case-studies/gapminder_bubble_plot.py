# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("Gapminder Bubble Plot", version=(0,1))

text = app.text("""
This example shows how to make a bubble plot showing the correlation between health and income for 187 countries in the world.
""")
app.place(text)

source = data.gapminder_health_income.url

spec = alt.Chart(source).mark_circle().encode(
    alt.X('income:Q', scale=alt.Scale(type='log')),
    alt.Y('health:Q', scale=alt.Scale(zero=False)),
    size='population:Q'
).interactive()

# Create altair chart widget
altair_chart = app.altair_chart(title='Gapminder Bubble Plot', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()