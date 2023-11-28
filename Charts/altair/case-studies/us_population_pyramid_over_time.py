# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("US Population Pyramid Over Time", version=(0,1))

text = app.text("""
A population pyramid shows the distribution of age groups within a population.
It uses a slider widget that is bound to the year to visualize the age
distribution over time.
""")
app.place(text)

source = data.population.url

slider = alt.binding_range(min=1850, max=2000, step=10)
select_year = alt.selection_single(name='year', fields=['year'],
                                   bind=slider, init={'year': 2000})

base = alt.Chart(source).add_selection(
    select_year
).transform_filter(
    select_year
).transform_calculate(
    gender=alt.expr.if_(alt.datum.sex == 1, 'Male', 'Female')
).properties(
    width=250
)


color_scale = alt.Scale(domain=['Male', 'Female'],
                        range=['#1f77b4', '#e377c2'])

left = base.transform_filter(
    alt.datum.gender == 'Female'
).encode(
    y=alt.Y('age:O', axis=None),
    x=alt.X('sum(people):Q',
            title='population',
            sort=alt.SortOrder('descending')),
    color=alt.Color('gender:N', scale=color_scale, legend=None)
).mark_bar().properties(title='Female')

middle = base.encode(
    y=alt.Y('age:O', axis=None),
    text=alt.Text('age:Q'),
).mark_text().properties(width=20)

right = base.transform_filter(
    alt.datum.gender == 'Male'
).encode(
    y=alt.Y('age:O', axis=None),
    x=alt.X('sum(people):Q', title='population'),
    color=alt.Color('gender:N', scale=color_scale, legend=None)
).mark_bar().properties(title='Male')

spec = alt.concat(left, middle, right, spacing=5)

# Create altair chart widget
altair_chart = app.altair_chart(title='US Population Pyramid Over Time', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()