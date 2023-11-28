# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
import pandas as pd

# Instantiate data app
app = DataApp("Window Rank Line Chart", version=(0,1))

text = app.text("""
This example shows the Group F rankings in the 2018 World Cup after each matchday. 
A window transformation is used to rank each after each match day, sorting by points and difference.
""")
app.place(text)

source = pd.DataFrame(
    [
        {"team": "Germany", "matchday": 1, "point": 0, "diff": -1},
        {"team": "Germany", "matchday": 2, "point": 3, "diff": 0},
        {"team": "Germany", "matchday": 3, "point": 3, "diff": -2},
        {"team": "Mexico", "matchday": 1, "point": 3, "diff": 1},
        {"team": "Mexico", "matchday": 2, "point": 6, "diff": 2},
        {"team": "Mexico", "matchday": 3, "point": 6, "diff": -1},
        {"team": "South Korea", "matchday": 1, "point": 0, "diff": -1},
        {"team": "South Korea", "matchday": 2, "point": 0, "diff": -2},
        {"team": "South Korea", "matchday": 3, "point": 3, "diff": 0},
        {"team": "Sweden", "matchday": 1, "point": 3, "diff": 1},
        {"team": "Sweden", "matchday": 2, "point": 3, "diff": 0},
        {"team": "Sweden", "matchday": 3, "point": 6, "diff": 3},
    ]
)

color_scale = alt.Scale(
    domain=["Germany", "Mexico", "South Korea", "Sweden"],
    range=["#000000", "#127153", "#C91A3C", "#0C71AB"],
)

spec = alt.Chart(source).mark_line().encode(
    x="matchday:O", y="rank:O", color=alt.Color("team:N", scale=color_scale)
).transform_window(
    rank="rank()",
    sort=[
        alt.SortField("point", order="descending"),
        alt.SortField("diff", order="descending"),
    ],
    groupby=["matchday"],
).properties(title="World Cup 2018: Group F Rankings", height=200)

# Create altair chart widget
altair_chart = app.altair_chart(title='Window Rank Line Chart', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()
