# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
import pandas as pd 

# Instantiate data app
app = DataApp("Cumulative Wikipedia Donations", version=(0,1))

text = app.text("""
This chart shows cumulative donations to Wikipedia over the past 10 years.
""")
app.place(text)

data = pd.read_csv("https://frdata.wikimedia.org/donationdata-vs-day.csv", nrows=5000)

spec = alt.Chart(data).mark_line().encode(
    alt.X('monthdate(date):T', title='Month', axis=alt.Axis(format='%B')),
    alt.Y('max(ytdsum):Q', title='Cumulative Donations', stack=None),
    alt.Color('year(date):O', legend=alt.Legend(title='Year')),
    alt.Order('year(data):O')
)

# Create altair chart widget
altair_chart = app.altair_chart(title='Cumulative Wikipedia Donations', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()