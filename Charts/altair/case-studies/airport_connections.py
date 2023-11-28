# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import altair as alt
from vega_datasets import data

# Instantiate data app
app = DataApp("Connections Among U.S. Airports Interactive", version=(0,1))

text = app.text("""
This example shows all the connections between major U.S. airports. Lookup transformations 
are used to find the coordinates of each airport and connecting airports. Connections 
are displayed on mouseover via a single selection.
""")
app.place(text)

# Since these data are each more than 5,000 rows we'll import from the URLs
airports = data.airports.url
flights_airport = data.flights_airport.url

states = alt.topo_feature(data.us_10m.url, feature="states")

# Create mouseover selection
select_city = alt.selection_single(
    on="mouseover", nearest=True, fields=["origin"], empty="none"
)

# Define which attributes to lookup from airports.csv
lookup_data = alt.LookupData(
    airports, key="iata", fields=["state", "latitude", "longitude"]
)

background = alt.Chart(states).mark_geoshape(
    fill="lightgray", 
    stroke="white"
).properties(
    width=750, 
    height=500
).project("albersUsa")

connections = alt.Chart(flights_airport).mark_rule(opacity=0.35).encode(
    latitude="latitude:Q",
    longitude="longitude:Q",
    latitude2="lat2:Q",
    longitude2="lon2:Q"
).transform_lookup(
    lookup="origin", 
    from_=lookup_data
).transform_lookup(
    lookup="destination", 
    from_=lookup_data, 
    as_=["state", "lat2", "lon2"]
).transform_filter(
    select_city
)

points = alt.Chart(flights_airport).mark_circle().encode(
    latitude="latitude:Q",
    longitude="longitude:Q",
    size=alt.Size("routes:Q", scale=alt.Scale(range=[0, 1000]), legend=None),
    order=alt.Order("routes:Q", sort="descending"),
    tooltip=["origin:N", "routes:Q"]
).transform_aggregate(
    routes="count()", 
    groupby=["origin"]
).transform_lookup(
    lookup="origin", 
    from_=lookup_data
).transform_filter(
    (alt.datum.state != "PR") & (alt.datum.state != "VI")
).add_selection(
    select_city
)

spec = (background + connections + points).configure_view(stroke=None)

# Create altair chart widget
altair_chart = app.altair_chart(title='Connections Among U.S. Airports Interactive', chart=spec)

# Place widget
app.place(altair_chart)

# Register data app
app.register()