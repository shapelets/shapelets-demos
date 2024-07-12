# Copyright (c) 2024 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets.apps as sa
import altair as alt
import requests 
import os

# Instantiate DataApp
app = sa.dataApp()
app.title("100k taxi rides")

# Download sample parquet file from S3
filename = "nyc_taxi.parquet"
if not os.path.exists(filename):
    print("Downloading...")
    data = requests.get("https://vegafusion-datasets.s3.amazonaws.com/datashader/nyc_taxi.parquet")
    with open(filename, "wb") as file:
        file.write(data.content)

source_data = app.sandbox.from_parquet('source',['nyc_taxi.parquet']).limit(100000).execute().to_pandas()

width = 400
height = 400

# Use for large datasets
alt.data_transformers.enable("vegafusion")

base = alt.Chart(source_data, width=width, height=height).transform_calculate(
    pickup_hour="utchours(datum.tpep_pickup_datetime)",
    pickup_day="day(datum.tpep_pickup_datetime)",
    tip_perc="datum.tip_amount",
).transform_filter(
    "datum.tip_perc < 100"
)

x_domain = [-8.243204e+06, -8.226511e+06]
y_domain = [4.968192e+06, 4.982886e+06]

width_bins = 180
height_bins = 180

scales = alt.selection_interval(
    name="pickup_scales",
    bind='scales',
    on="[mousedown[event.altKey], window:mouseup] > window:mousemove![event.altKey]",
    translate="[mousedown[event.altKey], window:mouseup] > window:mousemove![event.altKey]",
    zoom="wheel![event.altKey]"
)

pickup_selection = alt.selection_interval(
    on="[mousedown[!event.altKey], window:mouseup] > window:mousemove![!event.altKey]",
    translate="[mousedown[!event.altKey], window:mouseup] > window:mousemove![!event.altKey]",
    zoom="wheel![!event.altKey]",
)

distance_selection = alt.selection_interval(encodings=["x"])
day_hour_selection = alt.selection_interval()


# Distance
distance_chart = base.mark_bar().encode(
    x=alt.X("trip_distance:Q", bin=alt.Bin(maxbins=20, extent=[0, 12])),
    y="count()"
).add_params(
    distance_selection
).transform_filter(
    {"and": [
        pickup_selection,
        day_hour_selection
    ]}
)

# Pickup
pickup_chart = base.mark_rect().encode(
    x=alt.X(
        "pickup_x:Q",
        bin=alt.Bin(maxbins=width_bins),
        scale=alt.Scale(domain=x_domain),
        axis=alt.Axis(labels=False)
    ),
    y=alt.Y(
        "pickup_y:Q",
        bin=alt.Bin(maxbins=height_bins),
        scale=alt.Scale(domain=y_domain),
        axis=alt.Axis(labels=False),
    ),
    opacity=alt.Opacity("count():Q", scale=alt.Scale(type="log", range=[0.5, 1.0]), legend=None),
    color=alt.Color("count():Q", scale=alt.Scale(type="log", scheme="purpleblue", reverse=False), legend=None)
).transform_filter(
    {"and": [
        scales,
        distance_selection,
        day_hour_selection
    ]}
).add_params(
    scales
).add_params(
    pickup_selection
).interactive()

# Tip percentage
tip_perc_chart = base.mark_rect().encode(
    x=alt.X(
        "pickup_day:O", 
        scale=alt.Scale(domain=[1, 2, 3, 4, 5, 6, 0]),
        axis=alt.Axis(labelExpr="datum.label==1 ? 'Mon': datum.label==0? 'Sun': ''")
    ),
    y=alt.Y("pickup_hour:O"),
    color=alt.Color(
        'mean(tip_perc):Q', 
        scale=alt.Scale(type="linear"),
        legend=alt.Legend(title="Tip Ratio")
    ),
    opacity=alt.condition(day_hour_selection, alt.value(1.0), alt.value(0.3))
).properties(
    width=120
).add_params(
    day_hour_selection
).transform_filter(
    {"and": [
        pickup_selection,
        distance_selection
    ]}
)

layout = (distance_chart | pickup_chart | tip_perc_chart).resolve_scale(
    color="independent"
)

app.simplechart(spec=layout.to_json(format="vega"), type="Altair")