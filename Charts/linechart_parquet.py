# Copyright (c) 2023 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import os
import shapelets as sh
import requests

# Instantiate DataApp
app = DataApp("Linechart from parquet file", version=(1,0))

# Download sample parquet file from S3
filename = "sample_data.parquet"
if not os.path.exists(filename):
    print("Downloading...")
    data = requests.get("https://ursa-labs-taxi-data.s3.amazonaws.com/2009/01/data.parquet")
    with open(filename, "wb") as file:
        file.write(data.content)

# Load parquet file
session = sh.sandbox()
dataset = session.from_parquet(filename).limit(1000)
dataset = dataset.rename_columns({'pickup_at':'index'})

# Choose only useful columns
subset = session.map((col.index,
    col.passenger_count, 
    col.trip_distance, 
    col.fare_amount, 
    col.tip_amount,
    col.tolls_amount,
    col.total_amount) for col in dataset)

# Create line chart widget and plot df
lc = app.line_chart(data=subset, title="NYC Taxi data")

# Place line chart widget
app.place(lc)

# Login and register data app
sh.login(user_name="admin",password="admin")
app.register()