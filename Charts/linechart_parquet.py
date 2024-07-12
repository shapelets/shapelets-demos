# Copyright (c) 2024 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets.apps as sa
import os
import requests

# Instantiate DataApp
app = sa.dataApp()
app.title("Linechart from parquet file")

# Download sample parquet file from S3
filename = "sample_data.parquet"
if not os.path.exists(filename):
    print("Downloading...")
    data = requests.get("https://ursa-labs-taxi-data.s3.amazonaws.com/2009/01/data.parquet")
    with open(filename, "wb") as file:
        file.write(data.content)

# Load parquet file
full_data = app.sandbox.from_parquet('full_data',[filename]).with_columns(['pickup_at', 'passenger_count'])
dataset = full_data.limit(1000).with_name('dataset')

# Create line chart widget and plot df
app.mosaic(
    sa.vg.plot(
        sa.vg.marks.lineY(dataset, x='pickup_at', y='passenger_count', stroke='steelblue')
    )
)
