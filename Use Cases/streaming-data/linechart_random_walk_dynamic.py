# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp, LineChart
import pandas as pd
import requests
import json

def update_linechart()->LineChart:
    r = requests.get(url = "http://127.0.0.1:8000")
    data =  json.loads(r.json()["data"])
    return app.line_chart(title="Random walk", data=pd.DataFrame(data), multi_lane=False)

# Instantiate DataApp
app = DataApp("Random walk linechart", version=(1,0))

t = app.timer(title='Receive data', every=0.5)

app.place(t)

# Create line chart widget and plot df
lc = app.line_chart(title="Random walk")
lc.bind(update_linechart, triggers=[t])
app.place(lc)

# Register data app
app.register()
