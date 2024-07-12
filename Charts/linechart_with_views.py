# Copyright (c) 2024 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets.apps as sa
import altair as alt

# Instantiate DataApp
app = sa.dataApp()
app.title("Linechart with views")

# Load data from csv
dataset = app.sandbox.from_csv('dataset',['../Resources/mitdb102.csv'], has_header=True, skip_rows=1).limit(20000,200000).execute().to_pandas()

dataset['Start']=577
dataset['End']=577.5

# Use for large datasets
alt.data_transformers.enable("vegafusion")
ac = alt.Chart(dataset, width="container").mark_line().encode(
    x=alt.X('seconds', axis=alt.Axis(title="Seconds")) ,
    y=alt.X('mV_1', axis=alt.Axis(title="mV"))
).interactive(bind_y=False)

view = alt.Chart(dataset).mark_rect(color='#FF0000',opacity=0.005).encode(
    x=alt.X('Start'), 
    x2='End', 
    y=alt.value(0), 
    y2=alt.value(300))

app.simplechart(spec=(ac+view).to_json(format="vega"), type="Altair")