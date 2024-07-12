# Copyright (c) 2024 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets.apps as sa
import altair as alt


# Instantiate DataApp
app = sa.dataApp()
app.title("linechart_from_pandas_df")

# Create pd.DataFrame from list
data_dict = {"x":list(range(3)),
             "y":[420, 380, 390]}

data = app.sandbox.from_values('data', data_dict)

# Create line chart widget and plot df
alt_spec = alt.Chart(data.execute().to_pandas()).mark_line().encode(x='x', y='y')

app.simplechart(spec=(alt_spec).to_json(format="vega"), type="Altair")

