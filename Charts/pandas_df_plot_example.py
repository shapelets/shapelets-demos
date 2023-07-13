# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets as sh
sh.login(user_name='admin',password='admin'

from shapelets.apps import DataApp
import matplotlib.pyplot as plt
import pandas as pd

# Create a dataApp
app = DataApp("pandas_df_plot_example")

# Create sample pd.DataFrame from list
data = {"series1":[420, 380, 390]}
df = pd.DataFrame(data=data, index=range(len(data["series1"])))

# Plot using pd.DataFrame plot()
df.plot()

# Get the current figure
fig = plt.gcf()

# Create image from matplotlib figure
img = app.image(img=fig, caption="Pandas df plot image")

# Place image into the data app
app.place(img)

# Register data app
app.register()
