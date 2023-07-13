# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets as sh
sh.login(user_name='admin',password='admin')

import folium
from shapelets.apps import DataApp

# Create a data app
app = DataApp("folium_example")

m = folium.Map(location=[45.5236, -122.6750])

folium_chart = app.folium_chart(title='Folium Map', folium=m)
app.place(folium_chart)

app.register()
