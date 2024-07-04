# Copyright (c) 2024 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import folium
import shapelets.apps as sa

# Create a data app
app = sa.dataApp()

m = folium.Map(location=[45.5236, -122.6750])

folium_chart = app.simplechart(title='Folium Map', 
                              type='Folium',
                              cssStyles={
                                'width': '100%',
                                'height': '600px',
                                'color': 'blue',
                                'fontSize': '20px',
                                'border': '0px solid black'
                            }, 
                            spec=m._repr_html_())

if __name__ == '__main__':
    sa.serve(app, __file__)