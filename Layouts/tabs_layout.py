# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets as sh
sh.login(user_name='admin',password='admin')

from shapelets.apps import DataApp

# Create data app
app = DataApp("tabs_layout")

# Create two vertical layouts
vf = app.vertical_layout()
vf2 = app.vertical_layout()

# Create a tabs layout
tabs_fp = app.tabs_layout("My tabs layout")

# Create two tabs and add a vertical layout in each of them 
tabs_fp.add_tab("Tab 1", vf)
tabs_fp.add_tab("Tab 2", vf2)

# Place markdown texts in each of the vertical layouts
vf.place(app.text("""
    # MD for tab 1
    This markdown is rendered into the first tab
""", markdown=True))

vf2.place(app.text("""
    # MD for tab 2
    This markdown is rendered into the second tab
""", markdown=True))

# Place the tabs layout in the data app
# This layout contains the tabs, each contaning a vertical layout
app.place(tabs_fp)

# Register the data app
app.register()