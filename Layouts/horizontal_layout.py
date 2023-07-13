# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets as sh
sh.login(user_name='admin',password='admin')

from shapelets.apps import DataApp

# Create data app
app = DataApp("horizontal_layout")

# Create vertical layout
hl = app.horizontal_layout()

# Create buttons and place multiple text inputs in the horizontal layout
txt1 = app.text_input("Text input #1")
hl.place(txt1)

txt2 = app.text_input("Text input #2")
hl.place(txt2)

txt3 = app.text_input("Text input #3")
hl.place(txt3)

# Place the horizontal layout in the dataapp
app.place(hl)

# Register the data app
app.register()