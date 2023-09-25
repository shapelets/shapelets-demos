# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp

# Create data app
app = DataApp("vertical_layout")

# Create vertical layout
vl = app.vertical_layout()

# Create buttons and place multiple text inputs in the vertical layout
txt1 = app.text_input("Text input #1")
vl.place(txt1)

txt2 = app.text_input("Text input #2")
vl.place(txt2)

txt3 = app.text_input("Text input #3")
vl.place(txt3)

# Place the vertical layout in the dataapp
app.place(vl)

# Register the data app
app.register()