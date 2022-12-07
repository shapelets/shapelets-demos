# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp, Checkbox

# Create a dataApp
app = DataApp(name="checkbox")

def change_text(value:Checkbox)->str:
    if value is True:
        return "The checkbox is activated"
    else:
         return "The checkbox is NOT activated"   

# Create a checkbox widget
control = app.checkbox(title='Option', toggle=True)
app.place(control)

# Create a text widget
txt = app.text("This is an empty text")
app.place(txt)

txt.bind(change_text, control)

app.register()
