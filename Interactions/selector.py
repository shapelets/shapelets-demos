# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp

# Create function that returns the string to be shown to the user
def change_label(option_selected: str) -> str:
    return "Selected option " + option_selected

# Create data app
app = DataApp(name="selector")

# Create selector given title and options
selector = app.selector(title="My Selector", options=['a','b','c'])

# Place selector in data app
app.place(selector)

# Create text label
label = app.text("No option selected yet")

# Place label in data app
app.place(label)

# Bind label to selector
# When selector changes, change_label() will be called with the value of selector as its parameter.
label.bind(change_label, selector)

# Register data app
app.register()



