# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp

# Create function that returns the input value plus one
def change_label(num1: int) -> int:
    return num1+1

# Create a data app
app = DataApp("button")

# Create a button
button = app.button("Press me")
app.place(button)

# Create a number input
number = app.number_input(value=5)
app.place(number)

# Create a label
label = app.text()
app.place(label)

# label will be updated with the value returned by change_label()
# change_label() will be called every time its parameter changes (e.g. number) or when button is clicked
# mute=[number] prevents change_label() from being called when number changes
# If you want label to change also when number changes, remove mute=[number]
label.bind(change_label, number, mute=[number], triggers=[button])

# Register the Dataapp
app.register()