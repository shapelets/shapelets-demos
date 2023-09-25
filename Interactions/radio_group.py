# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp

def print_selected_option_1(radio:int)->str:
    return "Selected " + str(radio) + " in radiogroup 1"

def print_selected_option_2(radio:str)->str:
    return "Selected " + (radio) + " in radiogroup 2"

# Create a dataApp
app = DataApp("radiogroup")

# Radio group with number values
radiogroup1 = app.radio_group([1, 2, 3], value=2)
app.place(radiogroup1)

# Radio group with dict values, index_by, label_by and value_by property
radiogroup2 = app.radio_group(
    [{"id": 1, "label": "world", "value": "bar"}, 
    {"id": 2, "label": "moon", "value": "baz"}],
    label_by="label",
    value_by="value")
app.place(radiogroup2)

label = app.text("No selections done yet.")
label.bind(print_selected_option_1, radiogroup1)
label.bind(print_selected_option_2, radiogroup2)
app.place(label)

app.register()
