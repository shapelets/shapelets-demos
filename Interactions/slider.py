# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp

# Create a data app
app = DataApp(name="slider")

# Create a slider
slider = app.slider(title="Default slider")

# Place slider into the data app
app.place(slider)

# Create a simple slider and place it into the data app
slider = app.slider(title="Slider with default value to 5", 
    min_value=0, max_value=10, step=1, value=5)
app.place(slider)

# Create a slider with a range and place it into the data app
sliderRange = app.slider(title="Slider with a range", 
    min_value=0, max_value=10.5, step=0.5, value=[1, 7.5])
app.place(sliderRange)

# Create a text slider and place it into the data app
sliderRange = app.slider(title="Slider with text options", 
    value='Red', 
    options=['Yellow', 'Orange', 'Red', 'Green', 'Blue'])
app.place(sliderRange)

# Create a text slider with a range and place it into the data app
sliderRange = app.slider(title="Slider range with text options", 
    value=['Orange', 'Red'], 
    options=['Yellow', 'Orange', 'Red', 'Green', 'Blue'])
app.place(sliderRange)

# Register the Dataapp
app.register()
