# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp

# Create a dataApp
app = DataApp("number_input")

# Create a number_input and set its parameters
num = app.number_input(title="Number input title", 
    placeholder='Enter a number here', 
    step=5, 
    units='%')

# Place number_input into the data app    
app.place(num)

# Register the data app
app.register()
