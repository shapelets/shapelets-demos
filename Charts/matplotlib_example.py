# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import matplotlib.pyplot as plt

# Create a dataApp
app = DataApp("matplotlib_example")

# Create sample data
x = [1, 2, 3, 4, 5, 6]
y = [4, 3, 5, 6, 7, 4]

# Create a Matplotlib Figure
fig = plt.figure()
subplot1 = fig.add_subplot(2, 1, 1)
subplot1.plot(x, y)
subplot2 = fig.add_subplot(2, 1, 2)
subplot2.text(0.3, 0.5, '2nd Subplot')
fig.suptitle("This is a title")

# Create image from matplotlib figure
img = app.image(img=fig, caption="Matplotlib image")

# Place image into the data app
app.place(img)

# Register data app
app.register()
