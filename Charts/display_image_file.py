# Copyright (c) 2024 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import dataApp

# Create a data app
app = dataApp()

# Create an image widget
img = app.image(src="../Resources/hello.jpg")