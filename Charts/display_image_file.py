# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
from pathlib import Path

# Create a data app
app = DataApp("display_image_file")

# path to the image
currentDirectory = Path(__file__).parent.resolve()
resDirectory = currentDirectory.joinpath("../resources")
img_path = resDirectory / "hello.jpg"

# Create an image widget
img = app.image(img=img_path, caption="Test image")

# Place image into the data app
app.place(img)

# Register data app
app.register()
