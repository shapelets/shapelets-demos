# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets as sh
sh.login(user_name='admin',password='admin')

import datetime
import random

from shapelets.apps import DataApp

# Create function that returns a string with current datetime
def duration_test() -> str:
    return "Timer called at " + str(datetime.datetime.now())

# Create data app
app = DataApp(name="timer")

# Create a timer that gets updated every second
timer = app.timer(title="Timer", every=1.0, times=10)

# Place timer
app.place(timer)

# Create and place text label
label = app.text()
app.place(label)
label.bind(duration_test, triggers=[timer])

app.register()

