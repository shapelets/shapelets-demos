# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets as sh
sh.login(user_name='admin',password='admin')

from shapelets.apps import DataApp
import datetime

# Create a dataApp
app = DataApp(name="date_selector")

# Create a datetime_selector from current datetime
dateSelector = app.datetime_selector("Datetime now", 
    datetime.datetime.now(), '2009-10-10', '2029-10-18')
app.place(dateSelector)

# Create a datetime_selector for date only from a datetime object
date = datetime.date(2022, 10, 17)
date1 = datetime.date(2009, 10, 8)
date2 = datetime.date(2029, 10, 27)
dateSelector = app.datetime_selector("Date only", date, date1, date2)
app.place(dateSelector)

# Create a datetime_selector for date and time from a datetime object
date_string = '2021-09-01 15:27:05.004573 +0530'
datetime_obj = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S.%f %z')
dateSelector = app.datetime_selector("Datetime TZ", datetime_obj, date1, date2)
app.place(dateSelector)

# Register data app
app.register()