# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import datetime

# Create a dataApp
app = DataApp(name="datetime_range_selector")

# Create sample dates using datetime
start_date = datetime.date(2021, 1, 17)
end_date = datetime.date(2022, 1, 28)
min_date = datetime.date(2018, 1, 17)
max_date = datetime.date(2023, 1, 28)

# Create selector based on datetime objects
dateSelector1 = app.datetime_range_selector("selector from datetime", 
    start_date_time=start_date, 
    end_date_time=end_date, 
    min_date=min_date, 
    max_date=max_date)

# Place selector in data app
app.place(dateSelector1)

# Create selector based on ISO date/time strings
dateSelector2 = app.datetime_range_selector("selector from string", 
    start_date_time="2019-02-15 08:15:00", 
    end_date_time="2020-01-10 18:32:55", 
    min_date="2015-02-15", 
    max_date="2025-01-10")

# Place selector in data app    
app.place(dateSelector2)

# Register data app
app.register()