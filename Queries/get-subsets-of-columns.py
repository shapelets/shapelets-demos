# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets as sh

session = sh.sandbox()

data = session.load_test_data()

data = data.add_column("Sepal_Area", lambda row: row.Sepal_Length*row.Sepal_Width)

data_subset = data.select_columns(['Sepal_Area','Sepal_Length','Sepal_Width'])

# or 
data_subset = data.select_columns([0,1,2])

print(data_subset.head())
