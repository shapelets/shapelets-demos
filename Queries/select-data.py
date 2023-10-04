# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets as sh

session = sh.sandbox()

data = session.load_test_data()

data_selected = session.map(x for x in data)

data_filtered = session.map(x for x in data if x.Petal_Length > 1.5)

print(data_filtered)

#If you want an ascending order, pass a string with the column name to the function.
print(data.sort_by('Sepal_Length'))

#If you want a descending order, pass a string with the column name to the function.
print(data.sort_by('Sepal_Length',False))

#If you want to combine, or sort by multiple columns, just pass lists with the values to the function.
print(data.sort_by(['Sepal_Length','Petal_Length'],[False,True]))