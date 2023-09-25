# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.apps import DataApp
import pandas as pd

app = DataApp(name="table", description="table")

# Table as pd.DataFrame
rows = [[5, 5, 2, 3], [8, 8, 5, 6]]
colNames = ['key', 'name', 'age', 'address']
data = pd.DataFrame(data=rows, columns=colNames)
table1 = app.table(data=data, widget_id="Table 2", tools_visible=False)
app.place(table1)

app.register()
