# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets as sh
sh.login(user_name='admin',password='admin')


from shapelets.apps import DataApp, Table
import pandas as pd
import numpy as np


def change_table1(p1: int, p2: int = 2) -> pd.DataFrame:
    ret = None
    rows1 = [[1, 2, 3], [4, 5, 6]]
    rows2 = [[5, 6, 7], [8, 9, 10]]
    colNames = ["Col1", "Col2", "Col3"]
    data1 = pd.DataFrame(data=rows1, columns=colNames)
    data2 = pd.DataFrame(data=rows2, columns=colNames)

    if (p1 > 0):
        ret = data1
    else:
        ret = data2

    return ret


def change_table2(p1: int) -> Table:

    ret = None
    if p1 > 0:
        ret = Table(rows=[[1, 2], [3, 4], [5, 6], [6, 7]])
    else:
        ret = Table(rows=[[0, 0], [1, 1], [2, 2], [3, 3]])

    return ret


def change_table3(num1: int, num2: int) -> np.ndarray:
    array = None
    if (num1 > num2):
        array = np.array([[1, 2], [3, 4], [5, 6], [6, 7]])
    else:
        array = np.array([[11, 22], [33, 44], [55, 66], [77, 88]])

    return array


def change_table4(num1: int) -> list:
    lista = []
    if (num1 > 0):
        lista = [[1, 2], [3, 4], [5, 6], [6, 7]]
    else:
        lista = [[11, 22], [33, 44], [55, 66], [77, 88]]

    return lista


app = DataApp(name="13_table", description="table")

# Table as pd.DataFrame
#rows = [[1, 1, 2, 3], [2, 4, 5, 6]]
#colNames = ['key', 'name', 'age', 'address']
#data = pd.DataFrame(data=rows, columns=colNames)
#table1 = app.table(data=data, widget_id="Table 1", tools_visible=True)
#app.place(table1)

# Table as pd.DataFrame
rows = [[5, 5, 2, 3], [8, 8, 5, 6]]
colNames = ['key', 'name', 'age', 'address']
data = pd.DataFrame(data=rows, columns=colNames)
table1 = app.table(data=data, widget_id="Table 2", tools_visible=False)
app.place(table1)

app.register()
