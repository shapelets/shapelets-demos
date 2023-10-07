import shapelets as sh
from shapelets.apps.data_app import DataApp
from shapelets.apps.widgets import LineChart
import pandas as pd

sb = sh.sandbox()
data = sb.from_parquet('water_data.parquet')
data = data.rename_columns({'measurement_timestamp':'index'})

app = DataApp(name="Weather data stations", 
              description="Dataapp showing weather data for the City of Chicago", 
              version=(1,0))

tabs_fp = app.tabs_layout("Water data weather stations Chicago")
app.place(tabs_fp)
tab1 = app.vertical_layout()
tabs_fp.add_tab("Oak Street", tab1)
tab2 = app.vertical_layout()
tabs_fp.add_tab("63rd Street", tab2)
tab3 = app.vertical_layout()
tabs_fp.add_tab("63rd Street - Selector", tab3)
tab4 = app.vertical_layout()
tabs_fp.add_tab("63rd Street - Multi-line chart", tab4)

col_oak_street = sb.map(x for x in data if x.station_name == "\"Oak Street Weather Station")
col_63rd_street = sb.map(x for x in data if x.station_name == "\"63rd Street Weather Station")

sequences_wind = col_63rd_street.select_columns([x for x in col_63rd_street.columns if "wind_speed" in x])
sequences_temp = col_63rd_street.select_columns([x for x in col_63rd_street.columns if "temperature" in x])
sequences_rain = col_63rd_street.select_columns([x for x in col_63rd_street.columns if "rain" in x])
sequences_humidity = col_63rd_street.select_columns([x for x in col_63rd_street.columns if "humidity" in x])
sequences_pressure = col_63rd_street.select_columns([x for x in col_63rd_street.columns if "pressure" in x])

lc = app.line_chart(col_oak_street)
tab1.place(lc)

lc2 = app.line_chart(col_63rd_street)
tab2.place(lc2)

def get_series(dataset, selector)->LineChart:
    df = dataset[selector[1]]
    if df.dtype=='datetime64[ns]':
        return LineChart(title="Selected sequence")
    else:
        return LineChart(title="Selected sequence", data=df)

selector = app.selector(title="Select a sequence:",options=col_63rd_street.columns)
line_chart = app.line_chart(title="Selected sequence")
line_chart.bind(get_series,col_63rd_street.to_pandas(),selector,triggers=[selector])
tab3.place(selector)
tab3.place(line_chart)

lc1 = app.line_chart(title="Sequences wind", data=sequences_wind, multi_lane=False)
lc2 = app.line_chart(title="Sequences temperature", data=sequences_temp, multi_lane=False)
lc3 = app.line_chart(title="Sequences rain", data=sequences_rain, multi_lane=False)
lc4 = app.line_chart(title="Sequences humidity", data=sequences_humidity, multi_lane=False)
lc5 = app.line_chart(title="Sequences pressure", data=sequences_pressure, multi_lane=False)
tab4.place(lc1)
tab4.place(lc2)
tab4.place(lc3)
tab4.place(lc4)
tab4.place(lc5)

app.register()