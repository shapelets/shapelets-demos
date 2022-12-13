from shapelets.apps import DataApp
from shapelets.model import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

from texts import T1,T2,T3,T4,T5,T6,T7

# Create a dataApp
app = DataApp(
    name="Use Case REE",
    description="An example on how to predict the hourly price of electricity in the Spanish market",
    version=[1,0]
)

# Load data
df = pd.read_csv("data/data.csv")
df.datetime = pd.to_datetime(df.datetime)
df = df.set_index('datetime')

df_price = pd.read_csv("data/spot_esp_20150101_20220415.csv")
df_price = df_price[:-1]
df_price.timestamp = pd.to_datetime(df_price.timestamp)
df_price = df_price.set_index('timestamp')

df['price'] = df_price.SPOTPRICE_españa

df_real = pd.read_csv("data/data_treal.csv")
df_real.datetime = pd.to_datetime(df_real.datetime)
df_real = df_real.set_index('datetime')

app.place(app.text(T1, markdown=True))

aux = np.array([0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,2,4,5,5,5,5,6,7,7,7,7,7,7,7,8,8,8,8,8,8])
aux_2 = np.array([8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 3, 2, 2, 2,2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

fig, ax = plt.subplots(figsize=(15,7))
plt.plot(aux,label='Sellers')
plt.plot(aux_2,label='Buyers')

index_union = np.where(aux == aux_2)[0][0]

coords_circle = (index_union,aux[index_union])

circle2 = plt.Circle(coords_circle, 0.8, color='r', fill=False)
ax.add_patch(circle2)

plt.axvline(x = index_union, color = 'k', linestyle = '--')

plt.xlabel("Quantity [MWh]")
plt.ylabel("Price [€]")

plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)

plt.legend()

img = app.image(fig)
app.place(img)

app.place(app.text(T2, markdown=True))

# Create tabs layout
tabs_fp = app.tabs_layout("OMIE Data (Hour)")
app.place(tabs_fp)

lcolumns = ['price','GP_Nuclear','GP_Hidraulica','GP_Eolica','GP_Ciclo_Combinado','GP_Cogeneracion']
lcolsnames = ["Spot Price","Nuclear Energy","Hidraulic Energy","Eólica Energy","Combined Cycle Energy","Cogeneracion Energy"]

for index,column in enumerate(lcolumns): 
    vl_tab = app.vertical_layout(panel_id=index)
    tabs_fp.add_tab(lcolsnames[index],vl_tab)
    line_chart = app.line_chart(title=column, data=df[column])
    vl_tab.place(line_chart)


# Create tabs layout
tabs_fp = app.tabs_layout("OMIE Data (10Min)")
app.place(tabs_fp)

lcolumns = ['demanda_real','GTR_hidraulica','GTR_nuclear','GTR_ciclo_combinado',]
lcolsnames = ["Demanda real",'GTReal hidráulica',"GTReal nuclear","GTReal ciclo combinado",]

for index,column in enumerate(lcolumns): 
    vl_tab = app.vertical_layout(panel_id=index)
    tabs_fp.add_tab(lcolsnames[index],vl_tab)
    line_chart = app.line_chart(title=column, data=df_real[column])
    vl_tab.place(line_chart)

app.place(app.text(T3, markdown=True))

df_fil = df[['Spot_Price_Spain']].loc["2020-01-01":"2020-01-30"]
df_fil['color'] = df_fil.index.to_series().dt.day
df_fil = df_fil.reset_index()
df_fil['hora'] = df_fil.datetime.dt.time.astype(str)

alt.data_transformers.disable_max_rows()

graph = alt.Chart(df_fil).mark_line(strokeWidth=3).encode(
    x='hora',
    y='Spot_Price_Spain',
    color=alt.Color('color', legend=None)
)

app.place(app.altair_chart(spec = graph))

app.place(app.text(T4, markdown=True))

# RESAMPLE DATA
df_real_resampled = df_real.resample('H').agg(['min','mean','max'])

for col in df_real_resampled.columns:
    name = '_'.join(col)
    df[name] = df_real_resampled[col[0]][col[1]] 
    
# REMOVE OUTLIERS
    
aux_1 = df.loc['2022-03-04':'2022-03-06','price']
aux_2 = df.loc['2022-03-10':'2022-03-12','price']

aux = pd.DataFrame(pd.concat([aux_1,aux_2]))
input_values = aux.groupby(aux.index.to_series().dt.hour).max().values

df.loc['2022-03-07','price'] = input_values.reshape(-1)
df.loc['2022-03-08','price'] = input_values.reshape(-1)
df.loc['2022-03-09','price'] = input_values.reshape(-1)

# PROCESSING TARGET

daterange = pd.date_range(df.index.min(), df.index.max(),freq='D')

price_ts = []
for dia in daterange:
        data = df.loc[dia.strftime("%Y-%m-%d"),'price'].values.reshape(-1)
        if data.shape[0] != 24:
            print(dia)
            break
        price_ts.append(data)

price_ts_np = np.array(price_ts)

# Standard Scaler
sc = StandardScaler()
sc.fit(price_ts_np)
X_train_std = sc.transform(price_ts_np)

# PCA
pca = PCA(1)
data_fitted = pca.fit_transform(X_train_std)

df_pca = pd.DataFrame(data={'pca':data_fitted.reshape(-1)},index=daterange)

df_resample = df[['GP_Hidraulica', 'GP_Nuclear', 'GP_Ciclo_Combinado', 'GP_Eolica',
                   'GP_Cogeneracion', 'GTR_hidraulica_min',
                   'GTR_hidraulica_mean', 'GTR_hidraulica_max', 'GTR_nuclear_min',
                   'GTR_nuclear_mean', 'GTR_nuclear_max', 'GTR_ciclo_combinado_min',
                   'GTR_ciclo_combinado_mean', 'GTR_ciclo_combinado_max', 'GTR_eolica_min',
                   'GTR_eolica_mean', 'GTR_eolica_max', 'demanda_real_min',
                   'demanda_real_mean', 'demanda_real_max']]

df_min = df_resample.resample('D').min()
df_min.columns = [f'min_{x}' for x in df_min.columns]
df_mean = df_resample.resample('D').mean()
df_mean.columns = [f'mean_{x}' for x in df_mean.columns]
df_max = df_resample.resample('D').max()
df_max.columns = [f'max_{x}' for x in df_max.columns]

df_agg = pd.concat([df_min,df_mean,df_max],axis=1)

for columna in df_agg.columns:
    if 'gp48' not in columna.lower():
        nombre = f"{columna}_lag{1}"
        lag = 1
        df_agg[nombre] = df_agg[columna].shift(lag)
        df_agg.drop(columna,axis=1,inplace=True)
    elif 'gp48' in columna.lower():
        for i in range(1,3):
            nombre = f"{columna}_lag{i}"
            lag = i
            df_agg[nombre] = df_agg[columna].shift(lag)
    else:
        pass


df_mod2 = pd.concat([df_agg,df_pca],axis=1)
df_mod2['price_future'] = df_mod2.pca.shift(1)
df_mod2 = df_mod2.dropna()

# We add 1 as it goes from 0 to 6
df_mod2['dia_semana'] = pd.date_range(df_mod2.index.min(),df_mod2.index.max(),freq='H').to_series().dt.dayofweek + 1
df_mod2['ciclo_dia'] = df_mod2['dia_semana'] / 7
df_mod2['dias_mes'] = pd.date_range(df_mod2.index.min(),df_mod2.index.max(),freq='H').to_series().dt.daysinmonth
df_mod2['dia_mes'] = df_mod2.index.to_series().dt.day / df_mod2['dias_mes']
df_mod2['ciclo_mes'] = df_mod2['dia_mes'].apply(lambda x: round(x,3))
df_mod2 = df_mod2.drop(['dia_semana','dias_mes','dia_mes'],axis=1)

df_mod2 = df_mod2.drop(['pca'],axis=1)


df_mod2 = df_mod2.dropna()

train_data = df_mod2[:"2022-03-31"]
prediction_data = df_mod2["2022-04-01":]


y = train_data['price_future']
X = train_data.drop('price_future',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


app.place(app.text(T5, markdown=True))

def regression_metrics(y_true,y_pred):
    from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error, mean_squared_error
    errors = {"MSE": round(mean_squared_error(y_true,y_pred),4),
              "RMSE": round(mean_squared_error(y_true,y_pred,squared=False),4),
              "MAE": round(mean_absolute_error(y_true,y_pred),4),
              "MAPE": round(mean_absolute_percentage_error(y_true,y_pred),4)}
    return errors

regr = RandomForestRegressor(random_state=0,n_jobs=-1,**{'n_estimators': 400,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'max_features': 'sqrt',
                                                         'max_depth': None,
                                                         'bootstrap': False})

# Train model
regr.fit(X_train,y_train)
preds1 = regr.predict(X_test)
errores1 = regression_metrics(y_test,preds1)
errores1['model'] = "RandomForestRegressor"

model_ = lgb.LGBMRegressor(random_state=0,**{'n_estimators': 400,
                                             'min_samples_split': 2,
                                             'min_samples_leaf': 1,
                                             'max_features': 'sqrt',
                                             'max_depth': None,
                                             'bootstrap': False})

# Train model
model_.fit(X_train,y_train)
preds2 = model_.predict(X_test)
errores2 = regression_metrics(y_test,preds2)
errores2['model'] = "LightGBM"

model2_ = xgb.XGBRegressor(random_state=0,**{'n_estimators': 400,
                                             'min_samples_split': 2,
                                             'min_samples_leaf': 1,
                                             'max_features': 'sqrt',
                                             'max_depth': None,
                                             'bootstrap': False})


# Train model
model2_.fit(X_train,y_train)
preds3 = model2_.predict(X_test)
errores3 = regression_metrics(y_test,preds3)
errores3['model'] = "XGBoost"

df_errores = pd.DataFrame([errores1,errores2,errores3])

table = app.table(df_errores)
app.place(table)

app.place(app.text(T6, markdown=True))

predicts1 = regr.predict(prediction_data.drop('price_future',axis=1))
predicts2 = model_.predict(prediction_data.drop('price_future',axis=1))
predicts3 = model2_.predict(prediction_data.drop('price_future',axis=1))

prediction_data['preds_rfr'] = predicts1
prediction_data['preds_lgbm'] = predicts2
prediction_data['preds_xgboost'] = predicts3

dataset_plot = []

for i in pd.date_range("2022-04-01",periods=14,freq='D'):

    X_val_d = prediction_data.loc[i.strftime("%Y-%m-%d")]

    original_data = sc.inverse_transform(pca.inverse_transform(X_val_d['price_future'])).reshape(-1)
    preds_rfr     = sc.inverse_transform(pca.inverse_transform(X_val_d['preds_rfr'])).reshape(-1)
    preds_lgbm     = sc.inverse_transform(pca.inverse_transform(X_val_d['preds_lgbm'])).reshape(-1)
    preds_xgboost     = sc.inverse_transform(pca.inverse_transform(X_val_d['preds_xgboost'])).reshape(-1)

    new_index = pd.date_range(i.strftime("%Y-%m-%d"),periods=24,freq='H')
    new_df = pd.DataFrame({'real_data':original_data,
                           'prediction_rf':preds_rfr,
                           'prediction_lighgbm':preds_lgbm,
                           'prediction_xgboost':preds_xgboost},
                            index=new_index)

    dataset_plot.append(new_df)

dataset_plot = pd.concat(dataset_plot)

def plot_selected_day(dataf: pd.DataFrame, dia: int, model:str )-> Image:
    date = f"2022-04-{dia}"
    datafil =  dataf.loc[:,date]

    fig = plt.figure(figsize=(12,4)) 

    plt.title(f'Hourly prediction for day {dia}')
    plt.plot(datafil.loc[:, "real_data"],label="Real Price")

    if model[0] == "All Models":
        plt.plot(datafil.loc[:, "prediction_rf"],label="Prediction RandomForestRegressor")
        plt.plot(datafil.loc[:, "prediction_lighgbm"],label="Prediction LightGBM")
        plt.plot(datafil.loc[:, "prediction_xgboost"],label="Prediction XGBoost")
    elif model[0] == "RandomForestRegressor":
        plt.plot(datafil.loc[:, "prediction_rf"],label="Prediction RandomForestRegressor")
    elif model[0] == "LightGBM":
        plt.plot(datafil.loc[:, "prediction_lighgbm"],label="Prediction LightGBM")
    elif model[0] == "XGBoost":
        plt.plot(datafil.loc[:, "prediction_xgboost"],label="Prediction XGBoost")

    plt.legend()

    # return app.image(img=fig)
    return Image(fig)

# Create an horizontal_flow_panel
hf = app.horizontal_layout()

# Create a slider
slider = app.slider(title="Select a day", min_value=1, max_value=14, step=1, value=1)

# Place slider into the Dataapp
app.place(slider)
#hf.place(slider)#,width=9)

selector = app.selector(default=["All Models"], 
    options=["All Models", "RandomForestRegressor", "LightGBM", "XGBoost"])
app.place(selector)

button = app.button(text="Get predictions")

app.place(button)
#hf.place(button)#,width=3)

app.place(hf)

fig = plt.figure()
img_fig = app.image(fig)
app.place(img_fig)
img_fig.bind(plot_selected_day, dataset_plot, slider, selector, triggers=[button])

app.place(app.text(T7, markdown=True))

# Register the Dataapp
app.register()
