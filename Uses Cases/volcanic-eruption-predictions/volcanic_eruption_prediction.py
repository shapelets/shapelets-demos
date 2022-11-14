# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

# Shapelets imports
from shapelets.apps import DataApp
from shapelets.model import Dataframe, Capsule, Image

# Other imports
from datetime import datetime
import pandas as pd
import pickle
import requests
from io import StringIO

def encode_multipart_formdata(fields):
    # This function takes the data fields in a form and builds the body and content type for a multipart HTTP request
    boundary = '------WebKitFormBoundary'

    body = (
            "".join("--%s\r\n"
                    "Content-Disposition: form-data; name=\"%s\"\r\n"
                    "\r\n"
                    "%s\r\n" % (boundary, field, value)
                    for field, value in fields.items()) +
            "--%s--\r\n" % boundary
    )

    content_type = "multipart/form-data; boundary=%s" % boundary

    return body, content_type

def get_earthquakes_IGN(lat_min, lat_max, long_min, long_max, start_date, end_date):
    # This function takes lat., long. and date ranges and returns a pandas dataframe with all the earthquakes recorded
    url = f'https://www.ign.es/web/ign/portal/vlc-catalogo?p_p_id=IGNVLCCatalogoTerremotos_WAR_IGNVLCCatalogoTerremotosportlet&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_cacheability=cacheLevelPage&p_p_col_id=column-1&p_p_col_count=1&_IGNVLCCatalogoTerremotos_WAR_IGNVLCCatalogoTerremotosportlet_jspPage=/jsp/terremoto.jsp'
    payload = {'_IGNVLCCatalogoTerremotos_WAR_IGNVLCCatalogoTerremotosportlet_latMin': lat_min,
               '_IGNVLCCatalogoTerremotos_WAR_IGNVLCCatalogoTerremotosportlet_latMax': lat_max,
               '_IGNVLCCatalogoTerremotos_WAR_IGNVLCCatalogoTerremotosportlet_longMin': long_min,
               '_IGNVLCCatalogoTerremotos_WAR_IGNVLCCatalogoTerremotosportlet_longMax': long_max,
               '_IGNVLCCatalogoTerremotos_WAR_IGNVLCCatalogoTerremotosportlet_startDate': start_date,
               '_IGNVLCCatalogoTerremotos_WAR_IGNVLCCatalogoTerremotosportlet_endDate': end_date,
               '_IGNVLCCatalogoTerremotos_WAR_IGNVLCCatalogoTerremotosportlet_tipoDescarga': 'txt'}
    # Build the body and content type of the request
    body, content_type = encode_multipart_formdata(payload)
    # Execute the POST request and obtain the response
    response = requests.request('POST', url, data=body, headers={'Content-Type': content_type})
    # The response is returned as a txt file, which is decoded and converted into a pandas dataframe
    df = pd.read_fwf(StringIO(response.content.decode('utf-8')))
    # Convert the field Fecha into a datetime series
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
    # We build a key that will be shown to users when they choose an earthquake
    df['Key'] = df['Fecha'].astype(str) + ' ' \
                + df['Hora'].astype(str) + ' ' + \
                ' Localization: ' + df['Localización'].astype(str) + ' ' + \
                ' Magnitude: ' + df['Mag.'].astype(str).replace('nan', 'Not available')
    return df

def build_map(eqks:Dataframe,start_date:float,end_date:float)->Image:
    # This function takes a Dataframe containing all earthquakes, filters them and plots them into a plotly map
    from datetime import datetime
    import plotly.express as px
    from matplotlib import pyplot as plt
    from io import BytesIO
    # Convert the Shapelets dataframe into a pandas dataframe
    eqks_pd = pd.DataFrame(eqks)

    # Extract the earthquakes within the dates provided by the user
    eqks_pd_in_range = eqks_pd[(eqks_pd['Fecha'] >= start_date) & (eqks_pd['Fecha'] <= end_date)]
    # Build the map visualization using plotly express
    fig = px.scatter_mapbox(eqks_pd_in_range,
                            lat="Latitud",
                            lon="Longitud",
                            color_discrete_sequence=["brown"],
                            zoom=9.4)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=dict(
                lat=28.649212,
                lon=-17.856101
            ),
            pitch=0,
            zoom=9.4
        ),
    )

    # Use a BytesIO object to convert the map into an image, thus avoiding to use disk storage
    bio = BytesIO()
    fig.write_image(bio)
    bio.seek(0)
    img = plt.imread(bio)

    # Create a matplotlib figure and display the image in the figure
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Return an image based on the content of the figure
    return Image(fig)

def compute_probability(all_eqks:Dataframe, key:str, model:Capsule)->str:
    # This function takes a Dataframe containing all earthquakes and the key of the desired earthquake and
    # returns the probability of eruption
    import numpy as np
    # Convert the dataframe into a pandas dataframe
    all_eqks_pd = pd.DataFrame(all_eqks.dataframe)
    # Extract the year of the earthquake
    fecha = all_eqks_pd.loc[all_eqks_pd['Key'] == key, 'Fecha'].dt.year.values[0]
    # Extract the event id of the earthquake
    evento = all_eqks_pd.loc[all_eqks_pd['Key'] == key, 'Evento'].values[0]
    # Create the url pointing to the dat file containing the data of the chosen earthquake
    url = 'https://www.ign.es/web/resources/sismologia/www/dir_images_terremotos/fases/' + str(fecha) + '/' + str(evento) + '.dat'
    try:
        # Read the earthquake data file
        eqk = pd.read_fwf(url, skiprows=13)
        # Build the features required for prediction
        eqk['Amp_norm'] = eqk['Amp'] / eqk['Dist']
        feat1 = eqk['Amp'].min(skipna=True)
        feat2 = eqk['Amp'].skew(skipna=True)
        feat3 = eqk['Amp'].kurtosis(skipna=True)
        feat4 = eqk['Amp_norm'].std(skipna=True)
        feat5 = eqk['Amp_norm'].kurtosis(skipna=True)
        # Min-max scale the features
        feats_min = np.array([[0.3, -1.96022233, -5.84746309, 58.39449395, -5.99151427]])
        feats_max = np.array([[4.64849000e+04, 3.15408415e+00, 9.96200521e+00, 1.16938830e+08, 1.49807556e+01]])
        X = np.array([[feat1, feat2, feat3, feat4, feat5]])
        X = (X - feats_min) / (feats_max - feats_min)
        # Run inference on the model and obtain the probability scores
        prob = model.data.predict_proba(X)
        return "\nThe probability of eruption is {:0.2f}%.\n".format(100 * prob[0][1])
    except:
        return "Could not retrieve phase data for this earthquake. Try with a posterior one."


# Create the data app
app = DataApp(name="volcanic_eruption_prediction",
              description="Use case for predicting volcanic eruptions using data from earthquakes")



# Create and place a markdown
md = app.text("""
# Predicting the time to eruption of a volcano
""",markdown=True)
app.place(md)

# Create and place an image
img=app.image('la_palma.webp')
app.place(img)

# Create and place a markdown
md2 = app.text("""
## Introduction
Earthquakes and volcanoes are natural phenomena arising from plate tectonics. Volcanic eruptions are generally accompanied by earthquakes.\n
Earthquakes refer to the trembling or shaking of the Earth’s crust as a result of an abrupt release of energy in the form of seismic waves. These waves are mainly generated by natural phenomena but can also be caused by man-made events. Volcanoes, on the other hand, are openings in the Earth’s crust from which hot gases and molten rock materials are ejected onto the surface of the Earth.

Earthquakes and volcanic activity are closely related to each other. In fact, volcanic eruption are usually accompanied by earthquakes. Similarly, unusual earthquakes can also lead to volcanic eruptions. Before discussing the relationship between earthquakes and volcanoes, let’s learn a bit more about each of them.\n

## Earthquakes
Earthquakes are caused due to a sudden release of accumulated pressure. The generated seismic waves can be measured with the help of a seismometer, indicating the intensity or size of the earthquake.
The size of an earthquake is represented by the moment magnitude scale (MMS); a magnitude of 3 or lower is undetectable, whereas a magnitude equal to or greater than 7 causes maximum damage to life and property. The underground point where this process originates is called the hypocenter or focus. Epicenter refers to the point on the Earth’s surface, which is exactly above the hypocenter.

In this data app, you can retrieve historical data for earthquakes in the island of La Palma (Spain). The data is provided by the Spanish [**National Geographic Institute**](https://www.ign.es/).

## Please select a time period to search for earthquakes:
""",markdown=True)
app.place(md2)

# Download all earthquakes recorded to the date and store them into a pandas dataframe
eqks_pd = get_earthquakes_IGN(lat_min=28.3,
                            lat_max=28.9,
                            long_min=-18.1,
                            long_max=-17.5,
                            start_date=pd.Timestamp.min.strftime("%d/%m/%Y"),
                            end_date=datetime.now().strftime("%d/%m/%Y"))


# Convert the pandas dataframe into a Shapelets dataframe
#eqks_df = client.create_dataframe(eqks_pd, name='earthquakes', description='all earthquakes retrieved')

# create a date selector 
start_date = app.datetime_selector(title='Starting date', date_time=datetime(2019,1,1).timestamp())
end_date = app.datetime_selector(title='End date', date_time=datetime(2020,1,1).timestamp())

#start_date = app.datetime_selector(title='Starting date', date_time=datetime(2019,1,1).timestamp() )
#end_date = app.datetime_selector(title='End date', date_time=datetime(2020,1,1).timestamp())



# Create a button
button = app.button(text='Show earthquakes')
text1= app.text("Build a map using the earthquake data retrieved and the selected dates, and assign this action to the button")
# Build the map using the earthquake data retrieved and the selected dates, and assign this action to the button
map = build_map(eqks_pd, start_date, end_date)
text1.bind(button,trigger=map)

# Place date selectors and button
app.place(start_date)
app.place(end_date)
app.place(button)

# Create and place an image
img2=app.image(map)
app.place(img2)

# Create and place a markdown
md3 = app.text("""
## Volcanic Eruptions
Volcanic eruptions are geological processes that involve extrusion of magma. They usually form mountains or mountain-like landscapes after the ejected materials cool down. They can occur in any part of the Earth’s surface, either in land or seas and oceans. Volcanoes are classified into active (eruptive), dormant (presently not active), and extinct (not eruptive) types based on the activeness of a particular volcano. They are further classified into six different types:  submarine, subglacial, shield, cinder, stratovolcano, and supervolcano, depending upon the mode of ejection, among other features.

## How are earthquakes and volcanoes related?
The close relationship between earthquakes and volcanic outbursts is evident from the maps depicting the locations prone to both these phenomena. If you compare the maps that illustrate earthquake zones and volcanic zones, you will find them matching each other. This is because the main theory behind both types of events lies in the plate tectonics.
Planet Earth comprises plates with irregular shapes of different sizes, which constantly move at different speeds. To be precise, the plates drift over the mantle layer of the Earth. Consequently, magma is generated along the plate boundaries. When these plates collide, move apart, or slide with each other, this leads to generation and accumulation of pressure (strain), which when released causes earthquakes. The strongest earthquakes are manifested during the plate collision, while the slowest earthquakes are observed when plates move apart from each other.

Similar to earthquakes, volcanic activity is observed when the plates are divergent (move apart) or convergent (move towards each other). In such plate movements, the magma present in the plate boundaries may rise to the Earth’s surface, leading to volcanic eruptions. Divergent plates may cause long volcanic rifts, whereas convergent plates result in individual volcanic eruptions.\n

We have built a model that uses several features from earthquakes in order to predict volcanic eruptions. In particular, we rely the magnitude and distance measured from different monitoring stations and compute the following features:
1. Minimum amplitude measured
2. Skewness of all amplitudes measured
3. Kurtosis of all amplitudes measured
4. Standard deviation of all amplitudes divided by the distance
5. Kurtosis of all amplitudes divided by the distance

Our model achieves a classification precision of 67%, a recall of 52% and an F-score of 57% on unseen volcanic eruptions. Wanna see it in action?

Pick an earthquake in the list below and use our model to predict the probability of an eruption:\n

""",markdown=True)
app.place(md3)

# Create and place a selector which takes all earthquakes downloaded as options
selector = app.selector(title='Select an earthquake to compute its probability of eruption:',
                        options=eqks_pd['Key'].to_list())
app.place(selector)

# Create a button that will trigger the model inference
button2 = app.button(text="Compute probability of eruption")

# Load the model
model = Capsule(data=pickle.load(open('best_model_volcano_eruption.pkl', 'rb')), name='model')

# The probability of eruption is computed using all earthquakes to find the year and event id of the chosen earthquake
text =compute_probability(eqks_pd, selector, model)
text3 = app.text("The probability of eruption is computed using all earthquakes to find the year and event id of the chosen earthquake")
# Assign the action of computing the probability of eruption to the button, and place it
text3.bind(button2,trigger=text)
app.place(button2)

# Create and place a label with the result of the prediction in the form of a string
label = app.label(text)
app.place(label)

# Register the DataApp
app.register()
