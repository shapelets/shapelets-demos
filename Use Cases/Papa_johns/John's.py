
# ## EDA: Create an an exploratory data analysis for the Miami population  
# The point point of this eda is to figure out, who are the population of miami and how it distribution influence the restaurant industry in the city. 

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import folium # mapping
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster
import statistics
from pandas import DataFrame 
from shapelets.apps import DataApp, BarChart, FoliumChart, Widget, Image
app = DataApp(name= "Papa Johns restaurants in Miami", description= "This app shows the locations aim to find the best location to open a restaurant in the Miami Area")

markdown_text = app.text(""" # Papa John's Pizza in Miami-Dade County, Florida
In this dataapp we will try to analyze the restaurant's in the Miami Area, especially the Pizza places and the Papa John's. We are aimimg to determine, which neighborhood is the most successful and which one is more likely to succeed. 
We will use the data from the US Census Bureau and the Foursquare API to determine the best neighborhood for a new Papa John's Pizza restaurant.""", markdown = True)
app.place(markdown_text)

# create a tab for the EDA and the machine learning part
# Create two vertical layouts
vf = app.vertical_layout()
vf2 = app.vertical_layout()

# Create a tabs layout
tabs_fp = app.tabs_layout("My Vertical layout")

# Create two tabs and add a vertical layout in each of them 
tabs_fp.add_tab("EDA", vf)
tabs_fp.add_tab("ML", vf2)

vf.place(app.text("""
    # EDA tab with folium map of restaurants in Miami
    In This Tab we will see the information of each feature in the Miami area, as well as the distribution of the restaurant on the Map 
""", markdown=True))

vf2.place(app.text("""
    # Machine learning tab
    In this tab we looked at the influence of the features on the Papa John's restaurant reviews, to predict which Postal Code would be more Likely to host a new restaurant. 
""", markdown=True))

# Place the tabs layout in the data app
# This layout contains the tabs, each contaning a vertical layout
app.place(tabs_fp)


# import the full data csv file
papa_person = pd.read_csv('/root/Papa_johns/papa_person1.csv')
papa_person_group = papa_person.groupby("postalCode").mean()

one = papa_person_group.iloc[:,84+10:]
two = papa_person_group.iloc[:,41+10:44+10]
population = papa_person_group.iloc[:,7+10:12+10]
languages = papa_person_group.iloc[:,12+10:17+10]
married = papa_person_group.iloc[:,17+10:22+10]
male_female = papa_person_group.iloc[:,4+10:6+10]
household = papa_person_group.iloc[:,22+10:29+10]
income = papa_person_group.iloc[:,29+10:41+10]
education = two.join(one)
employment = papa_person_group.iloc[:,45+10:66+10]
employement_group = papa_person_group.iloc[:,66+10:69+10]
age = papa_person_group.iloc[:,69+10:84+10]
leftover = papa_person_group.iloc[:,0+10:4+10]




def plot_by_postalCode(papa_person_group: DataFrame, selector: str)-> Image:
    selector = int(selector)

    # transfort inc into an image object
    fig, ax = plt.subplots(3, 4, figsize=(50, 40))
    languages[languages.index == selector].plot(kind='bar', ax=ax[0, 1])
    married[married.index == selector].plot(kind='bar', ax=ax[0, 2])
    male_female[male_female.index == selector].plot(kind='bar', ax=ax[0, 3])
    population[population.index == selector].plot(kind='bar', ax=ax[0, 0])
    household[household.index == selector].plot(kind='bar', ax=ax[1, 0])
    income[income.index == selector].plot(kind='bar', ax=ax[1, 1])
    education[education.index == selector].plot(kind='bar', ax=ax[1, 2])
    employment[employment.index == selector].plot(kind='bar', ax=ax[1, 3])
    employement_group[employement_group.index == selector].plot(kind='bar', ax=ax[2, 0])
    age[age.index == selector].plot(kind='bar', ax=ax[2, 1])
    leftover[leftover.index == selector].plot(kind='bar', ax=ax[2, 2])

    return app.image(img=fig)


def box_plot_by_postalCode(papa_person_group: DataFrame, selector: str)-> Image :
    selector = int(selector)

    fig, ax = plt.subplots(3, 4, figsize=(50, 5))
    income[income.index == selector].plot(kind='box', ax=ax[0,0])
    languages[languages.index == selector].plot(kind='box',ax=ax[0,1])
    married[married.index == selector].plot(kind='box',ax=ax[0,2])
    male_female[male_female.index == selector].plot(kind='box',ax=ax[0,3])
    population[population.index == selector].plot(kind='box',ax=ax[1,0])
    education[education.index == selector].plot(kind='box', ax=ax[1,1])
    employment[employment.index == selector].plot(kind='box',ax=ax[1,2]) 
    employement_group[employement_group.index == selector].plot(kind='box', ax=ax[1,3])
    age[age.index == selector].plot(kind='box', ax=ax[2,0])
    household[household.index == selector].plot(kind='box', ax=ax[2,1])
    leftover[leftover.index == selector].plot(kind='box', ax=ax[2,2])
        

    # convert img to an image object
    return app.image(img=fig)

def hist_plot_by_postalCode(papa_person_group: DataFrame, selector: str)-> Image :
    selector = int(selector)

    fig, ax = plt.subplots(3, 4, figsize=(50, 40))
    income[income.index == selector].plot(kind='hist', ax=ax[0,0])
    languages[languages.index == selector].plot(kind='hist', ax=ax[0,1] )
    married[married.index == selector].plot(kind='hist', ax= ax[0,2] )
    male_female[male_female.index == selector].plot(kind='hist',ax= ax[0,3] )
    population[population.index == selector].plot(kind='hist', ax= ax[1,0] )
    education[education.index == selector].plot(kind='hist', ax= ax[1,1])
    employment[employment.index == selector].plot(kind='hist', ax= ax[1,2])
    employement_group[employement_group.index == selector].plot(kind='hist', ax= ax[1,3])
    age[age.index == selector].plot(kind='hist', ax= ax[2,0])
    household[household.index == selector].plot(kind='hist', ax= ax[2,1])
    leftover[leftover.index == selector].plot(kind='hist', ax= ax[2,2])
        

    # convert img to an image object
    return app.image(img=fig)

def folium_map(papa_person: DataFrame, selector: str)-> FoliumChart :
    selector = int(selector)
    # Folium map
    # create a map of Miami
    l=folium.Map(location = [25.761681
    ,-80.191788], #Initiate map on Miami city
    zoom_start = 10,min_zoom = 10)

    marker_cluster = MarkerCluster().add_to(l)

    papa_person = papa_person[papa_person.postalCode == selector]

    #select other than pizza restaurant on data frame
    not_pizza_place=papa_person[papa_person["pizza"] == False]


    #select pizza that are not Papa John´s
    not_papaj_place=papa_person[papa_person["pizza"] == True]
    not_papaj_place=not_papaj_place[not_papaj_place["Papa John's Pizza"] == False]

    #select papa johns pizza restaurant on data frame
    pizza_johns=papa_person[papa_person["Papa John's Pizza"] >0]

    #plot the map with all the locations
    for index,place in not_pizza_place.iterrows():
        folium.Marker([place["latitude"], place["longitude"]], popup=(place["name"],place["address"],place.index)).add_to(marker_cluster)

    #plot the map with all the location
    for index, place in not_papaj_place.iterrows():
        folium.Marker([place["latitude"], place["longitude"]],icon=folium.Icon(color='green', icon='ok-sign'), popup=(place["name"],place["address"],place.index)).add_to(marker_cluster)

    #plot the map with all the location
    for index, place in pizza_johns.iterrows():
        folium.Marker([place["latitude"], place["longitude"]],icon=folium.Icon(color='red', icon='ok-sign'), popup=(place["name"],place["address"],place.index)).add_to(marker_cluster)

    
    # plot the map with vectors between the restaurant and the minimum distance restaurant 
    for index, place in not_pizza_place.iterrows():
        folium.PolyLine(locations=[[place["latitude"], place["longitude"]],[place["latitude_min_restaurant"], place["longitude_min_restaurant"]]],color="black", weight=2.5, opacity=1).add_to(l)

    # plot the map with vectors between the restaurant and the minimum distance restaurant
    for index, place in not_papaj_place.iterrows():
        folium.PolyLine(locations=[[place["latitude"], place["longitude"]],[place["latitude_min_restaurant"], place["longitude_min_restaurant"]]],color="green", weight=2.5, opacity=1).add_to(l)

    # plot the map with vectors between the restaurant and the minimum distance restaurant
    for index, place in pizza_johns.iterrows():
        folium.PolyLine(locations=[[place["latitude"], place["longitude"]],[place["latitude_min_restaurant"], place["longitude_min_restaurant"]]],color="red", weight=2.5, opacity=1).add_to(l)

    return app.folium_chart(title= "Map of Miami's Restaurants", folium=l)
    

# create the selector for the postalCodes
selector = app.selector(title="Select the Postal Code", options=['33122','33125','33126','33127','33128','33129','33130','33131','33132','33133','33134','33135','33136','33137','33138','33139','33140','33141','33142','33143','33144','33145','33146','33147','33148','33149','33150','33151','33152','33153','33155','33156','33157','33161','33162','33165','33166','33167','33168','33169','33170','33172','33173','33174','33175','33176','33177','33178','33179','33180','33181','33182','33183','33184','33185','33186','33187','33188','33189','33190','33193','33194'])
vf.place(selector)

# create a button for the plot
image = app.image()
button_plot = app.button("trigger the computation of the plot")
vf.place(button_plot)
image.bind(plot_by_postalCode, papa_person_group,selector, triggers=[button_plot])

# create a button for the box plot
image3 = app.image()
button_box = app.button("trigger the computation of the box plot")
vf.place(button_box)
image3.bind(box_plot_by_postalCode, papa_person_group, selector ,triggers= [button_box])

# create a button for the plot
image4 = app.image()
button_hist = app.button("trigger the computation of the histogram")
vf.place(button_hist)
image4.bind(hist_plot_by_postalCode,papa_person_group, selector ,triggers= [button_hist])


# set the map of Miami
l=folium.Map(location = [25.761681
,-80.191788], #Initiate map on Miami city
zoom_start = 10,min_zoom = 10)
# create a button for the folium map
map = app.folium_chart(title= "Map of Miami's Restaurants", folium=l)
#plot the map
button_folium = app.button("trigger the computation of the map")
vf.place(button_folium)
map.bind(folium_map,papa_person, selector, triggers= [button_folium])

vf.place(image)
vf.place(image3)
vf.place(image4)
vf.place(map)


# Categorise the pizza places
#select other than pizza restaurant on data frame
not_pizza_place=papa_person[papa_person["pizza"] == False]

#select pizza that are not Papa John´s
not_papaj_place=papa_person[papa_person["pizza"] == True]
not_papaj_place=not_papaj_place[not_papaj_place["Papa John's Pizza"] == False]

#select papa johns pizza restaurant on data frame
pizza_johns=papa_person[papa_person["Papa John's Pizza"] >0]





# Machine Leraning part
vf2.place(app.text("""The goal of this algorithm is to determine the review of the Papa John's in every zipcode of the Miami area.""" 
"""We did the analysis based on the population of Miami, such as the income, the marital status, the employement, etc... """
"""The second part was base on the restaurants in the area, such as the reviews, the type of food, and the localisation."""
"""We will use a simple regression analysis to predict the score of the Papa Johns reviews, and use a min max scaling for continuous variable.""", markdown=True))

# select only the rows where review is greater than 0
papa_person = papa_person[papa_person['review'] > 0]
papa_person

vf2.place(app.text("""
    We will split our dataset into training and test sets (30% - 70% of the data) and we used a regression analysis: 
""", markdown=True))
# Import the necessary libraries
from sklearn.linear_model import LinearRegression

# define X and y 
import papa_functions
from papa_functions import select_col_machine, cols_machine
X = select_col_machine(papa_person, cols_machine)
# Select the columns that will be used as features and the target variable
y = papa_person["review"]

from sklearn.model_selection import train_test_split
# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create and fit the model

model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on the test set
y_pred = model.predict(X_test)

# import the necessary libraries
from sklearn.metrics import mean_squared_error, r2_score

# evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)

# calculate the r2 score
r2 = r2_score(y_test, y_pred)
vf2.place(app.text(
    f"""Intercept: {model.intercept_}""", 
    f"""Mean Squared Error: {mse}""",
    f"""R2 Score: {r2}""", markdown= True ))
y_pred = y_pred.round(1)
# create a dataframe with the actual and predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
vf2.place(app.text("""
    In this tab we looked at the influence of the features on the Papa John's restaurant reviews, to predict which Postal Code would be more Likely to host a new restaurant. 
""", markdown=True))
df= df.head(20)
vf2.place(app.table(df))

# Scale
# import required libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# preprocess the data
papa_person1 = select_col_machine(papa_person, cols_machine)
X = papa_person1
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# make predictions on the full dataset
predictions = model.predict(X)

# import required libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error

# import required libraries
import matplotlib.pyplot as plt

# create a scatter plot of the predictions
fig = plt.figure(figsize = (6,3), linewidth= 1)
plt.scatter(y_test, y_pred, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='True Values')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Model Predictions')
plt.legend()
plt.show()

img = app.image(fig)
vf2.place(img)

#get the papa data csv
papa = pd.read_csv("/root/Papa_johns/papa.csv")
# postalCode as the index
papa = papa.set_index('postalCode')
# predict the review for papa
papa_pred = model.predict(papa)
# predict the review for papa
papa["review"] = model.predict(papa)

# keep postalCode and review columns
papa["review"] = papa["review"].round(1)

# duplicate the postalCode column
papa["postalCode"] = papa.index
vf2.place(app.table(papa[["postalCode","review"]]))
vf2.place(app.text("""Here we have the predicted review for each postal code. We can clearly see that the zipcode that seems to be the most advantageous to open a new restaurant is 33134, which is West Miami""", markdown=True))


app.register()




