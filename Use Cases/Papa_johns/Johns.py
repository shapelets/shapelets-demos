
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
from shapelets.apps import DataApp, Image, FoliumChart, Widget
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
tabs_fp = app.tabs_layout("My tabs layout")

# Create two tabs and add a vertical layout in each of them 
tabs_fp.add_tab("Tab 1", vf)
tabs_fp.add_tab("Tab 2", vf2)

# Place markdown texts in each of the vertical layouts
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

# create a function that will group by postalCode plot it and return the result
def plot_by_postalCode(papa_person_group: DataFrame, selector: str)-> Image :

    selector = int(selector)

    # Create smaller dataframe to split some categories
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

    # Image 
    income = income[income["postalCode"] == selector]
    languages = languages[languages["postalCode"] == selector]
    population = population[population["postalCode"] == selector]
    married = married[married["postalCode"] == selector]
    male_female = male_female[male_female["postalCode"] == selector]
    household = household[household["postalCode"] == selector]
    education = education[education["postalCode"] == selector]
    employment = employment[employment["postalCode"] == selector]
    employement_group = employement_group[employement_group["postalCode"] == selector]
    age = age[age["postalCode"] == selector]
    education = education[education["postalCode"] == selector]

    inc = income.plot(kind='bar', figsize=(10, 6))
    lan=languages.plot(kind='bar', figsize=(10, 6))
    mar=married.plot(kind='bar', figsize=(10, 6))
    mal=male_female.plot(kind='bar', figsize=(10, 6))
    pop=population.plot(kind='bar', figsize=(10, 6))
    ed=education.plot(kind='bar', figsize=(10, 6))
    emp=employment.plot(kind='bar', figsize=(10, 6))
    empg=employement_group.plot(kind='bar', figsize=(10, 6))
    ag=age.plot(kind='bar', figsize=(10, 6))
    hou=household.plot(kind='bar', figsize=(10, 6))
    lef=leftover.plot(kind='bar', figsize=(10, 6))

    return Image(inc)
    # get inc into an image 
    img1 = app.image(inc)
    img2 = app.image(lan)
    img3 = app.image(mar)
    img4 = app.image(mal)
    img5 = app.image(pop)
    img6 = app.image(ed)
    img7 = app.image(emp)
    img8 = app.image(empg)
    img9 = app.image(ag)
    img10 = app.image(hou)
    img11 = app.image(lef)

    # convert img to an image object
    return vf.place(img1), vf.place(img2),vf.place(img3),vf.place(img4),vf.place(img5),vf.place(img6),vf.place(img7),vf.place(img8),vf.place(img9),vf.place(img10),vf.place(img11)


def cat_plot_by_postalCode(papa_person_group: DataFrame, selector: int)-> Image :

    # Create smaller dataframe to split some categories
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

    # Image 
    income = income[income["postalCode"] == selector]
    languages = languages[languages["postalCode"] == selector]
    population = population[population["postalCode"] == selector]
    married = married[married["postalCode"] == selector]
    male_female = male_female[male_female["postalCode"] == selector]
    household = household[household["postalCode"] == selector]
    education = education[education["postalCode"] == selector]
    employment = employment[employment["postalCode"] == selector]
    employement_group = employement_group[employement_group["postalCode"] == selector]
    age = age[age["postalCode"] == selector]
    education = education[education["postalCode"] == selector]

    inc = sns.catplot(data=income, kind="bar", height=5, aspect=2, orient="h")
    lan = sns.catplot(data=languages, kind="bar", height=6, aspect=2, orient='h')
    mar = sns.catplot(data=married, kind="bar", height=6, aspect=2, orient='h')
    mal = sns.catplot(data=male_female, kind="bar", height=6, aspect=2, orient='h')
    pop = sns.catplot(data=population, kind="bar", height=6, aspect=2, orient='h')
    ed = sns.catplot(data=education, kind="bar", height=6, aspect=2, orient='h')
    emp = sns.catplot(data=employment, kind="bar", height=6, aspect=2, orient='h')
    empg = sns.catplot(data=employement_group, kind="bar", height=6, aspect=2, orient='h')
    ag = sns.catplot(data=age, kind="bar", height=6, aspect=2, orient='h')
    hou = sns.catplot(data=household, kind="bar", height=6, aspect=2, orient='h')
    lef = sns.catplot(data=leftover, kind="bar", height=6, aspect=2, orient='h')
    
    return Image(inc)

    # get inc into an image 
    img1 = app.image(inc)
    img2 = app.image(lan)
    img3 = app.image(mar)
    img4 = app.image(mal)
    img5 = app.image(pop)
    img6 = app.image(ed)
    img7 = app.image(emp)
    img8 = app.image(empg)
    img9 = app.image(ag)
    img10 = app.image(hou)
    img11 = app.image(lef)

    # convert img to an image object
    return vf.place(img1), vf.place(img2),vf.place(img3),vf.place(img4),vf.place(img5),vf.place(img6),vf.place(img7),vf.place(img8),vf.place(img9),vf.place(img10),vf.place(img11)

def box_plot_by_postalCode(papa_person_group: DataFrame, selector: int)-> Image :

    # Create smaller dataframe to split some categories
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

    # Image 
    income = income[income["postalCode"] == selector]
    languages = languages[languages["postalCode"] == selector]
    population = population[population["postalCode"] == selector]
    married = married[married["postalCode"] == selector]
    male_female = male_female[male_female["postalCode"] == selector]
    household = household[household["postalCode"] == selector]
    education = education[education["postalCode"] == selector]
    employment = employment[employment["postalCode"] == selector]
    employement_group = employement_group[employement_group["postalCode"] == selector]
    age = age[age["postalCode"] == selector]
    education = education[education["postalCode"] == selector]

    inc = income.plot(kind='box', figsize=(8, 6), vert=False)
    lan = languages.plot(kind='box', figsize=(8, 6), vert=False)
    mar = married.plot(kind='box', figsize=(8, 6), vert=False)
    mal = male_female.plot(kind='box', figsize=(8, 6), vert=False)
    pop = population.plot(kind='box', figsize=(8, 6), vert=False)
    ed = education.plot(kind='box', figsize=(8, 6), vert=False)
    emp = employment.plot(kind='box', figsize=(8, 6), vert=False)
    empg = employement_group.plot(kind='box', figsize=(8, 6), vert=False)
    ag = age.plot(kind='box', figsize=(8, 6), vert=False)
    hou = household.plot(kind='box', figsize=(8, 6), vert=False)
    lef = leftover.plot(kind='box', figsize=(8, 6), vert=False) 

    return Image(inc)

    # get inc into an image 
    img1 = app.image(inc)
    img2 = app.image(lan)
    img3 = app.image(mar)
    img4 = app.image(mal)
    img5 = app.image(pop)
    img6 = app.image(ed)
    img7 = app.image(emp)
    img8 = app.image(empg)
    img9 = app.image(ag)
    img10 = app.image(hou)
    img11 = app.image(lef)

    # convert img to an image object
    return vf.place(img1), vf.place(img2),vf.place(img3),vf.place(img4),vf.place(img5),vf.place(img6),vf.place(img7),vf.place(img8),vf.place(img9),vf.place(img10),vf.place(img11)

def hist_plot_by_postalCode(papa_person_group: DataFrame, selector: int)-> Image :
    # Create smaller dataframe to split some categories
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

    # Image 
    income = income[income["postalCode"] == selector]
    languages = languages[languages["postalCode"] == selector]
    population = population[population["postalCode"] == selector]
    married = married[married["postalCode"] == selector]
    male_female = male_female[male_female["postalCode"] == selector]
    household = household[household["postalCode"] == selector]
    education = education[education["postalCode"] == selector]
    employment = employment[employment["postalCode"] == selector]
    employement_group = employement_group[employement_group["postalCode"] == selector]
    age = age[age["postalCode"] == selector]
    education = education[education["postalCode"] == selector]

    inc = income.plot(kind='hist', figsize=(10, 6), bins=50)
    lan = languages.plot(kind='hist', figsize=(10, 6), bins=50)
    mar = married.plot(kind='hist', figsize=(10, 6), bins=50)
    mal = male_female.plot(kind='hist', figsize=(10, 6), bins=50)
    pop = population.plot(kind='hist', figsize=(10, 6), bins=50)
    ed = education.plot(kind='hist', figsize=(10, 6), bins=50)
    emp = employment.plot(kind='hist', figsize=(10, 6), bins=50)
    empg = employement_group.plot(kind='hist', figsize=(10, 6), bins=50)
    ag = age.plot(kind='hist', figsize=(10, 6), bins=50)
    hou = household.plot(kind='hist', figsize=(10, 6), bins=50)
    lef = leftover.plot(kind='hist', figsize=(10, 6), bins=50)

    return Image(inc)    
    # get inc into an image 
    img1 = app.image(inc)
    img2 = app.image(lan)
    img3 = app.image(mar)
    img4 = app.image(mal)
    img5 = app.image(pop)
    img6 = app.image(ed)
    img7 = app.image(emp)
    img8 = app.image(empg)
    img9 = app.image(ag)
    img10 = app.image(hou)
    img11 = app.image(lef)

    # convert img to an image object
    return vf.place(img1), vf.place(img2),vf.place(img3),vf.place(img4),vf.place(img5),vf.place(img6),vf.place(img7),vf.place(img8),vf.place(img9),vf.place(img10),vf.place(img11)

def folium_map(papa_person: DataFrame, selector: int)-> FoliumChart :

    print("hello")
    return FoliumChart()

    # Folium map
    # create a map of Miami
    l=folium.Map(location = [25.761681
    ,-80.191788], #Initiate map on Miami city
                    zoom_start = 10,
                    min_zoom = 10)

    marker_cluster = MarkerCluster().add_to(l)

    papa_person = papa_person[papa_person["postalCode"] == selector]

    #select other than pizza restaurant on data frame
    not_pizza_place=papa_person[papa_person["pizza"] == False]


    #select pizza that are not Papa John´s
    not_papaj_place=papa_person[papa_person["pizza"] == True]
    not_papaj_place=not_papaj_place[not_papaj_place["Papa John's Pizza"] == False]

    #select papa johns pizza restaurant on data frame
    pizza_johns=papa_person[papa_person["Papa John's Pizza"] >0]

    #plot the map with all the locations
    for index,place in not_pizza_place.iterrows():
        folium.Marker([place["latitude"], place["longitude"]], popup=(place["name"],place["address"],place["postalCode"])).add_to(marker_cluster)

    #plot the map with all the location
    for index, place in not_papaj_place.iterrows():
        folium.Marker([place["latitude"], place["longitude"]],icon=folium.Icon(color='green', icon='ok-sign'), popup=(place["name"],place["address"],place["postalCode"])).add_to(marker_cluster)

    #plot the map with all the location
    for index, place in pizza_johns.iterrows():
        folium.Marker([place["latitude"], place["longitude"]],icon=folium.Icon(color='red', icon='ok-sign'), popup=(place["name"],place["address"],place["postalCode"])).add_to(marker_cluster)

    
    # plot the map with vectors between the restaurant and the minimum distance restaurant 
    for index, place in not_pizza_place.iterrows():
        folium.PolyLine(locations=[[place["latitude"], place["longitude"]],[place["latitude_min_restaurant"], place["longitude_min_restaurant"]]],color="black", weight=2.5, opacity=1).add_to(l)

    # plot the map with vectors between the restaurant and the minimum distance restaurant
    for index, place in not_papaj_place.iterrows():
        folium.PolyLine(locations=[[place["latitude"], place["longitude"]],[place["latitude_min_restaurant"], place["longitude_min_restaurant"]]],color="green", weight=2.5, opacity=1).add_to(l)

    # plot the map with vectors between the restaurant and the minimum distance restaurant
    for index, place in pizza_johns.iterrows():
        folium.PolyLine(locations=[[place["latitude"], place["longitude"]],[place["latitude_min_restaurant"], place["longitude_min_restaurant"]]],color="red", weight=2.5, opacity=1).add_to(l)

    #plot the map
    folium_chart = app.folium_chart(title= "Map of Miami's Restaurants", folium=l)
    
    
    return vf.place(folium_chart)
    

# import the full data csv file
papa_person = pd.read_csv('Papa_johns/papa_person.csv')


papa_person_group = papa_person.groupby("postalCode").mean()


# create the selector for the postalCodes
selector = app.selector(title="Select the Postal Code", options=['33122','33125','33126','33127','33128','33129','33130','33131','33132','33133','33134','33135','33136','33137','33138','33139','33140','33141','33142','33143','33144','33145','33146','33147','33148','33149','33150','33151','33152','33153','33155','33156','33157','33161','33162','33165','33166','33167','33168','33169','33170','33172','33173','33174','33175','33176','33177','33178','33179','33180','33181','33182','33183','33184','33185','33186','33187','33188','33189','33190','33193','33194'])
vf.place(selector)

image = app.image()
# create a button for the plot
button_plot = app.button("trigger the computation of the plot")
vf.place(button_plot)
image.bind(plot_by_postalCode,papa_person, selector,triggers= [button_plot])
vf.place(image)

'''
# create a button for the plot
button_cat = app.button("trigger the computation of the catplot")
vf.place(button_cat)
image.bind(cat_plot_by_postalCode,papa_person, selector ,triggers= [button_cat])

# create a button for the box plot
button_box = app.button("trigger the computation of the box plot")
vf.place(button_box)
image.bind(box_plot_by_postalCode,papa_person, selector ,triggers= [button_box])

# create a button for the plot
button_hist = app.button("trigger the computation of the histogram")
vf.place(button_hist)
image.bind(hist_plot_by_postalCode,papa_person, selector ,triggers= [button_hist])

# create a button for the folium map
map = app.folium_chart()
button_folium = app.button("trigger the computation of the map")
vf.place(button_folium)
map.bind(folium_map, papa_person, selector, triggers= [button_folium])
'''
# ## Take-away
# * There are more Spanish speaking home
# * There are equal amount of people married and never married 
# * There is slightly more women tan men 
# * There are a majority of white people
# * There are a majority of people who graduated with a High school diploma or less 
# * More people are white colar and are working as a salesperson or in admin
# * And more people are either single of 2 people household, but household with atleast 1 kid is higher


# ## Take-away
# * employement and population has outliers

#plt.figure(figsize=(20,60), facecolor='white')
#plotnumber =1
#for hist in population:
    #ax = plt.subplot(12,3,plotnumber)
    #sns.distplot(population[hist])
    #plt.xlabel(hist)
    #plotnumber+=1
#plt.show()

#plt.figure(figsize=(20,60), facecolor='white')
#plotnumber =1
#for hist in education:
   #ax = plt.subplot(12,3,plotnumber)
    #sns.distplot(education[hist])
    #plt.xlabel(hist)
    #plotnumber+=1
#plt.show()

#plt.figure(figsize=(20,60), facecolor='white')
#plotnumber =1
#for hist in age:
    #ax = plt.subplot(12,3,plotnumber)
    #sns.distplot(age[hist])
    #plt.xlabel(hist)
    #plotnumber+=1
#plt.show()

#plt.figure(figsize=(20,60), facecolor='white')
#plotnumber =1
#for hist in languages:
    #ax = plt.subplot(12,3,plotnumber)
    #sns.distplot(languages[hist])
    #plt.xlabel(hist)
    #plotnumber+=1
#plt.show()


# ## Take-away
# Most of the variables are well distributed, except the one that have outliers
# ## Insight
# We did an overview of the population in Miami, right now what we want to do is to see the relationship between this population and the papa Johns restaurant

# Machine learning Part



vf2.place(app.text("""
    # Model fitting, model selection and classification results
    We will split our dataset into training and test sets (70% - 30% of the data) and we used a regression analysis: 
""", markdown=True))
# import papa_person.csv
papa_person = pd.read_csv('Papa_johns/papa_person.csv')
# get only where name is Papa John's Pizza
papa_johns = papa_person[papa_person['name'] == "Papa John's Pizza"]

# Import the necessary libraries
from sklearn.linear_model import LinearRegression

# define X
from papa_functions import select_col_machine, cols_machine
X = select_col_machine(papa_johns, cols_machine)
# Select the columns that will be used as features and the target variable
y = papa_johns["review"]

from sklearn.model_selection import train_test_split
# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on the test set
y_pred = model.predict(X_test)

# print the model's coefficients
vf2.place(app.text(f"""Intercept: {model.intercept_}""", markdown= True ))
vf2.place(app.text(f"""Slope: {model.coef_}""", markdown= True ))

# evaluate the model's performance
mse = np.mean((y_pred - y_test) ** 2)
vf2.place(app.text(f"""Mean Squared Error: {mse}""", markdown= True ))


# Scale
# import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

# preprocess the data
papa_person1 = select_col_machine(papa_person, cols_machine)
X = papa_person1
scaler = StandardScaler()
X = scaler.fit_transform(X)

# make predictions on the full dataset
predictions = model.predict(X)

# import required libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error

# evaluate model's performance
vf2.place(app.text(f"""MSE: {mean_squared_error(y_test, y_pred)}""" , markdown= True ))
vf2.place(app.text(f"""MAE: {mean_absolute_error(y_test, y_pred)}""" , markdown= True ))

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

img = app.image(fig)
vf2.place(img)

papa_person1 = select_col_machine(papa_person, cols_machine)
# Create a Linear Regression model
model = LinearRegression()

y = papa_person["review"]
# Fit the model to the data
model.fit(X, y)
# Make predictions using the model
y_pred = model.predict(X)

# Print the coefficients and intercept of the model
vf2.place(app.text(f"""Coefficients: {model.coef_}""", markdown= True ))

vf2.place(app.text( f"Intercept: {model.intercept_}", markdown= True))
# Calculate the R2 score to evaluate the model's performance
r2_score = model.score(X, y)
vf2.place(app.text(f"""R2 score: {r2_score}"""))



# evaluate the model using 5-fold cross-validation
# import the cross_val_score function
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)

vf2.place(app.text(f""""Cross-validated scores: {scores}""" , markdown= True ))



app.register()




