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
from pandas import DataFrame 
from shapelets.apps import DataApp, FoliumChart, Image, AltairChart
import altair as alt
app = DataApp(name= "Papa Johns restaurants in Miami", description= "This app shows the locations aim to find the best location to open a restaurant in the Miami Area")

markdown_text = app.text("""
In this dataapp we will try to analyze the restaurant's in the Miami Area, especially the Pizza places and the Papa John's. We are aimimg to determine, which neighborhood is the most successful and which one is more likely to succeed. 
We will use the data from the US Census Bureau and the Foursquare API to determine the best neighborhood for a new Papa John's Pizza restaurant.""", markdown = True)
app.place(markdown_text)

# create a tab for the EDA and the machine learning part
# Create two vertical layouts
vf_eda = app.vertical_layout()
vf_ml = app.vertical_layout()

# Create a tabs layout
tabs_fp = app.tabs_layout("My Vertical layout")
tabs_fp.add_tab("EDA", vf_eda)
tabs_fp.add_tab("ML", vf_ml)
hf_bar = app.horizontal_layout()
hf_box = app.horizontal_layout()
hf_hist = app.horizontal_layout()



vf_eda.place(app.text("""
    # Population tab
    In this tab we are looking at the population of the Miami Area, to understand better its repartition over the neighborhood.
""", markdown=True))
vf_eda.place(app.text("""## Bar chart
In this section, we aim to gain insights into the population of the Miami Area by examining its distribution across different neighborhoods. Specifically, we will explore how the population is distributed across various neighborhoods in the Miami Area and identify any patterns or trends that may emerge. This analysis will enable us to better understand the demographics of the region and identify potential areas of interest for future research or development.""", markdown=True))
vf_eda.place(hf_bar)
vf_eda.place(app.text("""## Box plot
In this section, we will investigate the distribution of a numerical variable in the Miami Area using box plots. We will use this visualization technique to identify patterns in the distribution of the variable across different neighborhoods and compare their statistical properties. Specifically, we will examine the quartiles, median, and outliers of the variable in each neighborhood, and analyze any differences or similarities between them. This analysis will allow us to gain insights into the variability and central tendency of the variable in the Miami Area and identify any potential outliers or anomalies.""", markdown=True))
vf_eda.place(hf_box)
vf_eda.place(app.text("""## Histogram
In this section, we will explore the distribution of a numerical variable in the Miami Area using histograms. We will use this visualization technique to identify the shape and central tendency of the distribution and analyze any patterns or trends across different neighborhoods. Specifically, we will examine the frequency distribution of the variable in each neighborhood and compare their histograms to identify similarities and differences. This analysis will allow us to gain insights into the variability and central tendency of the variable in the Miami Area and identify any potential anomalies or patterns.""", markdown=True))
vf_eda.place(hf_hist)
vf_ml.place(app.text("""
    # ML tab
    In this tab we looked at the Machine learning model to see if we can determine which neighborhood would be the best suited to host a new restaurant. 
""", markdown=True))



# import the full data csv file
papa_person = pd.read_csv('/root/Papa_johns_last/papa_person1.csv')
papa_person_group = papa_person.groupby("postalCode").mean()

one = papa_person_group.iloc[:,84+10:]
two = papa_person_group.iloc[:,41+10:44+10]
population = papa_person_group.iloc[:,7+10:12+10]
language = papa_person_group.iloc[:,12+10:17+10]
married = papa_person_group.iloc[:,17+10:22+10]
male_female = papa_person_group.iloc[:,4+10:6+10]
household = papa_person_group.iloc[:,22+10:29+10]
income = papa_person_group.iloc[:,29+10:41+10]
education = two.join(one)
employment = papa_person_group.iloc[:,45+10:66+10]
employment_group = papa_person_group.iloc[:,66+10:69+10]
age = papa_person_group.iloc[:,69+10:84+10]
leftover = papa_person_group.iloc[:,0+10:4+10]
cat={"Population":population, "Language":language, "Married":married, "Male Female":male_female, "Household":household, "Income":income, "Education":education, "Employement":employment, "Employement Group":employment_group, "Age":age, "Leftover":leftover}

def full_bar(cat: dict, selector_cat: str) -> AltairChart:
    cat = cat[selector_cat]
    melted_cat = pd.melt(cat.reset_index(), id_vars=['postalCode'], var_name='Feature', value_name='Count')

    # create stacked bar chart
    chart = alt.Chart(melted_cat).mark_bar().encode(
        x='postalCode:N',
        y='Count:Q',
        color='Feature:N',
        tooltip=['postalCode:N', 'Feature:N', 'Count:Q']
    ).properties(width=600, height=400, title=f'Bar Chart by Zipcode and {selector_cat}').interactive()

    return app.altair_chart(chart=chart)

def full_box(cat: dict,selector_cat: str)-> AltairChart :
    cat = cat[selector_cat]
    melted_cat = pd.melt(cat.reset_index(), id_vars=['postalCode'], var_name='Feature', value_name='Count')

    # create box plot
    chart = alt.Chart(melted_cat[melted_cat['Feature'] != 'postalCode']).mark_boxplot().encode(
        y='Count:Q',
        x='Feature:N'
    ).properties(width=600, height=400, title=f'Box plot by {selector_cat}').interactive()
    return app.altair_chart(chart=chart)

def full_hist(cat: dict,selector_cat: str) -> AltairChart:
    cat = cat[selector_cat]
    melted_cat = pd.melt(cat.reset_index(), id_vars=['postalCode'], var_name='Feature', value_name='Count')
    melted_population_no_pc = melted_cat[melted_cat['Feature'] != 'postalCode']

    # create histogram chart
    chart = alt.Chart(melted_population_no_pc).mark_bar().encode(
        x=alt.X('Count:Q', bin=True),
        y=alt.Y('count()', axis=alt.Axis(title='Number of Zipcodes')),
        color='Feature:N'
    ).properties(width=600, height=400, title=f'Distribution of {selector_cat} by Zipcode').interactive()

    return app.altair_chart(chart=chart)


def part_bar(cat: dict,selector: int,selector_cat: str)-> AltairChart :
    
    cat = cat[selector_cat]
    cat = cat[cat.index == selector]
    melted_cat = pd.melt(cat.reset_index(), id_vars=['postalCode'], var_name='Feature', value_name='Count')

    # create stacked bar chart
    chart = alt.Chart(melted_cat).mark_bar().encode(
        x='postalCode:N',
        y='Count:Q',
        color='Feature:N',
        tooltip=['postalCode:N', 'Feature:N', 'Count:Q']
    ).properties(width=600, height=400, title=f'Bar Chart by {selector} and {selector_cat}').interactive()

    return app.altair_chart(chart=chart)

def part_box(cat:dict,selector: int,selector_cat: str)-> AltairChart :
    
    cat = cat[selector_cat]
    cat = cat[cat.index == selector]
    melted_cat = pd.melt(cat.reset_index(), id_vars=['postalCode'], var_name='Feature', value_name='Count')

    # create box plot
    box = alt.Chart(melted_cat).mark_boxplot().encode(
        y='Count:Q',
        x=alt.X('Feature:N', axis=alt.Axis(title='Feature')),
    )

    points = alt.Chart(melted_cat).mark_circle(opacity=0.5, size=30).encode(
        y='Count:Q',
        x=alt.X('Feature:N', axis=alt.Axis(title='Feature')),
    )
    chart = (box + points).properties(width=600, height=400, title=f'Box plot by {selector} and {selector_cat}')
    return app.altair_chart(chart=chart)

def part_hist(cat:dict,selector: int,selector_cat: str)-> AltairChart :
    cat = cat[selector_cat]
    cat = cat[cat.index == selector]
    melted_cat = pd.melt(cat.reset_index(), id_vars=['postalCode'], var_name='Feature', value_name='Count')
    melted_population_no_pc = melted_cat[melted_cat['Feature'] != 'postalCode']

    # create histogram chart
    chart = alt.Chart(melted_population_no_pc).mark_bar().encode(
        x=alt.X('Count:Q', bin=True),
        y=alt.Y('count()', axis=alt.Axis(title='Number of Zipcodes')),
        color='Feature:N'
    ).properties(width=600, height=400, title=f'Distribution of {selector} and {selector_cat}').interactive()

    return app.altair_chart(chart=chart)

def folium_map(papa_person: DataFrame, selector: int)-> FoliumChart :
    # Folium map
    l=folium.Map(location = [25.761681,-80.191788], #Initiate map on Miami city 
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
    for place in not_pizza_place.iterrows():
        folium.Marker([place["latitude"], place["longitude"]], popup=(place["name"],place["address"])).add_to(marker_cluster)
    #plot the map with all the location
    for place in not_papaj_place.iterrows():
        folium.Marker([place["latitude"], place["longitude"]],icon=folium.Icon(color='green', icon='ok-sign'), popup=(place["name"],place["address"])).add_to(marker_cluster)
    #plot the map with all the location
    for place in pizza_johns.iterrows():
        folium.Marker([place["latitude"], place["longitude"]],icon=folium.Icon(color='red', icon='ok-sign'), popup=(place["name"],place["address"])).add_to(marker_cluster)

    return app.folium_chart(title= "Map of Miami's Restaurants", folium=l)
    

# create the selector for the postalCodes
selector = app.selector(title="Select the Postal Code", options = ['33122','33125','33126','33127','33128','33129','33130','33131','33132','33133','33134','33135','33136','33137','33138','33139','33140','33141','33142','33143','33144','33145','33146','33147','33148','33149','33150','33151','33152','33153','33155','33156','33157','33161','33162','33165','33166','33167','33168','33169','33170','33172','33173','33174','33175','33176','33177','33178','33179','33180','33181','33182','33183','33184','33185','33186','33187','33188','33189','33190','33193','33194'])
selector_cat = app.selector(title="Select the Categorie", options=['Income', 'Language', 'Population', 'Married', 'Education', 'Male Female', 'Employment', 'Employement Group', 'Age', 'Household', 'Leftover'])

# set the map of Miami
l=folium.Map(location = [25.761681
,-80.191788], #Initiate map on Miami city
zoom_start = 10,min_zoom = 10)
# create a button for the folium map
map = app.folium_chart(title= "Map of Miami's Restaurants", folium=l)
#Bind the map
map.bind(folium_map,papa_person, selector)

"""
image_full_bar = app.altair_chart()
image_full_bar.bind(full_bar, cat, selector_cat)
image_full_box= app.altair_chart()
image_full_box.bind(full_box, cat, selector_cat)
image_full_hist = app.altair_chart()
image_full_hist.bind(full_hist, cat, selector_cat)
image_part_bar = app.altair_chart()
image_part_bar.bind(part_bar, cat, selector, selector_cat)
image_part_box = app.altair_chart()
image_part_box.bind(part_box, cat, selector, selector_cat)
image_part_hist = app.altair_chart()
image_part_hist.bind(part_hist, cat, selector, selector_cat)


hf_bar.place(image_full_bar)
hf_bar.place(image_part_bar)
hf_box.place(image_full_box)
hf_box.place(image_part_box)
hf_hist.place(image_full_hist)
hf_hist.place(image_part_hist)
"""
# Place everything in the data app
# first place the map
app.place(selector)
app.place(map)
app.place(selector_cat)
app.place(tabs_fp)


app.register()
