


T1 = """
    # Introduction



    Hello! 

    Would you like to know if it is possible to predict energy prices using Machine Learning? Do you think it is possible? Maybe this DataApp will interest you!

    The electricity market is regulated by REE (Red Electrica Española), and acts as an intermediary between companies that generate energy and 
    companies that buy this energy to distribute it to the customers. 

    Generated energy can come from different forms of generation, such as wind energy, solar energy with photovoltaic panels, nuclear energy... etc. 




    __How does the Spanish market work?__

    The price of electricity is calculated by means of a matching offer.

    Every day at 16:00 the next day's energy prices is calculated, and it is calculated by ordering from lowest to highest the prices at which energy 
    sellers want to sell the energy produced, and by ordering from highest to lowest the prices at which buyers want to buy the energy. The point where the two 
    lines intersect is called the matching point and that price is established as the energy price for that hour. 

    This process is repeated for each hourly segment of the following day and is managed by Operador del Mercado Ibérico-Polo Español (OMIE).


"""




T2 = """



    The left side of the graph is referred to as matched bids (the sell and buy bids that become firm commitments to deliver energy). 



    __The objective of this DataApp is to try to predict the energy matching price for each of the hours of a day.__


    ---

    # Data


    The data used in this project is a dataset extracted from the OMIE API. This API has 1400 indicators available for analysis, 
    among them information on scheduled power generation, real time generation, energy price including intraday market session prices. 

    For this use case a few KPIs have been selected, because they are the most complete data (no missing data) and in the case of power generation 
    indicators correspond to the energy sources that have the greatest impact on the Spanish energy market.



    Here's the list of KPIs used:


    - __Spot Price__                            Daily SPOT market price                             _Freq: Hourly data_
    - __Real Demand__                           Real energy demand in Spanish territory               _Freq: 10 Mins data_ 
    - __Scheduled Generation Hydraulics__       Scheduled hydroelectric power generation         _Freq: Hourly data_
    - __Scheduled Generation Nuclear__          Nuclear programmed power generation            _Freq: Hourly data_
    - __Scheduled Generation Combined cycle__   Combined Cycle programmed power generatio    _Freq: Hourly data_
    - __Scheduled Generation Wind Power__       Scheduled wind energy generation             _Freq: Hourly data_
    - __Scheduled Generation co-generation__    Programmed co-generation power generation         _Freq: Hourly data_
    - __Real time Generation Hydraulics__       Real-time generation of hydroelectric energy     _Freq: 10 Mins data_ 
    - __Real time Generation nuclear__          Real-time nuclear power generation        _Freq: 10 Mins data_ 
    - __Real time Generation Combined cycle__   Real-time combined cycle power generation _Freq: 10 Mins data_ 

    We can visualize the data below in this amazing Shapelets graphs! 


"""



T3 = """


    ---

    # Data Transform

    The objective of this use case is to __predict the price of energy__, so our target variable will be the Spot Price KPI,
     and our predictor variables, all the others.

    To do this we have to take into consideration that the data is not at the same frequency, we need to resample the data. 


    We have some initial business knowledge about this data, and we know that price evolution throughout the day has a very similar shape over the different days. 
    similar over the different days of the month. We can see it in this chart, which represents the price data for all the days of January 2020. 


"""

T4 = """

    Knowing this, you can use a dimensionality reduction algorithm such as PCA to reduce the dimensionality of the target variable. 


    ## Dimensionality reduction of the target variable.

    Currently, being in hourly format, I have 24 data points for each day. I am going to use the PCA algorithm to reduce the dimensionality for each day.
     The main goal is to be able to summarize the information from the 24 points into a single variable. We can do it easily with python and sci-kit learn!




    ```python
    # Create a dataset for PCA using price data

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA


    # Standard Scaler
    sc = StandardScaler()
    sc.fit(price_ts_np)
    X_train_std = sc.transform(price_ts_np)

    # PCA
    pca = PCA(1)
    data_fitted = pca.fit_transform(X_train_std)

    ```

    With just a single variable, I can explain 96% of the explanied variance!

    But, With this new target variable exist a problem. The data do not have the same time frequency!
    The target variable has daily data, some variables have hourly data, and another one every 10 minutes. Let's fix that!


    Let's resample the data so that they all have the same frequency, daily data.

    The data that has frequency every 10Mins will first be resampled to hourly.

    ```python
    df.resample('H').agg(['min','mean','max'])
    ```
    Now, for each variable that had frequency 10Min, we will have 3 new columns, indicating the minimum, average and maximum values. 

    With all the variables, repeat the same operation to have the data with daily frequency. 


    ```python
    df.resample('D').agg(['min','mean','max'])
    ```

    Now we can use models that are not specifically designed for time series, such as RandomForest, XGBoost or LightGBM.

    We have added some columns so that the model can know information about the day of the week and the day of the month of the data.

    _TODO: You can try adding information about the season of the year or month information, to see if the results improve!_



"""

T5 = """

    ---

    # Models

        This problem can be approached with different points of view. 

    - As an autoregressive time series, predicting the value of the principal component of the daily data.
    - As a multivariate time series problem, with the different KPIs we have. 
    - As a regression problem.

    For this approach, I have treated the problem as a regression problem, in which, with the processing of the previous point, I predict the value of the principal component that I have predicted. 
    the value of the principal component that I have calculated for the target. 

    We have used 3 algorithms:

    - the RandomForestRegressor algorithm from the `Scikit-Learn` package, and it is implemented like this:

    - The LightGBM algorithm, from microsoft.

    - The XGBoost algorithm. 


    How have we implemented the algorithms? Easy, look at this example.

    ```python
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(random_state=2022)

    # Train model
    model.fit(X_train,y_train)

    # Get predictions
    predictions = model.predict(X_test)

    ```

     With Shapelets you have all the power of python and its different packages, so it is as easy as importing and running them!

    To train the algorithm I have used data from all the KPIs from January 2015 to March 31, 2022.

    To evaluate the performance of the models, we are going to split the training dataset to have a train and a test, selecting 30% of the data randomly, and calculate some metrics. You can see which model is working better below!





"""



T6 = """

    Here's a short description about errors:
    - _MSE_ is the mean squared error.
    - _RMSE_ is the root of mean squarred error.
    - _MAE_ is the mean of absolute error.
    - _MAPE_ is the sum of the individual absolute errors divided by each period separately. It is the average of the percentage errors.



    --- 
    # Prediction


    Want to see the models in action?

    As of the date of development of this DataApp, we have data up to mid-April, so we will predict spot prices from April 1 to April 14, 2022.

"""

T7 = """
    ---

    # Conclussion



    We have reached the end!

    We have seen how we can process time series and how we can apply algorithms with that data.
    In addition, we have seen that the price of electricity is possible to predict, maybe with a more extensive data source, or more data sources, 
     it can be better tuned. 

    I leave that to you! 

"""
































