import numpy as np
import pandas as pd
from typing import Annotated
from fastapi import Depends, FastAPI

app = FastAPI()
app.state.initialized = False
app.state.df = pd.DataFrame()

def add_random_number(input):
    return input + np.random.choice(a=[-1, 0, 1])

@app.get("/")
async def read_root():
    dims = 2
    step_n = 1000
    step_set = [-1, 0, 1]
    if not app.state.initialized:
        #Create random walk
        origin = np.zeros((1,dims))
        # Simulate steps in 1D
        step_shape = (step_n-1,dims)
        steps = np.random.choice(a=step_set, size=step_shape)
        path = np.concatenate([origin, steps]).cumsum(0)
        # Create pd.DataFrame from list
        app.state.df = (pd.DataFrame(data=path, index=range(path.shape[0])))
        app.state.initialized = True  
    app.state.df.drop(index=0,inplace=True)
    app.state.df.reset_index(drop=True,inplace=True)
    s1 = pd.DataFrame(app.state.df.iloc[step_n-2].apply(add_random_number).to_dict(),index=[step_n-1])
    app.state.df = pd.concat([app.state.df,s1])
    return {"data": app.state.df.to_json()}
    






