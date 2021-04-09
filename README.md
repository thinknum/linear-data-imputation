# Linear Data Imputation

This allows you to fill-in missing values from your data. It uses a distribution estimated from the mean and covariance of your data.

## Installation

```
pip install linear-imputation
```

## How to use

You might have some data which is missing some values:

```
>>> import pandas as pd
>>> import numpy as np
>>> from linear_imputation import impute, Imputer
>>> 
>>> input_data = pd.DataFrame({'age': [10,20,30], 'pets':[100,200,None]})
>>> input_data                                                                                        
   age   pets
0   10  100.0
1   20  200.0
2   30    NaN
```

To fill-in the missing values of your data, you only have to call the `impute` function:

```
>>> impute(input_data) 
    age   pets
0  10.0  100.0
1  20.0  200.0
2  30.0  187.5
```

The filled-in values are considered the most likely, given the distribution of your data.

Sometimes it is useful to build a model from some training data to later apply this model to some other data.
Building a model is easy:

```
>>> model = Imputer(input_data)
```

You can then use it to fill-in missing values of other data you have:

```
>>> marty = {'name': "Marty", 'age': None, 'pets': 150} 
>>> model.impute(marty) 
{'name': 'Marty', 'age': 20.0, 'pets': 150}
```

The data to be completed can also be a pandas.DataFrame

```
>>> df = pd.DataFrame([marty, {'name': 'Tom', 'age': 35}]) 
>>> model.impute(df) 
    name   age    pets
0  Marty  20.0  150.00
1    Tom  35.0  206.25
```

You can also use a numpy.ndarray

```
>>> matrix = np.array([[10,100], [20, 200], [30, None]]) 
>>> impute(matrix)
array([[10, 100],
       [20, 200],
       [30, 187.5]])
```
