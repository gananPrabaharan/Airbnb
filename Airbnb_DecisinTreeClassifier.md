```python
import requests
import pandas as pd
from pandas import DataFrame 
!pip install geopy
from geopy.extra.rate_limiter import RateLimiter


import matplotlib.pyplot as plt
import zipfile
import numpy as np
import io
import time
from pprint import pprint 
print("Import Successful")
#!wget https://www150.statcan.gc.ca/n1/tbl/csv/10100084-eng.zip
#!unzip 10100084-eng.zip
```

    Requirement already satisfied: geopy in /opt/conda/lib/python3.8/site-packages (2.1.0)
    Requirement already satisfied: geographiclib<2,>=1.49 in /opt/conda/lib/python3.8/site-packages (from geopy) (1.50)
    Import Successful


# Downloading the data


```python



df = pd.read_csv("TorontoListings.csv")

```

# Data Cleaning


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>neighbourhood_group</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.874600e+04</td>
      <td>1.874600e+04</td>
      <td>0.0</td>
      <td>18746.000000</td>
      <td>18746.000000</td>
      <td>18746.000000</td>
      <td>18746.000000</td>
      <td>18746.000000</td>
      <td>14788.000000</td>
      <td>18746.000000</td>
      <td>18746.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.673976e+07</td>
      <td>1.087340e+08</td>
      <td>NaN</td>
      <td>43.680717</td>
      <td>-79.397420</td>
      <td>135.468473</td>
      <td>10.786568</td>
      <td>26.635602</td>
      <td>1.214266</td>
      <td>4.715246</td>
      <td>125.690761</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.298332e+07</td>
      <td>1.055992e+08</td>
      <td>NaN</td>
      <td>0.048263</td>
      <td>0.064453</td>
      <td>263.761517</td>
      <td>36.913904</td>
      <td>52.418576</td>
      <td>1.597121</td>
      <td>10.354866</td>
      <td>136.844548</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.419000e+03</td>
      <td>1.565000e+03</td>
      <td>NaN</td>
      <td>43.586710</td>
      <td>-79.634800</td>
      <td>11.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.634553e+07</td>
      <td>2.049040e+07</td>
      <td>NaN</td>
      <td>43.645170</td>
      <td>-79.426310</td>
      <td>61.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.190000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.775978e+07</td>
      <td>6.787450e+07</td>
      <td>NaN</td>
      <td>43.662815</td>
      <td>-79.397350</td>
      <td>99.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>0.600000</td>
      <td>1.000000</td>
      <td>83.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.853432e+07</td>
      <td>1.823192e+08</td>
      <td>NaN</td>
      <td>43.700153</td>
      <td>-79.376860</td>
      <td>150.000000</td>
      <td>7.000000</td>
      <td>27.000000</td>
      <td>1.580000</td>
      <td>3.000000</td>
      <td>252.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.578971e+07</td>
      <td>3.711952e+08</td>
      <td>NaN</td>
      <td>43.836900</td>
      <td>-79.127810</td>
      <td>13137.000000</td>
      <td>1125.000000</td>
      <td>828.000000</td>
      <td>15.440000</td>
      <td>88.000000</td>
      <td>365.000000</td>
    </tr>
  </tbody>
</table>
</div>



From above, We can see there is a column which is empty, Hence, We drop empty, NAN data from the set. 


```python

df.dtypes
```




    id                                  int64
    name                               object
    host_id                             int64
    host_name                          object
    neighbourhood_group               float64
    neighbourhood                      object
    latitude                          float64
    longitude                         float64
    room_type                          object
    price                               int64
    minimum_nights                      int64
    number_of_reviews                   int64
    last_review                        object
    reviews_per_month                 float64
    calculated_host_listings_count      int64
    availability_365                    int64
    dtype: object



Since id and host_name, both are giving the same information, We keep one of them and We will do the same thing for name and coordination(lattitude+longitude), We will concatenate them into coulumn location


```python
df['location'] = [', '.join(str(x) for x in y) for y in map(tuple, df[['latitude', 'longitude']].values)]
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
      <th>location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1419</td>
      <td>Beautiful home in amazing area!</td>
      <td>1565</td>
      <td>Alexandra</td>
      <td>NaN</td>
      <td>Little Portugal</td>
      <td>43.64617</td>
      <td>-79.42451</td>
      <td>Entire home/apt</td>
      <td>469</td>
      <td>4</td>
      <td>7</td>
      <td>2017-12-04</td>
      <td>0.11</td>
      <td>1</td>
      <td>0</td>
      <td>43.64617, -79.42451</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8077</td>
      <td>Downtown Harbourfront Private Room</td>
      <td>22795</td>
      <td>Kathie &amp; Larry</td>
      <td>NaN</td>
      <td>Waterfront Communities-The Island</td>
      <td>43.64105</td>
      <td>-79.37628</td>
      <td>Private room</td>
      <td>98</td>
      <td>180</td>
      <td>169</td>
      <td>2013-08-27</td>
      <td>1.24</td>
      <td>2</td>
      <td>365</td>
      <td>43.64105, -79.37628000000001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12604</td>
      <td>Seaton Village Parlour Bedroom</td>
      <td>48239</td>
      <td>Rona</td>
      <td>NaN</td>
      <td>Annex</td>
      <td>43.66724</td>
      <td>-79.41598</td>
      <td>Private room</td>
      <td>66</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>43.66724, -79.41598</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23691</td>
      <td>Queen Bedroom close to downtown</td>
      <td>93825</td>
      <td>Yohan &amp; Sarah</td>
      <td>NaN</td>
      <td>Briar Hill-Belgravia</td>
      <td>43.69602</td>
      <td>-79.45468</td>
      <td>Private room</td>
      <td>70</td>
      <td>1</td>
      <td>217</td>
      <td>2019-12-22</td>
      <td>1.72</td>
      <td>2</td>
      <td>240</td>
      <td>43.696020000000004, -79.45468000000001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26654</td>
      <td>World Class downtown @CN Tower Theatre MTCC ga...</td>
      <td>113345</td>
      <td>Adela</td>
      <td>NaN</td>
      <td>Waterfront Communities-The Island</td>
      <td>43.64530</td>
      <td>-79.38940</td>
      <td>Entire home/apt</td>
      <td>125</td>
      <td>21</td>
      <td>40</td>
      <td>2020-03-20</td>
      <td>0.34</td>
      <td>2</td>
      <td>295</td>
      <td>43.6453, -79.3894</td>
    </tr>
    <tr>
      <th>5</th>
      <td>27423</td>
      <td>Executive Studio Unit- Ideal for One Person</td>
      <td>118124</td>
      <td>Brent</td>
      <td>NaN</td>
      <td>Greenwood-Coxwell</td>
      <td>43.66890</td>
      <td>-79.32592</td>
      <td>Entire home/apt</td>
      <td>54</td>
      <td>120</td>
      <td>26</td>
      <td>2011-08-30</td>
      <td>0.21</td>
      <td>1</td>
      <td>0</td>
      <td>43.6689, -79.32592</td>
    </tr>
    <tr>
      <th>6</th>
      <td>28160</td>
      <td>Luxury, Safety, Affordability For Women Travel...</td>
      <td>86838</td>
      <td>Rita</td>
      <td>NaN</td>
      <td>Mount Pleasant West</td>
      <td>43.70376</td>
      <td>-79.39077</td>
      <td>Entire home/apt</td>
      <td>50</td>
      <td>60</td>
      <td>7</td>
      <td>2018-10-17</td>
      <td>0.11</td>
      <td>1</td>
      <td>364</td>
      <td>43.703759999999996, -79.39076999999999</td>
    </tr>
    <tr>
      <th>7</th>
      <td>30931</td>
      <td>Downtown Toronto - Waterview Condo</td>
      <td>22795</td>
      <td>Kathie &amp; Larry</td>
      <td>NaN</td>
      <td>Waterfront Communities-The Island</td>
      <td>43.64151</td>
      <td>-79.37643</td>
      <td>Entire home/apt</td>
      <td>131</td>
      <td>180</td>
      <td>1</td>
      <td>2010-08-11</td>
      <td>0.01</td>
      <td>2</td>
      <td>365</td>
      <td>43.64151, -79.37643</td>
    </tr>
    <tr>
      <th>8</th>
      <td>40456</td>
      <td>Downtown 2  Bdr.Apt with King Size Bed and Par...</td>
      <td>174063</td>
      <td>Denis</td>
      <td>NaN</td>
      <td>South Parkdale</td>
      <td>43.63532</td>
      <td>-79.44049</td>
      <td>Entire home/apt</td>
      <td>100</td>
      <td>30</td>
      <td>110</td>
      <td>2020-03-25</td>
      <td>0.88</td>
      <td>5</td>
      <td>359</td>
      <td>43.63532, -79.44049</td>
    </tr>
    <tr>
      <th>9</th>
      <td>41887</td>
      <td>Great location</td>
      <td>183071</td>
      <td>Kyle</td>
      <td>NaN</td>
      <td>Oakridge</td>
      <td>43.69466</td>
      <td>-79.28667</td>
      <td>Entire home/apt</td>
      <td>75</td>
      <td>28</td>
      <td>82</td>
      <td>2019-09-02</td>
      <td>1.74</td>
      <td>2</td>
      <td>342</td>
      <td>43.69466, -79.28667</td>
    </tr>
    <tr>
      <th>10</th>
      <td>42892</td>
      <td>Downtown 3 Beds 2 Baths @ Union &amp; Harbourfront</td>
      <td>187320</td>
      <td>Downtown Suite Living</td>
      <td>NaN</td>
      <td>Waterfront Communities-The Island</td>
      <td>43.64451</td>
      <td>-79.38185</td>
      <td>Entire home/apt</td>
      <td>110</td>
      <td>29</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13</td>
      <td>265</td>
      <td>43.64451, -79.38185</td>
    </tr>
    <tr>
      <th>11</th>
      <td>43964</td>
      <td>Bright entire 2-bedrm basement suite private e...</td>
      <td>192364</td>
      <td>Mitra</td>
      <td>NaN</td>
      <td>Wexford/Maryvale</td>
      <td>43.74948</td>
      <td>-79.29147</td>
      <td>Private room</td>
      <td>93</td>
      <td>2</td>
      <td>31</td>
      <td>2020-09-28</td>
      <td>0.68</td>
      <td>1</td>
      <td>363</td>
      <td>43.74948, -79.29146999999999</td>
    </tr>
    <tr>
      <th>12</th>
      <td>44452</td>
      <td>Yonge &amp; Bloor Studio Skyline</td>
      <td>195095</td>
      <td>Urbano</td>
      <td>NaN</td>
      <td>Rosedale-Moore Park</td>
      <td>43.67185</td>
      <td>-79.38583</td>
      <td>Entire home/apt</td>
      <td>107</td>
      <td>1</td>
      <td>55</td>
      <td>2020-09-10</td>
      <td>0.45</td>
      <td>11</td>
      <td>365</td>
      <td>43.67185, -79.38583</td>
    </tr>
    <tr>
      <th>13</th>
      <td>45399</td>
      <td>Fountain View Studio - Eaton center</td>
      <td>195095</td>
      <td>Urbano</td>
      <td>NaN</td>
      <td>Church-Yonge Corridor</td>
      <td>43.65972</td>
      <td>-79.38172</td>
      <td>Entire home/apt</td>
      <td>120</td>
      <td>1</td>
      <td>78</td>
      <td>2019-11-07</td>
      <td>0.64</td>
      <td>11</td>
      <td>365</td>
      <td>43.65972, -79.38172</td>
    </tr>
    <tr>
      <th>14</th>
      <td>45601</td>
      <td>Marigold Gardens</td>
      <td>188183</td>
      <td>Simon</td>
      <td>NaN</td>
      <td>South Riverdale</td>
      <td>43.66274</td>
      <td>-79.33096</td>
      <td>Private room</td>
      <td>50</td>
      <td>2</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>365</td>
      <td>43.66274, -79.33096</td>
    </tr>
    <tr>
      <th>15</th>
      <td>45893</td>
      <td>Yonge &amp; Bloor Lakeview Master BR</td>
      <td>195095</td>
      <td>Urbano</td>
      <td>NaN</td>
      <td>Rosedale-Moore Park</td>
      <td>43.67105</td>
      <td>-79.38481</td>
      <td>Private room</td>
      <td>86</td>
      <td>1</td>
      <td>18</td>
      <td>2019-06-01</td>
      <td>0.15</td>
      <td>11</td>
      <td>282</td>
      <td>43.67105, -79.38481</td>
    </tr>
    <tr>
      <th>16</th>
      <td>50110</td>
      <td>Yorkville one bedroom Condo</td>
      <td>195095</td>
      <td>Urbano</td>
      <td>NaN</td>
      <td>Church-Yonge Corridor</td>
      <td>43.66899</td>
      <td>-79.38548</td>
      <td>Entire home/apt</td>
      <td>126</td>
      <td>2</td>
      <td>56</td>
      <td>2020-02-08</td>
      <td>0.46</td>
      <td>11</td>
      <td>251</td>
      <td>43.66899, -79.38548</td>
    </tr>
    <tr>
      <th>17</th>
      <td>51616</td>
      <td>Large room in trendy King Street West.</td>
      <td>237587</td>
      <td>John</td>
      <td>NaN</td>
      <td>Niagara</td>
      <td>43.64296</td>
      <td>-79.40451</td>
      <td>Private room</td>
      <td>56</td>
      <td>91</td>
      <td>24</td>
      <td>2018-11-11</td>
      <td>0.47</td>
      <td>1</td>
      <td>179</td>
      <td>43.642959999999995, -79.40451</td>
    </tr>
    <tr>
      <th>18</th>
      <td>57483</td>
      <td>Downtown 3 Beds 2 Baths @ Union &amp; Harbour</td>
      <td>187320</td>
      <td>Downtown Suite Living</td>
      <td>NaN</td>
      <td>Waterfront Communities-The Island</td>
      <td>43.64286</td>
      <td>-79.38073</td>
      <td>Entire home/apt</td>
      <td>110</td>
      <td>29</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13</td>
      <td>195</td>
      <td>43.64286, -79.38073</td>
    </tr>
    <tr>
      <th>19</th>
      <td>64641</td>
      <td>Furnished Heritage Rooms Downtown</td>
      <td>304551</td>
      <td>Kintoo</td>
      <td>NaN</td>
      <td>Niagara</td>
      <td>43.64440</td>
      <td>-79.40829</td>
      <td>Private room</td>
      <td>65</td>
      <td>30</td>
      <td>16</td>
      <td>2019-08-04</td>
      <td>0.13</td>
      <td>4</td>
      <td>343</td>
      <td>43.6444, -79.40829000000001</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(['neighbourhood_group', 'name', 'host_name'], 1, inplace = True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>neighbourhood</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
      <th>location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1419</td>
      <td>1565</td>
      <td>Little Portugal</td>
      <td>43.64617</td>
      <td>-79.42451</td>
      <td>Entire home/apt</td>
      <td>469</td>
      <td>4</td>
      <td>7</td>
      <td>2017-12-04</td>
      <td>0.11</td>
      <td>1</td>
      <td>0</td>
      <td>43.64617, -79.42451</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8077</td>
      <td>22795</td>
      <td>Waterfront Communities-The Island</td>
      <td>43.64105</td>
      <td>-79.37628</td>
      <td>Private room</td>
      <td>98</td>
      <td>180</td>
      <td>169</td>
      <td>2013-08-27</td>
      <td>1.24</td>
      <td>2</td>
      <td>365</td>
      <td>43.64105, -79.37628000000001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12604</td>
      <td>48239</td>
      <td>Annex</td>
      <td>43.66724</td>
      <td>-79.41598</td>
      <td>Private room</td>
      <td>66</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>43.66724, -79.41598</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23691</td>
      <td>93825</td>
      <td>Briar Hill-Belgravia</td>
      <td>43.69602</td>
      <td>-79.45468</td>
      <td>Private room</td>
      <td>70</td>
      <td>1</td>
      <td>217</td>
      <td>2019-12-22</td>
      <td>1.72</td>
      <td>2</td>
      <td>240</td>
      <td>43.696020000000004, -79.45468000000001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26654</td>
      <td>113345</td>
      <td>Waterfront Communities-The Island</td>
      <td>43.64530</td>
      <td>-79.38940</td>
      <td>Entire home/apt</td>
      <td>125</td>
      <td>21</td>
      <td>40</td>
      <td>2020-03-20</td>
      <td>0.34</td>
      <td>2</td>
      <td>295</td>
      <td>43.6453, -79.3894</td>
    </tr>
  </tbody>
</table>
</div>





# taking care of missing data


```python
df = df[['id', 'host_id', 'neighbourhood', 'latitude', 'longitude', 'room_type','minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'location', 'price']]
cols = list(df.columns.values)
print(cols)

```

    ['id', 'host_id', 'neighbourhood', 'latitude', 'longitude', 'room_type', 'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'location', 'price']



```python
# count the number of missing values for each column
num_missing = (df[['id', 'host_id', 'neighbourhood', 'latitude', 'longitude', 'room_type', 'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'price']] == 0).sum()
# report the results
print(num_missing)
```

    id                                   0
    host_id                              0
    neighbourhood                        0
    latitude                             0
    longitude                            0
    room_type                            0
    minimum_nights                       0
    number_of_reviews                 3958
    last_review                          0
    reviews_per_month                    0
    calculated_host_listings_count       0
    availability_365                  6641
    price                                0
    dtype: int64



```python
# replace '0' values with 'nan'
from numpy import nan
df[['number_of_reviews', 'availability_365']] = df[['number_of_reviews', 'availability_365']].replace(0, nan)
# count the number of nan values in each column
print(df.isnull().sum())
```

    id                                   0
    host_id                              0
    neighbourhood                        0
    latitude                             0
    longitude                            0
    room_type                            0
    minimum_nights                       0
    number_of_reviews                 3958
    last_review                       3958
    reviews_per_month                 3958
    calculated_host_listings_count       0
    availability_365                  6641
    location                             0
    price                                0
    dtype: int64



```python
#conversion of the 'REF_DATE' from a string to a proper datetime object.
df['last_review'] = pd.to_datetime(df['last_review']) 
# fill missing values with mean column values
df.fillna(df.mean(), inplace=True)
# count the number of NaN values in each column
print(df.isnull().sum())


```

    <ipython-input-10-a497e4579ca7>:4: FutureWarning: DataFrame.mean and DataFrame.median with numeric_only=None will include datetime64 and datetime64tz columns in a future version.
      df.fillna(df.mean(), inplace=True)


    id                                   0
    host_id                              0
    neighbourhood                        0
    latitude                             0
    longitude                            0
    room_type                            0
    minimum_nights                       0
    number_of_reviews                    0
    last_review                       3958
    reviews_per_month                    0
    calculated_host_listings_count       0
    availability_365                     0
    location                             0
    price                                0
    dtype: int64


# Encoding categorical Data


```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
X=df.iloc[: , 5:12].values
y=df.iloc[:, 13].values
one_hot = pd.get_dummies(df['room_type'])
print(one_hot)
```

           Entire home/apt  Hotel room  Private room  Shared room
    0                    1           0             0            0
    1                    0           0             1            0
    2                    0           0             1            0
    3                    0           0             1            0
    4                    1           0             0            0
    ...                ...         ...           ...          ...
    18741                1           0             0            0
    18742                1           0             0            0
    18743                1           0             0            0
    18744                0           0             1            0
    18745                1           0             0            0
    
    [18746 rows x 4 columns]



```python
merged_df = pd.concat([df, one_hot], axis=1)
print(merged_df)
```

                 id    host_id                      neighbourhood  latitude  \
    0          1419       1565                    Little Portugal  43.64617   
    1          8077      22795  Waterfront Communities-The Island  43.64105   
    2         12604      48239                              Annex  43.66724   
    3         23691      93825               Briar Hill-Belgravia  43.69602   
    4         26654     113345  Waterfront Communities-The Island  43.64530   
    ...         ...        ...                                ...       ...   
    18741  45785334    1023135                     Yonge-Eglinton  43.70703   
    18742  45786222  371188170                   Newtonbrook East  43.79075   
    18743  45786961  371195218                    Bayview Village  43.76668   
    18744  45789233  304805850             Corso Italia-Davenport  43.67330   
    18745  45789708  344709977                            Niagara  43.64481   
    
           longitude        room_type  minimum_nights  number_of_reviews  \
    0      -79.42451  Entire home/apt               4           7.000000   
    1      -79.37628     Private room             180         169.000000   
    2      -79.41598     Private room               1          33.764606   
    3      -79.45468     Private room               1         217.000000   
    4      -79.38940  Entire home/apt              21          40.000000   
    ...          ...              ...             ...                ...   
    18741  -79.40052  Entire home/apt              90          33.764606   
    18742  -79.39699  Entire home/apt              90          33.764606   
    18743  -79.37158  Entire home/apt               1          33.764606   
    18744  -79.44542     Private room               1          33.764606   
    18745  -79.40677  Entire home/apt               1          33.764606   
    
          last_review  reviews_per_month  calculated_host_listings_count  \
    0      2017-12-04           0.110000                               1   
    1      2013-08-27           1.240000                               2   
    2             NaT           1.214266                               1   
    3      2019-12-22           1.720000                               2   
    4      2020-03-20           0.340000                               2   
    ...           ...                ...                             ...   
    18741         NaT           1.214266                               1   
    18742         NaT           1.214266                               1   
    18743         NaT           1.214266                               1   
    18744         NaT           1.214266                               2   
    18745         NaT           1.214266                               1   
    
           availability_365                                location  price  \
    0            194.646758                     43.64617, -79.42451    469   
    1            365.000000            43.64105, -79.37628000000001     98   
    2            194.646758                     43.66724, -79.41598     66   
    3            240.000000  43.696020000000004, -79.45468000000001     70   
    4            295.000000                       43.6453, -79.3894    125   
    ...                 ...                                     ...    ...   
    18741        344.000000           43.707029999999996, -79.40052     65   
    18742        179.000000                     43.79075, -79.39699    158   
    18743        194.646758            43.76668, -79.37158000000001     70   
    18744         52.000000                      43.6733, -79.44542     39   
    18745         33.000000                     43.64481, -79.40677     52   
    
           Entire home/apt  Hotel room  Private room  Shared room  
    0                    1           0             0            0  
    1                    0           0             1            0  
    2                    0           0             1            0  
    3                    0           0             1            0  
    4                    1           0             0            0  
    ...                ...         ...           ...          ...  
    18741                1           0             0            0  
    18742                1           0             0            0  
    18743                1           0             0            0  
    18744                0           0             1            0  
    18745                1           0             0            0  
    
    [18746 rows x 18 columns]



```python
#merged_df.drop(['room_type'], 1, inplace = True)
merged_df.columns
```




    Index(['id', 'host_id', 'neighbourhood', 'latitude', 'longitude', 'room_type',
           'minimum_nights', 'number_of_reviews', 'last_review',
           'reviews_per_month', 'calculated_host_listings_count',
           'availability_365', 'location', 'price', 'Entire home/apt',
           'Hotel room', 'Private room', 'Shared room'],
          dtype='object')




```python
merged_df = merged_df[['id', 'host_id', 'neighbourhood','minimum_nights', 'number_of_reviews',
       'calculated_host_listings_count', 'availability_365', 'reviews_per_month','latitude', 'longitude','Shared room', 'Entire home/apt',
       'Hotel room', 'Private room', 'price']]

```


```python
#merged_df.drop(['last_review'], 1, inplace = True)
#merged_df.drop(['location'], 1, inplace = True)
merged_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>neighbourhood</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
      <th>reviews_per_month</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>Shared room</th>
      <th>Entire home/apt</th>
      <th>Hotel room</th>
      <th>Private room</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1419</td>
      <td>1565</td>
      <td>Little Portugal</td>
      <td>4</td>
      <td>7.000000</td>
      <td>1</td>
      <td>194.646758</td>
      <td>0.110000</td>
      <td>43.64617</td>
      <td>-79.42451</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>469</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8077</td>
      <td>22795</td>
      <td>Waterfront Communities-The Island</td>
      <td>180</td>
      <td>169.000000</td>
      <td>2</td>
      <td>365.000000</td>
      <td>1.240000</td>
      <td>43.64105</td>
      <td>-79.37628</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12604</td>
      <td>48239</td>
      <td>Annex</td>
      <td>1</td>
      <td>33.764606</td>
      <td>1</td>
      <td>194.646758</td>
      <td>1.214266</td>
      <td>43.66724</td>
      <td>-79.41598</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23691</td>
      <td>93825</td>
      <td>Briar Hill-Belgravia</td>
      <td>1</td>
      <td>217.000000</td>
      <td>2</td>
      <td>240.000000</td>
      <td>1.720000</td>
      <td>43.69602</td>
      <td>-79.45468</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>70</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26654</td>
      <td>113345</td>
      <td>Waterfront Communities-The Island</td>
      <td>21</td>
      <td>40.000000</td>
      <td>2</td>
      <td>295.000000</td>
      <td>0.340000</td>
      <td>43.64530</td>
      <td>-79.38940</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>125</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Encoding neighbourhood
one_hot = pd.get_dummies(df['neighbourhood'])
print(one_hot)
```

           Agincourt North  Agincourt South-Malvern West  Alderwood  Annex  \
    0                    0                             0          0      0   
    1                    0                             0          0      0   
    2                    0                             0          0      1   
    3                    0                             0          0      0   
    4                    0                             0          0      0   
    ...                ...                           ...        ...    ...   
    18741                0                             0          0      0   
    18742                0                             0          0      0   
    18743                0                             0          0      0   
    18744                0                             0          0      0   
    18745                0                             0          0      0   
    
           Banbury-Don Mills  Bathurst Manor  Bay Street Corridor  \
    0                      0               0                    0   
    1                      0               0                    0   
    2                      0               0                    0   
    3                      0               0                    0   
    4                      0               0                    0   
    ...                  ...             ...                  ...   
    18741                  0               0                    0   
    18742                  0               0                    0   
    18743                  0               0                    0   
    18744                  0               0                    0   
    18745                  0               0                    0   
    
           Bayview Village  Bayview Woods-Steeles  Bedford Park-Nortown  ...  \
    0                    0                      0                     0  ...   
    1                    0                      0                     0  ...   
    2                    0                      0                     0  ...   
    3                    0                      0                     0  ...   
    4                    0                      0                     0  ...   
    ...                ...                    ...                   ...  ...   
    18741                0                      0                     0  ...   
    18742                0                      0                     0  ...   
    18743                1                      0                     0  ...   
    18744                0                      0                     0  ...   
    18745                0                      0                     0  ...   
    
           Willowdale West  Willowridge-Martingrove-Richview  Woburn  \
    0                    0                                 0       0   
    1                    0                                 0       0   
    2                    0                                 0       0   
    3                    0                                 0       0   
    4                    0                                 0       0   
    ...                ...                               ...     ...   
    18741                0                                 0       0   
    18742                0                                 0       0   
    18743                0                                 0       0   
    18744                0                                 0       0   
    18745                0                                 0       0   
    
           Woodbine Corridor  Woodbine-Lumsden  Wychwood  Yonge-Eglinton  \
    0                      0                 0         0               0   
    1                      0                 0         0               0   
    2                      0                 0         0               0   
    3                      0                 0         0               0   
    4                      0                 0         0               0   
    ...                  ...               ...       ...             ...   
    18741                  0                 0         0               1   
    18742                  0                 0         0               0   
    18743                  0                 0         0               0   
    18744                  0                 0         0               0   
    18745                  0                 0         0               0   
    
           Yonge-St.Clair  York University Heights  Yorkdale-Glen Park  
    0                   0                        0                   0  
    1                   0                        0                   0  
    2                   0                        0                   0  
    3                   0                        0                   0  
    4                   0                        0                   0  
    ...               ...                      ...                 ...  
    18741               0                        0                   0  
    18742               0                        0                   0  
    18743               0                        0                   0  
    18744               0                        0                   0  
    18745               0                        0                   0  
    
    [18746 rows x 140 columns]



```python
merged2_df = pd.concat([merged_df, one_hot], axis=1)
print(merged2_df)
```

                 id    host_id                      neighbourhood  minimum_nights  \
    0          1419       1565                    Little Portugal               4   
    1          8077      22795  Waterfront Communities-The Island             180   
    2         12604      48239                              Annex               1   
    3         23691      93825               Briar Hill-Belgravia               1   
    4         26654     113345  Waterfront Communities-The Island              21   
    ...         ...        ...                                ...             ...   
    18741  45785334    1023135                     Yonge-Eglinton              90   
    18742  45786222  371188170                   Newtonbrook East              90   
    18743  45786961  371195218                    Bayview Village               1   
    18744  45789233  304805850             Corso Italia-Davenport               1   
    18745  45789708  344709977                            Niagara               1   
    
           number_of_reviews  calculated_host_listings_count  availability_365  \
    0               7.000000                               1        194.646758   
    1             169.000000                               2        365.000000   
    2              33.764606                               1        194.646758   
    3             217.000000                               2        240.000000   
    4              40.000000                               2        295.000000   
    ...                  ...                             ...               ...   
    18741          33.764606                               1        344.000000   
    18742          33.764606                               1        179.000000   
    18743          33.764606                               1        194.646758   
    18744          33.764606                               2         52.000000   
    18745          33.764606                               1         33.000000   
    
           reviews_per_month  latitude  longitude  ...  Willowdale West  \
    0               0.110000  43.64617  -79.42451  ...                0   
    1               1.240000  43.64105  -79.37628  ...                0   
    2               1.214266  43.66724  -79.41598  ...                0   
    3               1.720000  43.69602  -79.45468  ...                0   
    4               0.340000  43.64530  -79.38940  ...                0   
    ...                  ...       ...        ...  ...              ...   
    18741           1.214266  43.70703  -79.40052  ...                0   
    18742           1.214266  43.79075  -79.39699  ...                0   
    18743           1.214266  43.76668  -79.37158  ...                0   
    18744           1.214266  43.67330  -79.44542  ...                0   
    18745           1.214266  43.64481  -79.40677  ...                0   
    
           Willowridge-Martingrove-Richview  Woburn  Woodbine Corridor  \
    0                                     0       0                  0   
    1                                     0       0                  0   
    2                                     0       0                  0   
    3                                     0       0                  0   
    4                                     0       0                  0   
    ...                                 ...     ...                ...   
    18741                                 0       0                  0   
    18742                                 0       0                  0   
    18743                                 0       0                  0   
    18744                                 0       0                  0   
    18745                                 0       0                  0   
    
           Woodbine-Lumsden  Wychwood  Yonge-Eglinton  Yonge-St.Clair  \
    0                     0         0               0               0   
    1                     0         0               0               0   
    2                     0         0               0               0   
    3                     0         0               0               0   
    4                     0         0               0               0   
    ...                 ...       ...             ...             ...   
    18741                 0         0               1               0   
    18742                 0         0               0               0   
    18743                 0         0               0               0   
    18744                 0         0               0               0   
    18745                 0         0               0               0   
    
           York University Heights  Yorkdale-Glen Park  
    0                            0                   0  
    1                            0                   0  
    2                            0                   0  
    3                            0                   0  
    4                            0                   0  
    ...                        ...                 ...  
    18741                        0                   0  
    18742                        0                   0  
    18743                        0                   0  
    18744                        0                   0  
    18745                        0                   0  
    
    [18746 rows x 155 columns]



```python
merged2_df["reviews_per_month"].mean()
```




    1.2142662969974656




```python
selected_df = merged2_df[merged2_df['reviews_per_month'] >= 1.2] 

```


```python
#merged2_df.drop(['neighbourhood'], 1, inplace = True)
selected_df.columns
```




    Index(['id', 'host_id', 'neighbourhood', 'minimum_nights', 'number_of_reviews',
           'calculated_host_listings_count', 'availability_365',
           'reviews_per_month', 'latitude', 'longitude',
           ...
           'Willowdale West', 'Willowridge-Martingrove-Richview', 'Woburn',
           'Woodbine Corridor', 'Woodbine-Lumsden', 'Wychwood', 'Yonge-Eglinton',
           'Yonge-St.Clair', 'York University Heights', 'Yorkdale-Glen Park'],
          dtype='object', length=155)




```python
def movecol(df, cols_to_move=[], ref_col='', place='After'):
    
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3])
selected_df = movecol(selected_df, 
             cols_to_move=['Yorkdale-Glen Park','price'], 
             ref_col='York University Heights',
             place='After')
selected_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>neighbourhood</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
      <th>reviews_per_month</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>...</th>
      <th>Willowridge-Martingrove-Richview</th>
      <th>Woburn</th>
      <th>Woodbine Corridor</th>
      <th>Woodbine-Lumsden</th>
      <th>Wychwood</th>
      <th>Yonge-Eglinton</th>
      <th>Yonge-St.Clair</th>
      <th>York University Heights</th>
      <th>Yorkdale-Glen Park</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>8077</td>
      <td>22795</td>
      <td>Waterfront Communities-The Island</td>
      <td>180</td>
      <td>169.000000</td>
      <td>2</td>
      <td>365.000000</td>
      <td>1.240000</td>
      <td>43.64105</td>
      <td>-79.37628</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12604</td>
      <td>48239</td>
      <td>Annex</td>
      <td>1</td>
      <td>33.764606</td>
      <td>1</td>
      <td>194.646758</td>
      <td>1.214266</td>
      <td>43.66724</td>
      <td>-79.41598</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23691</td>
      <td>93825</td>
      <td>Briar Hill-Belgravia</td>
      <td>1</td>
      <td>217.000000</td>
      <td>2</td>
      <td>240.000000</td>
      <td>1.720000</td>
      <td>43.69602</td>
      <td>-79.45468</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>70</td>
    </tr>
    <tr>
      <th>9</th>
      <td>41887</td>
      <td>183071</td>
      <td>Oakridge</td>
      <td>28</td>
      <td>82.000000</td>
      <td>2</td>
      <td>342.000000</td>
      <td>1.740000</td>
      <td>43.69466</td>
      <td>-79.28667</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
    </tr>
    <tr>
      <th>10</th>
      <td>42892</td>
      <td>187320</td>
      <td>Waterfront Communities-The Island</td>
      <td>29</td>
      <td>33.764606</td>
      <td>13</td>
      <td>265.000000</td>
      <td>1.214266</td>
      <td>43.64451</td>
      <td>-79.38185</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>110</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18741</th>
      <td>45785334</td>
      <td>1023135</td>
      <td>Yonge-Eglinton</td>
      <td>90</td>
      <td>33.764606</td>
      <td>1</td>
      <td>344.000000</td>
      <td>1.214266</td>
      <td>43.70703</td>
      <td>-79.40052</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>65</td>
    </tr>
    <tr>
      <th>18742</th>
      <td>45786222</td>
      <td>371188170</td>
      <td>Newtonbrook East</td>
      <td>90</td>
      <td>33.764606</td>
      <td>1</td>
      <td>179.000000</td>
      <td>1.214266</td>
      <td>43.79075</td>
      <td>-79.39699</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>158</td>
    </tr>
    <tr>
      <th>18743</th>
      <td>45786961</td>
      <td>371195218</td>
      <td>Bayview Village</td>
      <td>1</td>
      <td>33.764606</td>
      <td>1</td>
      <td>194.646758</td>
      <td>1.214266</td>
      <td>43.76668</td>
      <td>-79.37158</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>70</td>
    </tr>
    <tr>
      <th>18744</th>
      <td>45789233</td>
      <td>304805850</td>
      <td>Corso Italia-Davenport</td>
      <td>1</td>
      <td>33.764606</td>
      <td>2</td>
      <td>52.000000</td>
      <td>1.214266</td>
      <td>43.67330</td>
      <td>-79.44542</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>18745</th>
      <td>45789708</td>
      <td>344709977</td>
      <td>Niagara</td>
      <td>1</td>
      <td>33.764606</td>
      <td>1</td>
      <td>33.000000</td>
      <td>1.214266</td>
      <td>43.64481</td>
      <td>-79.40677</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>52</td>
    </tr>
  </tbody>
</table>
<p>8657 rows × 155 columns</p>
</div>



# Decision Tree Classifier

In the next lines, I am going to categorize the prices as low, affordable and expensive. Then then the model willl be trained sperately on each class.10 columns were selected to start with building our model and see how it works.


```python
#define dataset
X=selected_df.iloc[: , 7:18].values
y_low = selected_df.iloc[: , -1][selected_df[: , -1] < 100] 

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_low, test_size = 0.6, random_state = 0)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-27-766aefa052c5> in <module>
          1 #define dataset
          2 X=selected_df.iloc[: , 7:18].values
    ----> 3 y_low = selected_df.iloc[: , -1][selected_df[: , -1] < 100]
          4 
          5 #Splitting the dataset into the Training set and Test set


    /opt/conda/lib/python3.8/site-packages/pandas/core/frame.py in __getitem__(self, key)
       2897             if self.columns.nlevels > 1:
       2898                 return self._getitem_multilevel(key)
    -> 2899             indexer = self.columns.get_loc(key)
       2900             if is_integer(indexer):
       2901                 indexer = [indexer]


    /opt/conda/lib/python3.8/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2887             casted_key = self._maybe_cast_indexer(key)
       2888             try:
    -> 2889                 return self._engine.get_loc(casted_key)
       2890             except KeyError as err:
       2891                 raise KeyError(key) from err


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    TypeError: '(slice(None, None, None), -1)' is an invalid key



```python
y_train.shape
```




    (3462,)




```python
 if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    #If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    
    #If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    #the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    #the mode target feature value is stored in the parent_node_class variable.
    
    elif len(features) ==0:
        return parent_node_class
    
    #If none of the above holds true, grow the tree!
    
    else:
        #Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        #Select the feature which best splits the dataset
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-77-ff24a68a94b5> in <module>
          1 for i in range(len(y_train)):
          2 
    ----> 3     for price in y_train[i]:
          4         if price <100:
          5             y_train[i][1]=price


    TypeError: 'numpy.int64' object is not iterable



```python
from sklearn.tree import DecisionTreeClassifier
clfLow = DecisionTreeClassifier(random_state=0)


```


      File "<ipython-input-72-34853c5439e5>", line 3
        for i in range(len(y_train)):
        ^
    IndentationError: unexpected indent




```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
X1=selected_df.iloc[: , 7:-1].values
plt.hist(X1, bins=144)
plt.ylabel('Price')
plt.show()
```


![png](output_34_0.png)


# Decision Tree regression 


```python
#importing library
from sklearn.tree import DecisionTreeRegressor
#define dataset
X=selected_df.iloc[: , 7:-1].values
y=selected_df.iloc[:, -1].values
#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
#define the model
Regressor = DecisionTreeRegressor()
```

## Predicting the test dataset


```python
Regressor.fit(X_train, y_train)

```




    DecisionTreeRegressor()




```python
y_pred = Regressor.predict(X_test)
#Evaluation=np.concatenate((y_pred, y_test))
#np.vstack([y_pred, y_test])
#print(Evaluation.size)
prediction_dict = {"y_pred":y_pred, "y_test":y_test}
predictions_df = pd.DataFrame(prediction_dict)
print(predictions_df)
```

          y_pred  y_test
    0      350.0     169
    1      199.0     153
    2      117.0      73
    3       45.0      51
    4       75.0      51
    ...      ...     ...
    3458    84.0      80
    3459    30.0      30
    3460    38.0      90
    3461    45.0      70
    3462   115.0     110
    
    [3463 rows x 2 columns]


# Subtracting the predicted result from X test


```python
Sub_df = predictions_df['y_pred'] - predictions_df['y_test']
Index = np.arange(len(Sub_df))
Price_DiffDic = {"Index":Index,"y_pred" : predictions_df['y_pred'], "y_test" : predictions_df['y_test'] ,"Price_Difference":Sub_df}
priceDiff_df = pd.DataFrame(Price_DiffDic)
print(priceDiff_df)
```

          Index  y_pred  y_test  Price_Difference
    0         0   350.0     169             181.0
    1         1   199.0     153              46.0
    2         2   117.0      73              44.0
    3         3    45.0      51              -6.0
    4         4    75.0      51              24.0
    ...     ...     ...     ...               ...
    3458   3458    84.0      80               4.0
    3459   3459    30.0      30               0.0
    3460   3460    38.0      90             -52.0
    3461   3461    45.0      70             -25.0
    3462   3462   115.0     110               5.0
    
    [3463 rows x 4 columns]



```python
#plotting the subtraction
%matplotlib inline

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = priceDiff_df.iloc[: , 0].values
Price_Diff = priceDiff_df.iloc[: , 1].values
ax.bar(langs,Price_Diff)
plt.show()

```


![png](output_42_0.png)



```python
print(Regressor.feature_importances_)
```

    [2.89302815e-02 7.35085832e-01 1.54929270e-01 2.36924121e-04
     1.76304452e-02 3.93559545e-05 6.12140585e-03 1.73679862e-05
     1.14683891e-04 1.10227023e-07 1.05028482e-02 1.88909747e-04
     4.47023735e-05 8.37538704e-04 4.71074124e-05 7.80473630e-03
     2.50161170e-04 3.79645018e-05 5.84788906e-06 5.42831011e-06
     3.13869758e-04 4.29037141e-06 7.17843240e-06 2.65901970e-04
     3.26598587e-08 5.73501195e-06 1.40025655e-03 1.67917892e-06
     1.02160378e-05 1.31950932e-05 2.10698615e-05 2.52952527e-05
     4.56018962e-03 2.34498239e-05 1.32509212e-05 3.02113899e-06
     3.93671668e-05 2.04379272e-06 3.59742947e-03 1.22182106e-04
     2.03674951e-05 2.09629089e-04 1.42736267e-05 3.69354654e-04
     7.65747422e-06 2.31966727e-06 5.52044907e-04 8.23973304e-06
     5.15415990e-04 0.00000000e+00 5.67168579e-06 2.22963995e-03
     5.27320636e-08 7.03430091e-05 3.67580183e-05 5.63039830e-04
     1.65700049e-03 4.52512246e-04 3.39526448e-07 1.76567361e-07
     6.27681660e-08 4.23957720e-06 2.81345242e-06 1.87447603e-05
     9.18558526e-09 1.21509854e-05 2.86697571e-05 1.99274468e-06
     1.02673154e-04 1.16282938e-04 3.59394529e-06 1.23863725e-05
     7.10502925e-05 1.84694903e-06 1.25500270e-05 2.86472812e-03
     2.55155146e-08 9.84702476e-04 2.57006705e-05 6.79052896e-07
     1.80523606e-06 2.01423446e-04 2.55155146e-08 0.00000000e+00
     7.31190989e-06 3.35110703e-04 8.03483555e-06 8.42960089e-06
     7.00656031e-07 1.18482639e-04 6.29285013e-05 3.22227696e-04
     5.09922090e-05 2.39165424e-07 5.09551202e-05 2.19799488e-05
     7.95938253e-07 3.27848294e-06 2.20005101e-05 1.21970964e-06
     5.54183992e-05 1.90730941e-04 1.88682362e-04 2.40429989e-06
     3.40747069e-06 2.13300633e-05 0.00000000e+00 7.34026657e-04
     1.95244903e-06 1.18535774e-06 1.42294604e-04 1.83654904e-03
     6.86265281e-07 6.80900144e-05 2.27548313e-06 3.12437477e-06
     3.40719162e-05 5.18347897e-04 9.44447305e-04 5.80222802e-07
     4.54458532e-06 4.69958114e-06 1.52227776e-04 6.34512872e-04
     5.45508173e-06 0.00000000e+00 7.45363377e-05 9.07080963e-04
     2.17688590e-05 4.87100411e-03 0.00000000e+00 2.53668009e-05
     4.88386390e-06 2.86057548e-04 2.62607330e-06 2.90356510e-04
     3.50126266e-04 5.74557156e-05 0.00000000e+00 8.81093975e-06
     9.14782497e-05 0.00000000e+00 9.89171852e-05 2.31406439e-05
     1.85034742e-03 3.08654139e-06 2.36543160e-05]



```python
print(Regressor.feature_importances_)
```

    [2.89302815e-02 7.35085832e-01 1.54929270e-01 2.36924121e-04
     1.76304452e-02 3.93559545e-05 6.12140585e-03 1.73679862e-05
     1.14683891e-04 1.10227023e-07 1.05028482e-02 1.88909747e-04
     4.47023735e-05 8.37538704e-04 4.71074124e-05 7.80473630e-03
     2.50161170e-04 3.79645018e-05 5.84788906e-06 5.42831011e-06
     3.13869758e-04 4.29037141e-06 7.17843240e-06 2.65901970e-04
     3.26598587e-08 5.73501195e-06 1.40025655e-03 1.67917892e-06
     1.02160378e-05 1.31950932e-05 2.10698615e-05 2.52952527e-05
     4.56018962e-03 2.34498239e-05 1.32509212e-05 3.02113899e-06
     3.93671668e-05 2.04379272e-06 3.59742947e-03 1.22182106e-04
     2.03674951e-05 2.09629089e-04 1.42736267e-05 3.69354654e-04
     7.65747422e-06 2.31966727e-06 5.52044907e-04 8.23973304e-06
     5.15415990e-04 0.00000000e+00 5.67168579e-06 2.22963995e-03
     5.27320636e-08 7.03430091e-05 3.67580183e-05 5.63039830e-04
     1.65700049e-03 4.52512246e-04 3.39526448e-07 1.76567361e-07
     6.27681660e-08 4.23957720e-06 2.81345242e-06 1.87447603e-05
     9.18558526e-09 1.21509854e-05 2.86697571e-05 1.99274468e-06
     1.02673154e-04 1.16282938e-04 3.59394529e-06 1.23863725e-05
     7.10502925e-05 1.84694903e-06 1.25500270e-05 2.86472812e-03
     2.55155146e-08 9.84702476e-04 2.57006705e-05 6.79052896e-07
     1.80523606e-06 2.01423446e-04 2.55155146e-08 0.00000000e+00
     7.31190989e-06 3.35110703e-04 8.03483555e-06 8.42960089e-06
     7.00656031e-07 1.18482639e-04 6.29285013e-05 3.22227696e-04
     5.09922090e-05 2.39165424e-07 5.09551202e-05 2.19799488e-05
     7.95938253e-07 3.27848294e-06 2.20005101e-05 1.21970964e-06
     5.54183992e-05 1.90730941e-04 1.88682362e-04 2.40429989e-06
     3.40747069e-06 2.13300633e-05 0.00000000e+00 7.34026657e-04
     1.95244903e-06 1.18535774e-06 1.42294604e-04 1.83654904e-03
     6.86265281e-07 6.80900144e-05 2.27548313e-06 3.12437477e-06
     3.40719162e-05 5.18347897e-04 9.44447305e-04 5.80222802e-07
     4.54458532e-06 4.69958114e-06 1.52227776e-04 6.34512872e-04
     5.45508173e-06 0.00000000e+00 7.45363377e-05 9.07080963e-04
     2.17688590e-05 4.87100411e-03 0.00000000e+00 2.53668009e-05
     4.88386390e-06 2.86057548e-04 2.62607330e-06 2.90356510e-04
     3.50126266e-04 5.74557156e-05 0.00000000e+00 8.81093975e-06
     9.14782497e-05 0.00000000e+00 9.89171852e-05 2.31406439e-05
     1.85034742e-03 3.08654139e-06 2.36543160e-05]



```python
print(len(selected_df.columns))
```

    155



```python
print((X_test.shape))
```

    (3463, 147)



```python
print(len(Regressor.feature_importances_))
```

    147



```python
DTAssesment_dict = { "Feature_Name": selected_df.iloc[: , 7:-1].columns, "Feature_Importance":Regressor.feature_importances_,}
DTAssesment_df = pd.DataFrame(DTAssesment_dict)
print(DTAssesment_df)
```

                    Feature_Name  Feature_Importance
    0          reviews_per_month            0.028930
    1                   latitude            0.735086
    2                  longitude            0.154929
    3                Shared room            0.000237
    4            Entire home/apt            0.017630
    ..                       ...                 ...
    142                 Wychwood            0.000099
    143           Yonge-Eglinton            0.000023
    144           Yonge-St.Clair            0.001850
    145  York University Heights            0.000003
    146       Yorkdale-Glen Park            0.000024
    
    [147 rows x 2 columns]



```python

pd.set_option('display.max_rows', None)
DTAssesment_df.sort_values(by=['Feature_Name', 'Feature_Importance'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature_Name</th>
      <th>Feature_Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Agincourt North</td>
      <td>1.736799e-05</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Agincourt South-Malvern West</td>
      <td>1.146839e-04</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Alderwood</td>
      <td>1.102270e-07</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Annex</td>
      <td>1.050285e-02</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Banbury-Don Mills</td>
      <td>1.889097e-04</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Bathurst Manor</td>
      <td>4.470237e-05</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bay Street Corridor</td>
      <td>8.375387e-04</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Bayview Village</td>
      <td>4.710741e-05</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Bayview Woods-Steeles</td>
      <td>7.804736e-03</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Bedford Park-Nortown</td>
      <td>2.501612e-04</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Beechborough-Greenbrook</td>
      <td>3.796450e-05</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Bendale</td>
      <td>5.847889e-06</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Birchcliffe-Cliffside</td>
      <td>5.428310e-06</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Black Creek</td>
      <td>3.138698e-04</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Blake-Jones</td>
      <td>4.290371e-06</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Briar Hill-Belgravia</td>
      <td>7.178432e-06</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Bridle Path-Sunnybrook-York Mills</td>
      <td>2.659020e-04</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Broadview North</td>
      <td>3.265986e-08</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Brookhaven-Amesbury</td>
      <td>5.735012e-06</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Cabbagetown-South St.James Town</td>
      <td>1.400257e-03</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Caledonia-Fairbank</td>
      <td>1.679179e-06</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Casa Loma</td>
      <td>1.021604e-05</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Centennial Scarborough</td>
      <td>1.319509e-05</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Church-Yonge Corridor</td>
      <td>2.106986e-05</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Clairlea-Birchmount</td>
      <td>2.529525e-05</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Clanton Park</td>
      <td>4.560190e-03</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Cliffcrest</td>
      <td>2.344982e-05</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Corso Italia-Davenport</td>
      <td>1.325092e-05</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Danforth</td>
      <td>3.021139e-06</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Danforth East York</td>
      <td>3.936717e-05</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Don Valley Village</td>
      <td>2.043793e-06</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Dorset Park</td>
      <td>3.597429e-03</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Dovercourt-Wallace Emerson-Junction</td>
      <td>1.221821e-04</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Downsview-Roding-CFB</td>
      <td>2.036750e-05</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Dufferin Grove</td>
      <td>2.096291e-04</td>
    </tr>
    <tr>
      <th>42</th>
      <td>East End-Danforth</td>
      <td>1.427363e-05</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Edenbridge-Humber Valley</td>
      <td>3.693547e-04</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Eglinton East</td>
      <td>7.657474e-06</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Elms-Old Rexdale</td>
      <td>2.319667e-06</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Englemount-Lawrence</td>
      <td>5.520449e-04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Entire home/apt</td>
      <td>1.763045e-02</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Eringate-Centennial-West Deane</td>
      <td>8.239733e-06</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Etobicoke West Mall</td>
      <td>5.154160e-04</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Flemingdon Park</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Forest Hill North</td>
      <td>5.671686e-06</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Forest Hill South</td>
      <td>2.229640e-03</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Glenfield-Jane Heights</td>
      <td>5.273206e-08</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Greenwood-Coxwell</td>
      <td>7.034301e-05</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Guildwood</td>
      <td>3.675802e-05</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Henry Farm</td>
      <td>5.630398e-04</td>
    </tr>
    <tr>
      <th>56</th>
      <td>High Park North</td>
      <td>1.657000e-03</td>
    </tr>
    <tr>
      <th>57</th>
      <td>High Park-Swansea</td>
      <td>4.525122e-04</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Highland Creek</td>
      <td>3.395264e-07</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Hillcrest Village</td>
      <td>1.765674e-07</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Hotel room</td>
      <td>3.935595e-05</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Humber Heights-Westmount</td>
      <td>6.276817e-08</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Humber Summit</td>
      <td>4.239577e-06</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Humbermede</td>
      <td>2.813452e-06</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Humewood-Cedarvale</td>
      <td>1.874476e-05</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Ionview</td>
      <td>9.185585e-09</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Islington-City Centre West</td>
      <td>1.215099e-05</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Junction Area</td>
      <td>2.866976e-05</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Keelesdale-Eglinton West</td>
      <td>1.992745e-06</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Kennedy Park</td>
      <td>1.026732e-04</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Kensington-Chinatown</td>
      <td>1.162829e-04</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Kingsview Village-The Westway</td>
      <td>3.593945e-06</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Kingsway South</td>
      <td>1.238637e-05</td>
    </tr>
    <tr>
      <th>72</th>
      <td>L'Amoreaux</td>
      <td>7.105029e-05</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Lambton Baby Point</td>
      <td>1.846949e-06</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Lansing-Westgate</td>
      <td>1.255003e-05</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Lawrence Park North</td>
      <td>2.864728e-03</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Lawrence Park South</td>
      <td>2.551551e-08</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Leaside-Bennington</td>
      <td>9.847025e-04</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Little Portugal</td>
      <td>2.570067e-05</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Long Branch</td>
      <td>6.790529e-07</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Malvern</td>
      <td>1.805236e-06</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Maple Leaf</td>
      <td>2.014234e-04</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Markland Wood</td>
      <td>2.551551e-08</td>
    </tr>
    <tr>
      <th>83</th>
      <td>Milliken</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Mimico (includes Humber Bay Shores)</td>
      <td>7.311910e-06</td>
    </tr>
    <tr>
      <th>85</th>
      <td>Morningside</td>
      <td>3.351107e-04</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Moss Park</td>
      <td>8.034836e-06</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Mount Dennis</td>
      <td>8.429601e-06</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Mount Olive-Silverstone-Jamestown</td>
      <td>7.006560e-07</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Mount Pleasant East</td>
      <td>1.184826e-04</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Mount Pleasant West</td>
      <td>6.292850e-05</td>
    </tr>
    <tr>
      <th>91</th>
      <td>New Toronto</td>
      <td>3.222277e-04</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Newtonbrook East</td>
      <td>5.099221e-05</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Newtonbrook West</td>
      <td>2.391654e-07</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Niagara</td>
      <td>5.095512e-05</td>
    </tr>
    <tr>
      <th>95</th>
      <td>North Riverdale</td>
      <td>2.197995e-05</td>
    </tr>
    <tr>
      <th>96</th>
      <td>North St.James Town</td>
      <td>7.959383e-07</td>
    </tr>
    <tr>
      <th>97</th>
      <td>O'Connor-Parkview</td>
      <td>3.278483e-06</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Oakridge</td>
      <td>2.200051e-05</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Oakwood Village</td>
      <td>1.219710e-06</td>
    </tr>
    <tr>
      <th>100</th>
      <td>Old East York</td>
      <td>5.541840e-05</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Palmerston-Little Italy</td>
      <td>1.907309e-04</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Parkwoods-Donalda</td>
      <td>1.886824e-04</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Pelmo Park-Humberlea</td>
      <td>2.404300e-06</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Playter Estates-Danforth</td>
      <td>3.407471e-06</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Pleasant View</td>
      <td>2.133006e-05</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Princess-Rosethorn</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Private room</td>
      <td>6.121406e-03</td>
    </tr>
    <tr>
      <th>107</th>
      <td>Regent Park</td>
      <td>7.340267e-04</td>
    </tr>
    <tr>
      <th>108</th>
      <td>Rexdale-Kipling</td>
      <td>1.952449e-06</td>
    </tr>
    <tr>
      <th>109</th>
      <td>Rockcliffe-Smythe</td>
      <td>1.185358e-06</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Roncesvalles</td>
      <td>1.422946e-04</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Rosedale-Moore Park</td>
      <td>1.836549e-03</td>
    </tr>
    <tr>
      <th>112</th>
      <td>Rouge</td>
      <td>6.862653e-07</td>
    </tr>
    <tr>
      <th>113</th>
      <td>Runnymede-Bloor West Village</td>
      <td>6.809001e-05</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Rustic</td>
      <td>2.275483e-06</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Scarborough Village</td>
      <td>3.124375e-06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Shared room</td>
      <td>2.369241e-04</td>
    </tr>
    <tr>
      <th>116</th>
      <td>South Parkdale</td>
      <td>3.407192e-05</td>
    </tr>
    <tr>
      <th>117</th>
      <td>South Riverdale</td>
      <td>5.183479e-04</td>
    </tr>
    <tr>
      <th>118</th>
      <td>St.Andrew-Windfields</td>
      <td>9.444473e-04</td>
    </tr>
    <tr>
      <th>119</th>
      <td>Steeles</td>
      <td>5.802228e-07</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Stonegate-Queensway</td>
      <td>4.544585e-06</td>
    </tr>
    <tr>
      <th>121</th>
      <td>Tam O'Shanter-Sullivan</td>
      <td>4.699581e-06</td>
    </tr>
    <tr>
      <th>122</th>
      <td>Taylor-Massey</td>
      <td>1.522278e-04</td>
    </tr>
    <tr>
      <th>123</th>
      <td>The Beaches</td>
      <td>6.345129e-04</td>
    </tr>
    <tr>
      <th>124</th>
      <td>Thistletown-Beaumond Heights</td>
      <td>5.455082e-06</td>
    </tr>
    <tr>
      <th>125</th>
      <td>Thorncliffe Park</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>126</th>
      <td>Trinity-Bellwoods</td>
      <td>7.453634e-05</td>
    </tr>
    <tr>
      <th>127</th>
      <td>University</td>
      <td>9.070810e-04</td>
    </tr>
    <tr>
      <th>128</th>
      <td>Victoria Village</td>
      <td>2.176886e-05</td>
    </tr>
    <tr>
      <th>129</th>
      <td>Waterfront Communities-The Island</td>
      <td>4.871004e-03</td>
    </tr>
    <tr>
      <th>130</th>
      <td>West Hill</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>131</th>
      <td>West Humber-Clairville</td>
      <td>2.536680e-05</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Westminster-Branson</td>
      <td>4.883864e-06</td>
    </tr>
    <tr>
      <th>133</th>
      <td>Weston</td>
      <td>2.860575e-04</td>
    </tr>
    <tr>
      <th>134</th>
      <td>Weston-Pellam Park</td>
      <td>2.626073e-06</td>
    </tr>
    <tr>
      <th>135</th>
      <td>Wexford/Maryvale</td>
      <td>2.903565e-04</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Willowdale East</td>
      <td>3.501263e-04</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Willowdale West</td>
      <td>5.745572e-05</td>
    </tr>
    <tr>
      <th>138</th>
      <td>Willowridge-Martingrove-Richview</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Woburn</td>
      <td>8.810940e-06</td>
    </tr>
    <tr>
      <th>140</th>
      <td>Woodbine Corridor</td>
      <td>9.147825e-05</td>
    </tr>
    <tr>
      <th>141</th>
      <td>Woodbine-Lumsden</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>142</th>
      <td>Wychwood</td>
      <td>9.891719e-05</td>
    </tr>
    <tr>
      <th>143</th>
      <td>Yonge-Eglinton</td>
      <td>2.314064e-05</td>
    </tr>
    <tr>
      <th>144</th>
      <td>Yonge-St.Clair</td>
      <td>1.850347e-03</td>
    </tr>
    <tr>
      <th>145</th>
      <td>York University Heights</td>
      <td>3.086541e-06</td>
    </tr>
    <tr>
      <th>146</th>
      <td>Yorkdale-Glen Park</td>
      <td>2.365432e-05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>latitude</td>
      <td>7.350858e-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>longitude</td>
      <td>1.549293e-01</td>
    </tr>
    <tr>
      <th>0</th>
      <td>reviews_per_month</td>
      <td>2.893028e-02</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = DTAssesment_df.iloc[: , 0].values
feature_importance = DTAssesment_df.iloc[: , 1].values
ax.bar(langs,feature_importance)
plt.show()

```


![png](output_50_0.png)


# Estimating the model accuracy


```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
```




    80301.81634421022




```python
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```




    -0.3682118217337307




```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)
```




    93.03667340456252




```python
from numpy import math
from sklearn.model_selection import cross_val_score
Regressor_scores = cross_val_score(Regressor, X, y, cv = 10, scoring="neg_mean_absolute_error")
print(Regressor_scores)
Regressor_score = round(sum(Regressor_scores )/len(Regressor_scores ), 3)
print(Regressor_score)

```

    [-144.40415704 -107.33602771 -124.47806005  -90.97344111 -101.7551963
      -87.04965358 -116.14318707 -101.54682081  -98.35144509 -157.73757225]
    -112.978


# Random Forest Model


```python
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor() 
RF.fit(X_train, y_train)
y_RFpred = RF.predict(X_test)
prediction_RFdict = {"y_pred":y_RFpred, "y_test":y_test}
predictions_RFdf = pd.DataFrame(prediction_RFdict)
print(predictions_RFdf) 


```

               y_pred  y_test
    0      813.120000     169
    1      152.160000     153
    2      144.870000      73
    3       44.300000      51
    4       64.640000      51
    5      182.050000     130
    6       47.140000      50
    7      131.730000      78
    8      128.760000     146
    9       78.700000      62
    10     294.640000     150
    11     140.210000     104
    12     111.920000      99
    13     144.950000     199
    14      69.110000      38
    15     172.880000     105
    16     138.570000     225
    17     168.470000     134
    18     172.200000     199
    19      95.380000     110
    20      82.470000      85
    21     139.400000      90
    22      73.840000      54
    23     121.910000     118
    24     112.420000     110
    25      94.250000     119
    26     149.430000     100
    27      62.180000      53
    28     114.670000      70
    29     122.520000      88
    30     125.800000     113
    31     155.390000     100
    32      50.210000      53
    33     249.310000     290
    34     132.040000      45
    35      85.370000      32
    36      56.760000      61
    37     132.480000     292
    38      76.890000      55
    39      80.170000      75
    40      45.610000      32
    41      62.950000      71
    42      92.700000      87
    43      46.520000      46
    44      48.530000      51
    45     143.800000      95
    46     139.250000      87
    47     276.430000     175
    48      59.320000     100
    49      58.260000      30
    50     153.820000      36
    51     151.810000     150
    52      88.870000      35
    53     149.340000      49
    54      66.810000      50
    55     153.940000     167
    56     136.590000     325
    57      46.210000      77
    58      53.670000      26
    59      50.130000      70
    60     112.250000     357
    61     113.190000     110
    62     113.110000      70
    63     125.570000     100
    64     191.930000      99
    65     277.650000     150
    66      26.960000      21
    67     121.540000      93
    68      47.070000      40
    69     114.560000      92
    70     114.540000     125
    71      88.650000     100
    72     133.680000     145
    73      17.120000      14
    74     203.020000     194
    75      30.480000      29
    76      58.910000      29
    77      54.210000      66
    78      84.150000      74
    79     247.150000      99
    80     196.040000     100
    81     113.460000     118
    82      94.840000     100
    83      90.680000      80
    84     133.970000      94
    85      62.130000      39
    86      60.460000      20
    87      18.570000      15
    88     119.150000     350
    89     151.820000     159
    90      51.280000      55
    91     131.950000     200
    92     290.230000     220
    93      42.600000      40
    94      90.080000      98
    95     220.330000     102
    96      46.110000      14
    97     167.390000     233
    98      53.970000      70
    99      50.830000      70
    100    124.990000     100
    101     70.330000      75
    102    303.250000     393
    103    145.260000     149
    104     80.830000      68
    105     68.800000      38
    106     79.410000     120
    107    125.050000     490
    108     84.080000      54
    109     65.430000      60
    110    117.810000      55
    111    134.760000      80
    112     51.200000     500
    113     72.130000      45
    114    106.810000     120
    115    158.040000     136
    116    262.200000     105
    117    794.070000     375
    118    114.120000      61
    119    656.260000     125
    120    105.470000     102
    121    409.490000      80
    122     67.110000      24
    123    151.330000     328
    124     61.300000      70
    125    232.930000     149
    126     64.410000     125
    127     82.880000      85
    128     66.790000     175
    129    213.200000      50
    130     75.080000      30
    131    178.210000      51
    132    458.690000    1000
    133     51.640000      51
    134     71.240000      85
    135     68.780000      57
    136    177.950000     111
    137     84.320000      80
    138     67.840000      31
    139    208.680000     125
    140    117.340000      73
    141    137.570000     148
    142     56.700000      73
    143    185.770000     105
    144    115.760000     100
    145    183.980000     208
    146     81.080000     170
    147    409.380000      75
    148    304.140000      70
    149    113.320000      90
    150     43.060000      49
    151    157.810000     150
    152    145.230000      79
    153    118.000000     116
    154     45.570000      40
    155     84.680000     429
    156    119.040000     144
    157     58.850000      65
    158     95.310000      88
    159     61.310000      50
    160    163.830000     220
    161    143.220000     300
    162     54.720000      50
    163     46.590000      55
    164     79.170000      55
    165     45.820000      45
    166     92.380000      36
    167     76.860000     175
    168     97.360000      98
    169    149.320000     200
    170     81.540000      57
    171     98.840000      93
    172     47.950000      49
    173     43.910000      45
    174     78.180000     109
    175    197.200000      85
    176    291.130000     299
    177     53.890000      38
    178    148.020000     114
    179    130.160000     100
    180     90.890000     124
    181     64.220000      45
    182     87.750000     450
    183     68.130000      60
    184    303.870000     250
    185    210.720000      51
    186    203.170000      75
    187    132.760000     171
    188    109.770000     105
    189     78.190000     180
    190    113.280000      90
    191    145.000000     185
    192     94.160000     153
    193    111.480000     105
    194    585.680000     170
    195    142.960000      90
    196     82.330000     100
    197    208.170000     135
    198    184.540000     199
    199     87.180000      50
    200     92.640000      99
    201    300.970000     225
    202    226.670000     265
    203     95.190000      84
    204     68.510000     110
    205     57.940000      66
    206     47.640000      31
    207    146.060000     132
    208    122.670000     146
    209    109.880000      57
    210    108.910000     200
    211    107.030000      80
    212    215.080000     150
    213    168.960000     100
    214    132.740000      47
    215     96.600000      80
    216    194.780000     114
    217     66.910000      77
    218     53.670000      68
    219    109.050000     209
    220    143.450000     104
    221    150.950000     150
    222     41.000000      32
    223    170.730000     140
    224     43.920000     288
    225    189.860000      90
    226    218.130000     100
    227     92.560000     120
    228     61.360000      75
    229     55.620000      70
    230     44.040000      26
    231    207.240000     200
    232     45.340000      45
    233     68.210000      55
    234    123.670000      79
    235     69.030000      60
    236     82.670000      42
    237    142.600000     130
    238    585.120000     149
    239    516.330000     166
    240     40.960000      62
    241    125.370000    2143
    242     69.980000      65
    243    140.490000     200
    244    193.430000     179
    245    569.700000     119
    246     83.270000      42
    247    306.980000     154
    248    140.750000     107
    249     43.970000      75
    250    146.780000      48
    251    141.010000      82
    252   1263.120000      90
    253     82.100000      95
    254    114.180000      59
    255    113.090000      95
    256     44.260000      69
    257    608.540000     350
    258    239.870000     132
    259    104.900000      74
    260     86.630000      45
    261    128.310000     196
    262     67.440000      40
    263    133.010000     120
    264    102.620000     135
    265    146.200000     100
    266    104.020000      75
    267    111.960000      80
    268    175.960000     105
    269     77.050000      64
    270     56.770000      45
    271    142.410000      94
    272    188.590000     203
    273    126.510000     146
    274     39.700000      78
    275    153.550000     110
    276     93.190000     141
    277    132.760000     500
    278     70.360000      83
    279    208.380000     170
    280     91.240000      69
    281     66.640000      90
    282    189.630000      33
    283    167.750000     179
    284    177.670000     159
    285     46.350000      48
    286    106.810000     150
    287    160.520000     120
    288    146.100000     275
    289    148.250000     388
    290    234.060000      99
    291     51.290000      99
    292    144.540000      91
    293    148.710000     149
    294    215.100000     149
    295     93.760000      50
    296    135.320000      80
    297    116.770000     591
    298    133.250000     189
    299    167.470000     130
    300    126.400000     118
    301    774.730000     150
    302     98.030000      92
    303    178.660000      80
    304    143.200000      78
    305    189.080000     184
    306    137.020000     155
    307    157.390000      80
    308    171.800000     128
    309    128.970000     181
    310    182.570000      92
    311    127.180000     164
    312     65.470000      85
    313     46.590000      45
    314     60.580000      35
    315     78.550000      39
    316    154.050000     124
    317    117.280000      70
    318    947.190000     656
    319    106.330000      99
    320    131.290000     195
    321     97.520000     135
    322    114.370000      76
    323    117.180000     113
    324    105.490000      99
    325    186.330000     102
    326    155.680000      89
    327    354.920000     324
    328    186.950000     170
    329     60.080000      48
    330     78.680000      70
    331    495.330000      60
    332    285.850000     230
    333    171.730000      80
    334    134.900000     112
    335     96.210000      89
    336    194.160000     249
    337    122.880000     199
    338    248.520000     300
    339     55.520000      65
    340    216.500000      44
    341    435.580000      60
    342    187.440000      80
    343    189.720000     357
    344     78.260000      70
    345    170.220000     164
    346     67.580000      66
    347    226.750000     107
    348     73.890000      57
    349     80.200000      89
    350     81.000000      67
    351     59.780000     100
    352    159.360000     150
    353     88.380000      48
    354     75.310000      44
    355    148.000000     207
    356    255.510000     175
    357     75.800000     460
    358     87.020000      98
    359    141.240000      66
    360    128.650000      91
    361     87.460000     119
    362    114.510000     166
    363    190.140000      99
    364    129.030000     189
    365    691.340000    1500
    366    140.880000     156
    367    212.530000     102
    368    142.210000     175
    369    106.740000      65
    370     54.560000      45
    371    146.870000     280
    372     91.950000     185
    373    200.480000     134
    374     98.690000     110
    375    132.350000     125
    376    152.450000     156
    377    324.310000     499
    378    156.250000     100
    379     99.600000     449
    380    151.860000     150
    381     33.440000      54
    382    152.090000     102
    383    127.510000     299
    384    457.460000     199
    385    169.350000      60
    386    531.090000     399
    387    168.670000     111
    388    111.580000      92
    389    107.230000     104
    390    105.720000      96
    391     18.780000      15
    392    135.760000      40
    393    153.040000     106
    394    186.700000      82
    395     60.500000      50
    396    138.960000      82
    397    180.820000     149
    398    131.320000     120
    399    222.110000      60
    400    113.150000      87
    401    103.070000      65
    402     90.930000      89
    403     51.440000     699
    404    132.460000     230
    405     42.730000      36
    406    132.260000     119
    407    159.350000      70
    408    808.300000     139
    409    123.300000     280
    410     82.090000      99
    411     81.770000      59
    412    112.630000     278
    413    125.780000     195
    414     62.820000     120
    415    116.390000     150
    416    145.350000      89
    417     48.620000      59
    418    102.010000      92
    419    134.420000      98
    420    108.060000      65
    421    112.490000     130
    422    137.250000     110
    423     58.570000      55
    424     60.890000      58
    425    214.170000      73
    426    109.360000      80
    427    113.000000      93
    428     91.380000      65
    429    295.070000     220
    430    155.050000     184
    431    151.400000      90
    432    187.460000     300
    433     59.030000      50
    434    209.160000    1750
    435     43.030000      37
    436     15.710000      14
    437    166.040000     151
    438     96.560000      83
    439    202.000000     233
    440     59.850000      60
    441    236.580000     150
    442     65.540000     180
    443     65.710000      50
    444    229.430000     120
    445    104.970000      85
    446    436.540000     210
    447    118.050000      96
    448    162.620000      99
    449    160.100000     115
    450     35.220000      37
    451    156.390000     388
    452    143.530000      80
    453    124.910000     164
    454     58.050000      35
    455     75.110000     200
    456    197.280000     125
    457     47.460000      30
    458   1165.540000      70
    459    199.940000      70
    460     50.110000      36
    461     63.600000      32
    462     88.710000     120
    463    126.350000     499
    464    156.740000     154
    465    108.280000     112
    466     67.190000      45
    467    110.960000     189
    468     89.610000      85
    469     82.230000      91
    470     94.760000     102
    471     70.780000      85
    472    205.030000      88
    473    144.250000     143
    474     58.720000      15
    475    110.520000     182
    476    156.830000     171
    477    116.850000      67
    478    185.250000      57
    479     98.240000     142
    480     90.760000     297
    481    165.360000      81
    482    115.780000     120
    483    209.540000     101
    484     83.030000      60
    485     63.120000      50
    486     57.230000      90
    487     97.690000     155
    488     97.390000     119
    489     85.350000      49
    490    153.740000     499
    491     88.170000      95
    492    152.940000     206
    493     82.310000      48
    494     76.190000      50
    495     40.030000      50
    496    142.110000      70
    497    104.870000     168
    498     80.500000      83
    499    149.560000      65
    500    104.830000     158
    501    129.420000     200
    502   1121.650000     160
    503    116.540000     410
    504    211.390000     643
    505     70.310000     200
    506    143.480000      86
    507    148.090000     290
    508     68.410000      55
    509    183.300000      87
    510    106.840000     119
    511    121.590000     125
    512    117.040000     328
    513     78.230000     153
    514     45.440000      85
    515     68.390000      50
    516    316.240000     500
    517     91.810000      54
    518    178.710000     695
    519     56.840000      60
    520    193.680000     113
    521     49.710000      55
    522    186.810000     179
    523    122.230000     120
    524    129.350000     500
    525    392.490000     301
    526     50.970000      49
    527    131.610000     121
    528     51.890000      65
    529    234.100000     247
    530    102.270000      36
    531    211.650000     167
    532     78.630000     123
    533     41.790000      80
    534    144.400000      60
    535     99.970000      58
    536    119.150000      99
    537    267.400000     190
    538    124.720000      50
    539    138.030000      75
    540     53.250000     117
    541     88.910000      52
    542    130.510000      65
    543    126.740000     116
    544    226.920000     253
    545     72.140000      70
    546    175.860000     103
    547    100.880000      65
    548    110.870000     375
    549    181.190000     180
    550    145.830000     117
    551    172.170000     249
    552    187.980000     110
    553    134.430000      87
    554    182.390000      97
    555     71.430000      36
    556    147.160000     355
    557     84.820000      45
    558    311.020000      75
    559    151.690000     143
    560     44.700000     102
    561     83.750000     150
    562    116.230000     130
    563    130.620000      97
    564     47.020000      43
    565    198.230000     300
    566     58.260000     101
    567    133.490000      50
    568     40.290000      32
    569    189.800000     161
    570    153.080000     100
    571    163.210000     129
    572    107.410000      44
    573     77.240000      90
    574    195.960000     121
    575     96.500000     145
    576    113.860000     134
    577    221.420000     121
    578     46.210000     100
    579    219.220000     100
    580     42.010000      39
    581     83.460000      99
    582    149.510000     110
    583    120.460000     113
    584    133.620000     400
    585     67.940000      90
    586     72.260000      40
    587    136.930000     151
    588     60.110000      85
    589    230.400000      80
    590    452.660000     199
    591    117.560000     239
    592     54.020000      40
    593     58.860000      67
    594     77.500000      90
    595     46.340000      58
    596     43.270000      45
    597     51.940000      40
    598    160.310000     120
    599    164.070000     115
    600     75.540000     200
    601    158.910000      66
    602    149.240000     116
    603     60.360000      65
    604     84.800000      72
    605    138.400000     201
    606     40.680000      34
    607     63.110000      35
    608    128.290000      35
    609     75.160000      40
    610    139.570000     120
    611    157.090000     151
    612    105.290000   10000
    613     33.620000      36
    614    122.030000     145
    615    111.060000     175
    616    142.640000     153
    617     75.990000      65
    618    249.420000      84
    619    102.040000     107
    620     59.200000      83
    621    147.650000      97
    622    163.490000     130
    623     40.270000      55
    624     61.820000      35
    625     38.160000     104
    626     58.290000      55
    627    111.390000      50
    628    189.900000      90
    629    200.890000     180
    630    151.210000     155
    631     96.400000     154
    632     70.580000      35
    633     97.380000     108
    634     50.710000      44
    635    128.780000      80
    636    114.430000     249
    637     51.840000      42
    638    188.890000     178
    639    287.620000     150
    640    168.750000     125
    641     65.360000      90
    642     74.810000     219
    643    123.020000     140
    644     43.750000      27
    645     66.350000      59
    646    145.190000     100
    647     65.650000     175
    648    143.550000     120
    649    152.950000     110
    650    333.720000     699
    651    129.930000      29
    652     30.670000      43
    653    116.690000     100
    654    129.880000     169
    655     88.710000      65
    656    114.300000      90
    657    305.080000      69
    658    122.610000     495
    659    191.110000      81
    660     61.110000     110
    661     47.810000      45
    662     77.340000      80
    663    172.050000     103
    664     86.800000      80
    665    116.060000     120
    666     91.140000     120
    667     53.785000      65
    668    187.330000     350
    669     28.780000      35
    670    231.810000     100
    671    107.810000      95
    672     67.143333      60
    673    111.730000     107
    674    594.460000     500
    675     94.650000     141
    676    122.720000     140
    677     71.210000      34
    678     45.990000      65
    679     96.250000      73
    680    129.710000     164
    681     56.850000      50
    682    147.220000     114
    683    139.560000     113
    684     48.880000      40
    685    127.160000     140
    686     48.150000      80
    687    154.900000     298
    688    205.700000      78
    689    189.010000      79
    690    155.590000      87
    691    118.220000     166
    692    146.790000     103
    693    109.060000      84
    694    129.520000     101
    695    127.190000     113
    696    123.270000     180
    697    220.090000     175
    698    153.550000     110
    699     43.300000      40
    700     53.880000      60
    701    224.210000     118
    702     61.860000      60
    703     62.610000      52
    704    104.380000     260
    705     60.170000     130
    706     85.930000      60
    707    165.590000      50
    708    225.810000     193
    709    218.540000     109
    710     99.310000      80
    711    391.450000     135
    712    145.900000     264
    713     75.150000      75
    714    168.230000     122
    715    130.890000     106
    716    141.130000     111
    717    131.500000     250
    718    274.550000     213
    719    147.010000      31
    720     94.690000      50
    721     80.810000      89
    722     67.270000      37
    723    150.010000     187
    724     78.690000      96
    725    169.570000     163
    726     74.050000      80
    727    184.060000     163
    728    113.250000     166
    729     98.660000     114
    730     94.450000     120
    731     43.810000     100
    732    144.660000      55
    733    180.460000      40
    734    114.000000      87
    735    116.390000      81
    736    123.190000     850
    737     48.360000      36
    738     54.630000      33
    739    141.880000     199
    740    193.300000      71
    741    219.570000     150
    742    189.960000     524
    743    104.020000     146
    744    157.680000     107
    745     48.320000      26
    746     58.680000      78
    747     78.370000      68
    748     50.560000      43
    749    149.530000      69
    750     80.470000     120
    751     65.260000      58
    752     77.720000      46
    753     75.290000     131
    754    139.610000     100
    755     56.210000      28
    756     53.550000      70
    757    171.390000     300
    758    152.450000     155
    759    138.150000      15
    760    205.530000      90
    761    195.430000     118
    762     90.550000      68
    763    140.120000     230
    764    121.490000      99
    765     45.630000      40
    766    167.830000     104
    767    117.230000      82
    768     70.190000      35
    769     84.110000      50
    770     80.550000      85
    771    135.920000     128
    772    151.580000     260
    773     96.870000      92
    774    122.070000      73
    775    201.080000     114
    776    151.060000     200
    777    102.330000      98
    778    118.230000      91
    779    126.130000     239
    780    132.560000      60
    781    123.340000      60
    782     66.400000      50
    783     78.160000      40
    784    770.350000     100
    785    118.490000      77
    786     51.010000      45
    787    138.590000     165
    788    103.970000     232
    789    142.990000     150
    790    117.120000     100
    791    193.730000      45
    792     18.680000      15
    793     61.410000      80
    794    194.050000     147
    795    147.910000     139
    796    142.510000     154
    797    127.600000     216
    798    137.420000     145
    799    132.380000     179
    800    182.708000     140
    801     59.410000      69
    802    120.260000      51
    803    128.160000      90
    804    105.640000     182
    805    158.540000     200
    806    457.200000     207
    807    146.470000      99
    808     59.570000      65
    809    123.850000     100
    810    119.080000     171
    811    129.870000     550
    812    238.110000      95
    813     66.380000      77
    814    148.600000     117
    815    160.950000     278
    816    193.590000      98
    817     65.960000      65
    818     42.620000      37
    819     35.560000      33
    820    191.230000      80
    821    228.400000     219
    822    113.240000      75
    823     40.810000      35
    824    208.700000     124
    825    232.860000      70
    826     59.480000      65
    827    105.400000      76
    828    136.030000     148
    829    140.000000      86
    830    249.320000     134
    831    298.900000     264
    832     56.650000      35
    833    124.930000     110
    834    152.770000      69
    835     85.170000      35
    836     90.480000      31
    837     57.360000      46
    838    183.980000     140
    839    182.640000     282
    840    109.500000     120
    841    157.390000     150
    842    158.780000     185
    843    144.000000      98
    844    176.070000     120
    845    103.100000      65
    846    181.310000     194
    847    291.380000     400
    848    183.880000      40
    849    140.680000     286
    850    133.720000     200
    851     86.320000      50
    852    225.110000     800
    853     46.530000     100
    854    140.460000      95
    855     54.530000      69
    856    127.030000      91
    857    160.200000     180
    858     83.090000      73
    859    356.560000      65
    860    115.910000      60
    861    182.790000      81
    862    228.840000     799
    863    175.350000      94
    864    139.280000     169
    865     65.430000      42
    866    120.150000     174
    867    131.130000     139
    868    120.510000     122
    869     64.140000      70
    870     58.380000      51
    871    502.790000     100
    872    145.890000      80
    873     40.320000     263
    874     72.330000      45
    875     83.110000      39
    876    347.350000     600
    877     47.170000     200
    878    124.200000     129
    879     52.890000      50
    880     86.770000      86
    881     93.930000     280
    882    104.290000     115
    883    285.910000      80
    884     15.840000      14
    885    189.040000     124
    886     50.260000      51
    887    137.150000     111
    888    242.400000      90
    889    143.130000     109
    890     53.030000      69
    891    153.160000     151
    892    206.580000     102
    893    120.990000      77
    894     47.420000      41
    895    147.420000     112
    896     62.950000      42
    897     79.700000      80
    898    171.330000     140
    899     55.830000      69
    900    218.080000     174
    901    111.540000     140
    902     85.080000     109
    903     48.280000      60
    904    273.560000      99
    905     59.040000      55
    906    106.150000     200
    907     83.440000     150
    908     48.030000      62
    909    296.640000     204
    910    213.160000     133
    911    138.990000     105
    912     56.180000      50
    913     96.640000      86
    914    142.670000     173
    915     88.610000      69
    916    121.430000     155
    917    123.850000     143
    918    172.450000     105
    919    161.390000     150
    920    180.330000      88
    921    198.480000      69
    922    116.460000     100
    923     44.040000      32
    924    314.200000      58
    925     59.270000      30
    926     78.680000      56
    927     62.550000      47
    928    119.570000     200
    929     51.630000      37
    930    126.960000      66
    931    180.880000     142
    932    165.720000     157
    933    124.660000      60
    934     80.160000      90
    935    162.250000     188
    936     63.260000      86
    937    149.580000     110
    938     51.740000      47
    939     57.210000     100
    940    103.810000     100
    941     41.440000     257
    942    142.800000     200
    943    152.360000     300
    944     59.710000     119
    945    166.410000     110
    946     42.270000      48
    947    211.920000      70
    948     44.593333      99
    949     94.800000      75
    950    214.100000     189
    951     85.670000      90
    952    168.760000     187
    953     73.440000     152
    954    184.370000     155
    955    283.950000     189
    956    258.820000     105
    957    118.250000      73
    958     51.630000     125
    959    152.440000     138
    960    101.450000     116
    961    150.150000     106
    962     41.730000      30
    963    484.020000     109
    964    224.220000     250
    965    156.280000     135
    966     58.270000     139
    967    139.510000     138
    968    399.790000     179
    969    127.120000      80
    970    124.360000     165
    971    214.910000     700
    972     65.750000      60
    973     83.440000      85
    974    112.140000     127
    975    139.850000     157
    976     63.590000     300
    977    164.620000     120
    978     45.090000      50
    979    123.430000     139
    980    200.660000      70
    981     64.010000     175
    982     66.610000      51
    983    129.080000     117
    984    288.820000     200
    985    181.860000      79
    986     52.750000      55
    987    153.590000     160
    988   1217.130000     115
    989   1060.270000     100
    990    100.650000      82
    991    313.790000     173
    992     75.650000     120
    993     52.090000     105
    994    279.240000     120
    995     54.840000      30
    996    135.190000     128
    997    102.220000      88
    998    132.150000      69
    999    127.180000      75
    1000    40.500000      25
    1001    59.210000      35
    1002    80.150000     100
    1003   137.260000     191
    1004   149.140000      74
    1005   202.400000     100
    1006    53.040000      75
    1007    36.620000      88
    1008    83.900000      55
    1009   148.880000      69
    1010   112.600000      84
    1011    87.680000      65
    1012    66.760000      50
    1013    51.180000      60
    1014   193.900000     169
    1015    69.230000      88
    1016    97.770000     135
    1017    66.570000      65
    1018   115.710000     157
    1019    58.800000      85
    1020   138.610000     100
    1021   145.650000      97
    1022    54.410000      46
    1023   131.330000      52
    1024   124.140000      79
    1025    45.420000      42
    1026   115.640000     199
    1027    57.110000      46
    1028   151.400000     109
    1029   102.190000     120
    1030    77.470000      79
    1031   139.230000     159
    1032    93.750000      75
    1033   157.360000     104
    1034   121.750000     120
    1035   169.850000      67
    1036   123.850000     177
    1037   117.480000      40
    1038   183.550000     130
    1039   271.880000     173
    1040    76.320000      50
    1041   125.530000     100
    1042   127.560000     100
    1043    69.190000      89
    1044   153.840000     100
    1045   156.020000     250
    1046   124.230000      90
    1047   218.130000     150
    1048   119.420000     129
    1049    57.750000      40
    1050   196.150000     124
    1051    94.880000      60
    1052   117.190000     199
    1053    50.520000      30
    1054   137.400000     104
    1055   160.640000      55
    1056    59.270000      38
    1057    38.460000      36
    1058   146.440000     114
    1059    86.350000     165
    1060    84.920000      93
    1061    53.410000      70
    1062   131.190000     136
    1063   153.180000     115
    1064    62.000000      75
    1065    47.300000      29
    1066   122.170000     160
    1067    49.060000      70
    1068    65.150000      65
    1069   124.840000     105
    1070   221.610000     150
    1071   136.740000      75
    1072    51.430000     150
    1073   482.210000     111
    1074    51.270000     129
    1075   127.020000     150
    1076   146.690000     156
    1077   123.270000     147
    1078   126.490000     150
    1079   136.580000     245
    1080    49.650000      78
    1081    79.120000      69
    1082    92.620000      86
    1083   692.830000      69
    1084    46.110000      85
    1085   394.190000      75
    1086   148.320000     260
    1087   254.300000     151
    1088   103.830000      98
    1089   222.270000     179
    1090    61.180000     165
    1091    68.780000      50
    1092    84.580000      50
    1093    71.090000      45
    1094   137.860000      80
    1095    54.120000      42
    1096    77.430000      70
    1097   137.210000     109
    1098   129.900000      79
    1099   161.450000     125
    1100   358.710000      65
    1101    54.980000      30
    1102   114.870000      79
    1103    44.860000      39
    1104   145.790000     108
    1105    54.560000      37
    1106   171.910000      79
    1107   165.010000      99
    1108   122.570000     156
    1109   139.180000      92
    1110   211.540000     100
    1111    31.960000      35
    1112   267.800000     199
    1113    53.780000     100
    1114    67.890000      73
    1115   107.760000     144
    1116    77.730000      78
    1117   107.450000     100
    1118   113.140000      99
    1119    70.840000      55
    1120    61.183000      65
    1121    86.980000      65
    1122   125.560000     175
    1123    50.900000      45
    1124    96.940000     893
    1125   144.200000     400
    1126   142.070000     129
    1127   157.150000     115
    1128   735.260000     295
    1129    94.300000     152
    1130   157.490000     100
    1131   154.050000     139
    1132   171.320000     126
    1133    43.950000     165
    1134   114.770000     125
    1135    76.870000      50
    1136   133.330000      76
    1137    45.390000      38
    1138    53.940000     200
    1139   237.080000     216
    1140   129.980000     450
    1141   160.830000     232
    1142    65.280000      50
    1143   128.100000     839
    1144   122.030000      71
    1145    77.920000      68
    1146   114.090000     395
    1147    64.140000      91
    1148   173.530000     328
    1149   136.010000     200
    1150   102.560000     106
    1151    61.640000      40
    1152   180.350000     110
    1153    51.620000      66
    1154    98.950000      55
    1155   131.620000      63
    1156   142.020000     153
    1157    56.180000      55
    1158   199.250000      80
    1159    88.130000      61
    1160   115.910000      45
    1161    87.710000      71
    1162   168.700000      89
    1163    52.030000      75
    1164   179.050000     139
    1165   373.270000     250
    1166   164.130000     110
    1167   262.530000     321
    1168    99.350000      80
    1169   130.700000     115
    1170    79.080000      90
    1171   128.500000      99
    1172    56.770000      86
    1173   203.490000     200
    1174   113.970000     166
    1175   127.000000     144
    1176    78.720000      80
    1177   164.300000     184
    1178   115.250000      80
    1179   107.750000      96
    1180   142.420000      42
    1181    43.800000      26
    1182   209.540000     139
    1183    69.210000      49
    1184   131.630000     100
    1185    86.130000      90
    1186   114.810000      79
    1187   128.140000     102
    1188    75.830000      43
    1189   149.450000      85
    1190   188.300000     217
    1191   190.460000     154
    1192   126.030000      76
    1193   128.850000     120
    1194   134.610000      38
    1195    48.710000      45
    1196   201.010000      20
    1197   103.240000     120
    1198    50.330000      33
    1199   184.900000      99
    1200   153.380000      88
    1201   194.900000      99
    1202   116.430000     129
    1203   114.930000      74
    1204    99.370000     164
    1205    97.320000     929
    1206    87.040000     129
    1207    56.480000      33
    1208   113.530000     120
    1209   418.850000     135
    1210    96.000000      75
    1211   131.780000     200
    1212    46.700000      48
    1213    47.940000      33
    1214   110.600000     104
    1215   134.200000     107
    1216   317.290000      85
    1217    67.440000      90
    1218   240.520000      85
    1219   113.810000     119
    1220   250.200000     200
    1221    55.710000    1312
    1222    62.730000      54
    1223    78.630000      86
    1224    61.340000      95
    1225    45.990000      60
    1226   142.850000     102
    1227    84.040000      75
    1228   167.660000     600
    1229   183.390000      69
    1230    82.550000      40
    1231   101.440000     150
    1232   106.310000      29
    1233   166.580000     200
    1234   215.450000     713
    1235   102.510000     130
    1236   105.320000      85
    1237    73.150000      50
    1238   157.240000     200
    1239   222.350000     180
    1240    58.080000     100
    1241   350.370000     250
    1242    42.190000     140
    1243   187.510000     166
    1244   187.530000     126
    1245   158.790000      85
    1246   213.320000     150
    1247   132.290000      75
    1248   364.450000      75
    1249   119.090000     100
    1250   122.600000      88
    1251   149.170000      97
    1252   127.960000     111
    1253   124.500000     138
    1254    70.350000      50
    1255    50.980000      50
    1256   175.380000      75
    1257   127.080000     100
    1258   104.380000      90
    1259    39.460000      41
    1260   191.690000     102
    1261   145.100000     125
    1262   269.430000      87
    1263    52.260000      45
    1264    64.490000      51
    1265   117.210000     129
    1266    59.570000      55
    1267   158.930000     190
    1268   123.770000      55
    1269    83.080000      57
    1270   110.520000     115
    1271    55.880000      72
    1272   148.630000      98
    1273   204.150000     221
    1274    77.420000      87
    1275   350.930000      76
    1276   203.060000     106
    1277    55.050000      69
    1278   146.240000     356
    1279   139.410000     110
    1280   251.100000     208
    1281   199.000000     160
    1282   128.260000      99
    1283    70.170000      35
    1284   127.150000      65
    1285   125.450000     131
    1286   234.550000      90
    1287   142.720000     347
    1288   118.790000     175
    1289    50.480000      55
    1290   135.870000      83
    1291   175.180000      29
    1292   319.440000     219
    1293   146.240000     315
    1294    67.410000      95
    1295   128.550000      60
    1296   139.120000     213
    1297   206.970000     190
    1298    77.870000      52
    1299    92.800000     275
    1300   131.790000      77
    1301   139.460000     123
    1302   146.770000      79
    1303   115.140000     109
    1304   236.430000      75
    1305   107.870000      68
    1306   150.990000     150
    1307   139.170000     178
    1308   135.080000     108
    1309    78.530000      40
    1310    92.100000     139
    1311    52.540000      40
    1312   140.580000     133
    1313   187.820000     290
    1314    36.920000      45
    1315   104.180000      64
    1316    73.140000      37
    1317   217.520000     120
    1318   144.960000      98
    1319   195.960000     226
    1320   187.830000     111
    1321   317.500000      89
    1322   138.170000      82
    1323   207.070000     185
    1324   133.280000     131
    1325    74.090000      72
    1326    18.450000      18
    1327   185.840000      94
    1328    76.590000      65
    1329   110.760000      68
    1330   126.110000     120
    1331   164.850000     129
    1332    57.530000     142
    1333   598.410000      65
    1334    44.970000      75
    1335    54.920000      34
    1336   153.640000      85
    1337   113.750000      45
    1338   123.710000     200
    1339   219.090000     111
    1340   165.410000     100
    1341    79.950000     140
    1342   131.590000     136
    1343   105.590000      90
    1344   147.640000    1999
    1345    61.750000      86
    1346    46.190000      70
    1347    38.340000      65
    1348   222.870000     224
    1349   274.120000     135
    1350   117.800000     123
    1351   132.290000     100
    1352    46.960000      50
    1353   112.700000     112
    1354   160.710000     148
    1355   150.430000     600
    1356    58.000000      29
    1357    54.450000      31
    1358    62.480000      61
    1359   210.500000     120
    1360   239.900000     235
    1361   257.700000      60
    1362   411.060000     180
    1363   953.770000     145
    1364    52.330000      52
    1365   145.050000     105
    1366   132.880000      67
    1367    84.590000      60
    1368   174.900000     134
    1369   169.990000     150
    1370    60.540000      59
    1371    79.570000      70
    1372    36.150000      82
    1373   175.620000      97
    1374   103.870000     122
    1375    42.680000      33
    1376   124.120000     167
    1377   150.580000      77
    1378   171.020000      95
    1379    85.230000      85
    1380   108.830000      66
    1381   122.160000     233
    1382   188.590000     350
    1383   126.230000     350
    1384   130.940000     225
    1385   205.860000      38
    1386    94.420000      97
    1387    94.500000      40
    1388   117.860000      69
    1389   114.070000      59
    1390    73.270000      55
    1391    47.160000      48
    1392    66.020000      62
    1393   157.290000     213
    1394    97.520000      30
    1395    48.670000      19
    1396   157.700000     399
    1397   136.870000     175
    1398    76.750000      65
    1399    75.940000      99
    1400   162.410000     900
    1401   108.420000     126
    1402    50.490000      28
    1403   150.710000     163
    1404   103.220000     140
    1405   149.340000     124
    1406   268.290000     400
    1407   142.290000      90
    1408   391.670000      99
    1409   164.750000     174
    1410    49.900000      35
    1411   107.830000      55
    1412   116.220000      99
    1413   160.950000     165
    1414    74.240000      70
    1415    76.980000      36
    1416    48.700000      55
    1417    34.970000      65
    1418    98.500000     108
    1419    80.830000      39
    1420    91.400000      60
    1421   104.680000      89
    1422   125.770000     300
    1423   149.790000     163
    1424    90.490000      40
    1425    96.120000      40
    1426   136.090000     130
    1427    81.090000      29
    1428   173.120000     172
    1429    47.420000      55
    1430    65.310000      50
    1431    72.830000     100
    1432   177.380000      84
    1433    70.990000      89
    1434   152.130000     159
    1435   170.460000     119
    1436   114.570000     189
    1437   183.520000     189
    1438    77.020000      70
    1439   137.720000      99
    1440   136.140000     139
    1441    50.440000      67
    1442   132.710000     248
    1443   126.840000     105
    1444   155.680000     175
    1445    70.000000      55
    1446    54.300000      65
    1447    91.260000      87
    1448   281.380000      89
    1449   279.740000     300
    1450    67.440000      39
    1451    92.310000      65
    1452   162.740000     118
    1453    71.740000     330
    1454   119.350000     105
    1455   514.250000     299
    1456   159.310000      76
    1457   185.520000     140
    1458   157.220000     479
    1459   219.100000    1218
    1460   211.220000     356
    1461   175.420000     167
    1462   107.500000      97
    1463    68.290000     150
    1464   110.120000     137
    1465   111.010000     133
    1466   184.030000     195
    1467    53.570000     130
    1468   134.030000     126
    1469   137.760000     170
    1470    95.150000      85
    1471    88.670000      75
    1472   139.420000      59
    1473    58.720000      72
    1474   160.990000     199
    1475   110.950000      82
    1476   257.460000      70
    1477    59.160000      34
    1478    99.210000      50
    1479    64.400000      41
    1480   134.540000     150
    1481   121.530000    1000
    1482   295.860000     750
    1483   154.700000      89
    1484   146.590000     500
    1485   174.000000     300
    1486    72.040000      23
    1487    71.490000      80
    1488   126.770000     170
    1489    47.890000      99
    1490   120.240000     125
    1491   174.480000     200
    1492    51.430000      50
    1493   129.140000     104
    1494   130.610000     148
    1495   393.090000      43
    1496   247.970000     150
    1497   197.980000      88
    1498   101.810000      70
    1499   148.200000     313
    1500    94.460000     110
    1501    92.580000     220
    1502    71.330000      49
    1503   197.940000      99
    1504    60.420000      90
    1505   173.640000     115
    1506    69.620000      50
    1507   132.800000     185
    1508   888.460000     850
    1509    78.920000     109
    1510   260.530000    2000
    1511   165.300000     166
    1512    59.890000      74
    1513   119.880000     152
    1514   120.730000     141
    1515   245.660000     120
    1516   120.170000     250
    1517   126.540000     165
    1518   170.300000     328
    1519   168.520000      99
    1520   179.390000     100
    1521   198.030000     204
    1522    52.700000      42
    1523   195.490000     431
    1524   162.730000     134
    1525   141.270000     140
    1526   123.590000     220
    1527    66.410000      32
    1528   104.660000      50
    1529    47.340000      45
    1530   143.740000     125
    1531   141.200000     160
    1532    49.550000      55
    1533   131.880000      84
    1534   227.110000    1999
    1535    48.490000      33
    1536   112.400000     214
    1537   145.750000     164
    1538    81.530000      60
    1539    92.430000     359
    1540    94.090000      80
    1541   141.050000    1499
    1542   120.590000     150
    1543    96.460000      80
    1544   375.650000      98
    1545    78.620000      65
    1546   121.120000     100
    1547   115.010000      90
    1548   212.130000     175
    1549    84.960000      47
    1550    69.850000      55
    1551   125.900000     135
    1552    62.260000      80
    1553   150.250000      88
    1554   105.720000      89
    1555   239.100000      70
    1556    85.850000      60
    1557    89.630000      69
    1558   138.960000     232
    1559   177.420000     119
    1560   228.940000      85
    1561   169.710000     150
    1562   720.090000     150
    1563   130.520000      65
    1564    95.590000     129
    1565   107.820000     145
    1566    85.330000      90
    1567    61.750000      83
    1568   135.830000     213
    1569    86.300000      60
    1570    91.130000      70
    1571    95.260000     171
    1572    85.960000      60
    1573   129.600000     105
    1574   120.640000      96
    1575    37.480000      30
    1576    72.150000      71
    1577   400.280000     200
    1578    63.450000      38
    1579   130.770000      99
    1580   117.350000      97
    1581   257.860000     115
    1582   193.180000     109
    1583   122.780000      34
    1584   107.310000     500
    1585   146.010000     172
    1586   128.410000     200
    1587   115.720000     214
    1588   134.800000     699
    1589    85.250000      86
    1590    60.860000      60
    1591   142.200000     100
    1592   157.600000      95
    1593   114.450000      89
    1594   214.360000     202
    1595    87.790000      60
    1596    57.520000      47
    1597   101.910000     121
    1598    70.870000      55
    1599    58.840000      51
    1600   150.950000     900
    1601    51.180000      50
    1602    95.110000      85
    1603    84.820000      56
    1604   275.990000      69
    1605    89.590000      53
    1606   105.030000     307
    1607    63.140000      75
    1608   140.670000     125
    1609   112.030000     117
    1610    59.710000      65
    1611   151.470000      90
    1612   146.460000     183
    1613    61.750000      55
    1614    46.110000      27
    1615    88.350000      62
    1616   487.750000     200
    1617   129.210000      70
    1618   140.560000      99
    1619   153.180000     200
    1620    45.460000      65
    1621   130.940000     151
    1622    35.860000      31
    1623    70.470000      75
    1624   119.370000     120
    1625   122.290000     128
    1626   178.730000     182
    1627    83.460000     268
    1628    66.890000     100
    1629   285.530000     175
    1630   133.750000     250
    1631    76.720000      35
    1632   263.730000     132
    1633   107.970000      60
    1634    48.170000      51
    1635   126.830000     104
    1636    82.980000     169
    1637   176.780000     118
    1638   166.790000     180
    1639   219.780000     265
    1640    26.320000      24
    1641    88.300000      80
    1642   148.550000    2096
    1643   102.850000      54
    1644   168.210000     107
    1645    39.720000      30
    1646    56.360000      72
    1647   386.960000      78
    1648   132.490000     180
    1649   145.450000     289
    1650    83.040000      28
    1651    98.720000      35
    1652   555.900000      75
    1653   136.230000     140
    1654   246.110000      65
    1655   149.700000     399
    1656    90.020000      45
    1657    65.400000      44
    1658   132.610000     200
    1659    70.240000      45
    1660   157.490000     150
    1661    76.880000     150
    1662    60.600000      75
    1663   101.690000      70
    1664   127.230000     125
    1665    83.070000      64
    1666    76.150000     900
    1667    93.480000     101
    1668    66.890000      85
    1669    71.180000      30
    1670   120.630000      77
    1671   291.610000     195
    1672    85.980000      66
    1673    89.690000      87
    1674   214.040000     100
    1675   172.210000     500
    1676   147.120000     125
    1677    70.000000      83
    1678   189.420000     295
    1679   188.050000      99
    1680    51.320000      53
    1681    76.630000      25
    1682    96.670000     295
    1683   113.490000      89
    1684   188.880000    1000
    1685   134.600000     179
    1686   139.960000     282
    1687    37.900000      39
    1688   120.460000     299
    1689   133.080000     120
    1690    91.600000      90
    1691   142.970000     157
    1692   114.890000      81
    1693    85.540000      65
    1694    88.160000     120
    1695   165.200000     100
    1696   239.300000     178
    1697   175.090000     149
    1698   130.270000     395
    1699   153.050000     150
    1700    17.130000      16
    1701   100.780000      80
    1702   223.770000     144
    1703    44.170000      40
    1704    54.300000     198
    1705   117.660000     170
    1706   263.590000     350
    1707    66.050000      66
    1708   168.870000     319
    1709    54.550000      75
    1710   339.220000      89
    1711   336.450000     195
    1712   106.180000      98
    1713   160.570000     109
    1714   232.250000     196
    1715    88.300000     150
    1716   101.030000     130
    1717    88.000000      65
    1718   124.250000     140
    1719    60.340000      39
    1720   115.830000     500
    1721   170.880000     120
    1722    60.110000      70
    1723   132.900000     383
    1724    74.670000      40
    1725    59.100000      45
    1726    49.450000      45
    1727    91.780000     125
    1728    63.370000      28
    1729   107.850000     106
    1730    89.850000     120
    1731   130.670000     125
    1732   553.960000     250
    1733   177.640000     185
    1734   189.480000     119
    1735    74.170000      35
    1736   147.700000      81
    1737    90.950000      80
    1738   108.170000     190
    1739    94.960000      78
    1740   150.330000     119
    1741    87.820000      70
    1742   133.020000      50
    1743   125.060000     134
    1744   123.420000     126
    1745   204.640000     120
    1746   199.610000     215
    1747   121.570000     168
    1748   236.190000     211
    1749   122.990000     120
    1750    70.610000     675
    1751    45.150000      50
    1752   139.040000      50
    1753    48.190000      49
    1754   130.280000      92
    1755    96.680000      90
    1756    51.680000      69
    1757   137.710000     999
    1758   100.870000     100
    1759    52.070000      41
    1760    79.980000     443
    1761   198.820000     100
    1762   239.520000     210
    1763   111.620000     175
    1764    61.020000      93
    1765    60.360000      59
    1766    98.140000     110
    1767   303.990000     354
    1768   161.560000     150
    1769   125.460000    3500
    1770   179.520000      89
    1771   149.460000     151
    1772   204.810000     175
    1773   186.380000      60
    1774   191.320000     110
    1775   127.770000     186
    1776   139.040000      85
    1777    95.560000     110
    1778    50.930000      45
    1779    99.360000     152
    1780   209.670000     210
    1781   177.050000     292
    1782   198.830000      99
    1783    91.640000      95
    1784   403.800000      67
    1785   276.530000      70
    1786   123.850000     245
    1787    99.370000     100
    1788   108.620000     218
    1789   190.290000     199
    1790    52.640000      42
    1791    38.680000     169
    1792    47.870000      75
    1793   177.170000      80
    1794   230.100000     456
    1795   125.440000     900
    1796   105.590000      80
    1797   114.170000      75
    1798   187.300000     300
    1799   172.220000     152
    1800    52.170000     200
    1801    15.730000      16
    1802   292.050000     299
    1803    53.010000      80
    1804   130.890000     195
    1805   123.070000     140
    1806   145.140000     150
    1807    57.370000     100
    1808   116.730000     150
    1809    72.530000      33
    1810   305.850000      59
    1811   205.530000      71
    1812   147.270000      60
    1813    48.320000      50
    1814    63.410000      50
    1815   169.420000      70
    1816   173.040000     148
    1817   213.040000     100
    1818    46.960000      53
    1819   214.060000     500
    1820    66.980000     100
    1821   149.140000     109
    1822    85.330000      80
    1823   157.020000     110
    1824    48.460000     150
    1825   170.380000     132
    1826    90.720000      88
    1827   132.430000     342
    1828   107.650000      94
    1829    87.020000      90
    1830   153.720000     263
    1831    62.940000      33
    1832   123.110000     190
    1833   139.370000     149
    1834   105.030000     163
    1835    48.240000      35
    1836    78.080000     216
    1837   150.690000     100
    1838   140.790000      98
    1839   134.650000     130
    1840   149.350000     138
    1841   141.870000     139
    1842   207.280000      65
    1843   164.860000      78
    1844    63.660000      60
    1845   150.500000      89
    1846   170.670000     500
    1847    50.340000      50
    1848   167.250000      85
    1849   172.790000      99
    1850   106.100000     115
    1851   160.480000     159
    1852   216.140000     101
    1853   281.660000     280
    1854   132.830000     151
    1855   151.590000     185
    1856   213.890000     117
    1857    97.190000      95
    1858   102.630000      97
    1859    58.320000      39
    1860   117.510000     250
    1861   131.470000     143
    1862   119.260000      70
    1863   134.480000     193
    1864   172.780000     120
    1865   111.150000    1400
    1866    48.200000      36
    1867   204.110000     214
    1868   118.660000     100
    1869   165.360000     117
    1870   107.380000     110
    1871   105.250000      82
    1872    47.990000      60
    1873    68.060000      47
    1874   114.530000      70
    1875    47.910000     700
    1876   113.110000      72
    1877    52.960000      60
    1878   171.350000      97
    1879   145.790000     110
    1880    50.010000      36
    1881    45.380000      35
    1882    92.990000     164
    1883   140.090000     109
    1884   134.480000     284
    1885   117.470000     160
    1886   180.750000      57
    1887   116.870000     136
    1888   194.030000     139
    1889   113.630000      75
    1890   194.630000     880
    1891   147.010000     234
    1892   221.620000     179
    1893    44.530000      45
    1894    29.240000      35
    1895    95.140000      79
    1896    92.380000      76
    1897   117.150000     279
    1898   295.550000     200
    1899   120.990000     136
    1900  1226.260000      80
    1901    92.290000      89
    1902    56.150000      87
    1903   150.240000      69
    1904    53.280000      40
    1905   100.980000     100
    1906    55.460000      85
    1907    58.060000      66
    1908    52.620000      30
    1909   196.030000     180
    1910   125.670000     150
    1911   174.300000     120
    1912   122.740000     150
    1913   124.300000     179
    1914   196.640000     150
    1915   109.340000     550
    1916    49.670000      50
    1917    69.310000      55
    1918    48.440000      26
    1919    35.220000      26
    1920   128.500000      50
    1921   144.130000     200
    1922   181.620000      94
    1923    18.770000      17
    1924    62.390000      40
    1925    51.110000      70
    1926   132.590000      98
    1927    99.100000      50
    1928    60.810000      31
    1929    55.730000      62
    1930   440.960000     225
    1931   103.300000      81
    1932    17.340000      15
    1933   110.610000     102
    1934   230.710000     143
    1935   149.420000     129
    1936   101.020000      99
    1937   143.800000      95
    1938   166.930000      78
    1939   198.060000     138
    1940   116.950000      60
    1941   144.750000      90
    1942    91.190000      50
    1943    99.380000     150
    1944   138.570000     361
    1945    45.090000      40
    1946    51.710000      61
    1947    88.840000      75
    1948   126.130000      84
    1949    97.350000      79
    1950    89.490000      83
    1951    70.170000      60
    1952   124.650000     129
    1953    57.560000      80
    1954   141.040000     205
    1955   116.980000      68
    1956   122.270000      85
    1957   146.750000     220
    1958    45.930000      57
    1959    96.860000     208
    1960   154.660000      90
    1961    88.460000     180
    1962   104.840000      78
    1963   393.960000     125
    1964   202.490000      90
    1965   111.630000     129
    1966    95.450000      95
    1967    94.780000      70
    1968   163.750000     229
    1969   428.030000     174
    1970  1280.250000     420
    1971   130.150000      75
    1972   136.450000     170
    1973   147.350000      90
    1974   113.870000      66
    1975    89.020000      80
    1976    36.010000      55
    1977   183.540000     200
    1978    67.190000      46
    1979   129.760000     100
    1980    81.870000      60
    1981    54.210000     171
    1982    41.050000      42
    1983   189.020000    1000
    1984   216.110000     120
    1985   143.990000    1999
    1986    88.800000      80
    1987   115.740000     250
    1988   133.850000     300
    1989    91.730000      60
    1990    47.620000      43
    1991   106.650000      99
    1992   109.200000      39
    1993    42.860000      36
    1994    19.120000      15
    1995   374.700000     415
    1996   115.780000      70
    1997   168.270000     207
    1998    73.270000      65
    1999   117.330000     350
    2000    53.230000      53
    2001   146.540000     140
    2002    45.240000     333
    2003   171.380000      99
    2004    82.480000      90
    2005    78.770000      50
    2006    53.560000      34
    2007    81.150000      85
    2008   150.580000     133
    2009    88.460000      74
    2010   238.530000     550
    2011    54.070000      50
    2012   186.410000      80
    2013   157.010000     140
    2014    68.150000      70
    2015   177.180000     400
    2016   146.830000      78
    2017    59.160000      37
    2018    63.700000      40
    2019   732.200000     450
    2020    43.770000      37
    2021   215.240000     259
    2022   132.660000     183
    2023   325.730000      78
    2024   235.250000     500
    2025   156.350000     134
    2026   247.670000     129
    2027   141.290000     121
    2028   181.840000     115
    2029    99.030000      74
    2030   108.990000     115
    2031    57.580000      60
    2032   632.560000     200
    2033    57.110000      48
    2034   102.860000     170
    2035   112.340000     160
    2036   139.740000     113
    2037   171.310000     100
    2038    75.730000     170
    2039   134.930000     107
    2040   148.460000     110
    2041   124.310000     135
    2042   196.110000     132
    2043   195.100000     122
    2044   153.850000     163
    2045   402.090000     100
    2046    45.280000      93
    2047   123.760000     149
    2048    87.990000      70
    2049    92.030000      66
    2050   282.870000     250
    2051   118.780000     109
    2052   168.280000     148
    2053   168.340000     100
    2054   178.190000     300
    2055   139.750000      89
    2056   297.470000      89
    2057   140.610000      82
    2058   141.350000      89
    2059    54.390000     100
    2060   178.040000     154
    2061    53.470000      75
    2062   224.340000     140
    2063    71.660000      80
    2064    92.540000     192
    2065   168.710000     104
    2066    61.060000      74
    2067    90.000000      45
    2068   124.570000     199
    2069   179.890000     200
    2070    53.720000      42
    2071    72.400000     130
    2072   194.570000     140
    2073    64.150000      25
    2074    92.910000      99
    2075    57.290000      60
    2076   108.550000      50
    2077   160.250000     250
    2078   134.470000     101
    2079   133.420000     110
    2080  1947.600000     180
    2081    55.520000      38
    2082    54.780000      31
    2083   107.770000     141
    2084   121.460000      50
    2085   133.220000      46
    2086    46.450000      50
    2087    85.860000      90
    2088    66.780000      60
    2089    70.850000      80
    2090    57.500000      35
    2091   378.080000     114
    2092    49.210000      35
    2093   189.820000     130
    2094   130.360000      60
    2095    97.410000      40
    2096   107.280000     100
    2097    60.090000      60
    2098   188.430000     149
    2099    50.440000      60
    2100   198.700000     180
    2101    91.860000     186
    2102   136.590000      89
    2103   127.320000      91
    2104    62.230000      59
    2105    95.530000      32
    2106    73.440000      30
    2107   373.370000     288
    2108   164.450000     105
    2109   133.300000     140
    2110   143.090000     139
    2111   135.410000      74
    2112    52.040000      92
    2113   140.720000     106
    2114   315.500000     153
    2115   145.570000     170
    2116    66.610000     119
    2117   168.830000     700
    2118    96.510000      86
    2119   150.810000     180
    2120    71.020000      35
    2121   121.700000     121
    2122   125.370000      96
    2123    48.340000      30
    2124   137.570000     189
    2125   165.610000     347
    2126   168.890000     140
    2127   119.060000      94
    2128   151.670000      82
    2129   101.810000      60
    2130   229.020000     138
    2131   176.020000      89
    2132   115.290000     149
    2133   177.770000     220
    2134    40.050000      85
    2135   256.480000      62
    2136   113.290000     119
    2137    37.620000      14
    2138   139.770000     131
    2139   144.070000     106
    2140   146.150000    1328
    2141    49.820000      39
    2142   218.190000     117
    2143   347.420000     500
    2144   185.710000      95
    2145    60.730000     130
    2146   125.490000     147
    2147   118.840000     179
    2148    45.280000      76
    2149   120.170000     263
    2150   240.770000     170
    2151   160.130000     100
    2152   156.660000     199
    2153   151.380000     130
    2154   219.710000     179
    2155   109.030000      88
    2156   105.940000      99
    2157   132.060000     403
    2158   111.040000     150
    2159    63.920000      35
    2160    97.010000     150
    2161    58.510000      45
    2162    94.860000      51
    2163    60.970000      62
    2164   211.320000     125
    2165    45.550000      80
    2166    87.800000      86
    2167    66.760000      41
    2168   126.400000     175
    2169   137.640000      62
    2170    43.050000      30
    2171   139.780000     109
    2172   100.830000      52
    2173   162.700000     150
    2174   207.440000     145
    2175   171.850000     178
    2176   125.180000     200
    2177   165.650000      95
    2178   169.250000     149
    2179   193.770000     499
    2180   129.630000     461
    2181   171.240000     114
    2182   120.500000      69
    2183    78.050000     109
    2184   109.670000      77
    2185   214.280000      80
    2186    72.580000      75
    2187    69.700000      40
    2188   115.680000     129
    2189   159.600000     105
    2190    62.400000     499
    2191    47.830000      40
    2192   169.000000     159
    2193   373.250000     800
    2194    68.410000      45
    2195   116.080000      50
    2196   120.500000    1837
    2197   102.490000      51
    2198   245.850000     139
    2199    77.870000      75
    2200   131.970000      85
    2201   134.380000     200
    2202    52.990000      32
    2203    61.740000      95
    2204    73.030000     130
    2205    64.360000      50
    2206   297.210000     100
    2207   153.260000     200
    2208    55.970000      34
    2209    57.900000     122
    2210    61.260000      47
    2211    61.900000      75
    2212    78.010000     100
    2213    47.660000      42
    2214   102.790000      85
    2215   209.180000     109
    2216    63.780000      71
    2217   226.540000     509
    2218   137.480000      99
    2219   104.560000     130
    2220   148.780000     121
    2221    98.730000     175
    2222    65.050000      79
    2223   203.460000     149
    2224    45.810000      49
    2225   158.930000     148
    2226    77.060000      60
    2227   116.480000     148
    2228   140.250000     190
    2229   174.990000     150
    2230   151.490000      25
    2231   126.550000     140
    2232   110.290000     195
    2233    46.960000      40
    2234    37.360000      28
    2235    79.570000      78
    2236   645.180000     100
    2237    45.280000      76
    2238    95.690000     149
    2239    60.360000      89
    2240   143.110000     136
    2241    62.800000      56
    2242   152.850000      45
    2243   224.920000     900
    2244    54.490000      59
    2245    48.350000      45
    2246   107.140000     263
    2247   125.170000     200
    2248   169.310000     104
    2249   145.740000     750
    2250    60.260000      21
    2251   127.160000     249
    2252   138.950000     300
    2253   163.140000     100
    2254   180.320000     160
    2255   181.540000     336
    2256   195.380000      40
    2257    61.460000      76
    2258   151.580000     126
    2259   119.490000     120
    2260   134.060000     158
    2261    58.170000      29
    2262   116.490000     150
    2263   183.842500     110
    2264   112.010000     177
    2265   132.780000      80
    2266    83.520000      73
    2267    56.290000      79
    2268    81.140000      39
    2269   358.620000     900
    2270   114.720000     203
    2271   199.480000     140
    2272    51.390000      48
    2273    82.610000      60
    2274   112.030000     226
    2275   182.610000     690
    2276    47.370000      39
    2277   122.610000      99
    2278    35.180000      40
    2279   526.880000     160
    2280    98.920000     584
    2281    38.050000      40
    2282   169.160000      55
    2283   110.980000     175
    2284   136.970000     218
    2285    48.140000      40
    2286    66.330000     150
    2287   153.420000      95
    2288    70.160000      45
    2289    46.500000      25
    2290    82.000000      65
    2291   160.220000      88
    2292   455.150000      73
    2293   143.030000      80
    2294   108.500000     295
    2295   115.840000      50
    2296   314.660000     150
    2297    50.290000      32
    2298   144.760000     103
    2299   235.080000     197
    2300   229.860000     185
    2301    44.820000      40
    2302   211.010000     160
    2303   124.420000     100
    2304   107.460000     125
    2305    72.390000      74
    2306   112.710000     100
    2307   194.310000     190
    2308    83.630000     150
    2309    51.990000     135
    2310    66.250000      40
    2311    69.110000      90
    2312    88.440000      85
    2313   124.790000      47
    2314    43.140000      35
    2315   196.380000     171
    2316    56.860000      55
    2317   150.180000      95
    2318   150.080000      89
    2319    69.020000      65
    2320   140.600000      59
    2321    73.700000      51
    2322   140.180000      95
    2323    48.950000      42
    2324   125.950000     128
    2325    46.060000      45
    2326    72.880000      79
    2327    44.260000      65
    2328   134.190000     190
    2329   134.930000     232
    2330   172.890000      91
    2331    59.230000      45
    2332   216.290000     158
    2333    53.020000      25
    2334   136.430000     164
    2335   170.480000      66
    2336    55.610000      45
    2337   117.330000     100
    2338    78.900000     105
    2339   259.380000     130
    2340   143.530000      40
    2341   149.590000     148
    2342   188.810000     300
    2343   149.510000     128
    2344   138.540000     264
    2345    83.990000      93
    2346    55.750000      58
    2347   267.960000     435
    2348    37.210000      38
    2349   278.420000     130
    2350   247.040000     553
    2351    98.670000      55
    2352    48.140000      39
    2353   142.030000      80
    2354    62.800000      96
    2355   104.890000      45
    2356    98.400000      76
    2357   142.700000      84
    2358   142.520000     149
    2359   118.880000     160
    2360   197.970000     155
    2361   113.820000     300
    2362   118.860000    1200
    2363   404.720000     106
    2364   103.540000     110
    2365    90.090000     120
    2366   208.550000     248
    2367   126.610000     114
    2368    79.010000      75
    2369   280.220000      75
    2370   182.260000     131
    2371   107.210000      60
    2372    63.670000      95
    2373    65.160000     130
    2374   237.360000     133
    2375   156.000000      95
    2376   147.560000     100
    2377    58.420000     117
    2378   115.060000     129
    2379   137.270000     656
    2380   143.240000     140
    2381   136.110000     201
    2382   131.710000     100
    2383   168.000000     329
    2384   205.440000      95
    2385    99.430000      70
    2386   101.940000     125
    2387   153.780000     125
    2388    91.510000      80
    2389    66.090000      62
    2390   118.500000      55
    2391   126.870000      99
    2392    43.990000      40
    2393   157.040000      45
    2394   384.340000     300
    2395    34.500000      34
    2396    81.120000      81
    2397    58.890000     150
    2398   227.330000     195
    2399    47.180000      39
    2400    64.100000      45
    2401   100.950000      40
    2402   117.350000     118
    2403   134.240000      98
    2404   176.890000      99
    2405    90.670000      95
    2406   103.650000      86
    2407   156.670000     199
    2408    80.640000      70
    2409   284.910000     215
    2410   110.470000     100
    2411   134.030000      99
    2412    40.150000      75
    2413   128.210000      79
    2414   106.670000     220
    2415    84.040000      90
    2416   137.840000      83
    2417    99.820000     110
    2418   193.060000     185
    2419   134.990000      99
    2420   125.790000     165
    2421   182.080000      70
    2422   225.250000     196
    2423   123.050000     198
    2424   275.650000      90
    2425   171.280000     115
    2426    54.000000      40
    2427   145.200000     199
    2428   164.920000     110
    2429   237.960000     188
    2430   596.490000     849
    2431   148.350000     135
    2432    33.210000      30
    2433   284.320000     100
    2434   115.300000    2000
    2435    77.900000      80
    2436   124.250000      49
    2437    85.610000      74
    2438    33.720000      35
    2439   103.840000     100
    2440    45.620000      39
    2441   135.410000     213
    2442    75.340000      69
    2443    67.310000      60
    2444    38.300000      48
    2445   159.680000     115
    2446    52.260000      27
    2447   277.150000     100
    2448    75.820000      36
    2449    39.210000      40
    2450   217.440000     100
    2451   141.400000     185
    2452    54.670000      49
    2453    84.170000      79
    2454    88.510000     175
    2455   157.680000     179
    2456    57.870000      90
    2457    92.770000      99
    2458    45.930000      46
    2459    44.780000      40
    2460    44.370000      37
    2461   194.580000     145
    2462   137.130000      57
    2463    59.050000      80
    2464    65.970000     150
    2465   154.940000      74
    2466   192.590000     163
    2467   154.430000      86
    2468    60.920000     100
    2469    61.350000      50
    2470   145.240000     148
    2471    62.850000      65
    2472    48.620000      43
    2473   128.090000     214
    2474   215.080000      55
    2475    54.520000      63
    2476   126.070000     132
    2477    67.400000     115
    2478   831.580000     575
    2479   171.680000     300
    2480    80.080000     101
    2481    52.830000      33
    2482    50.250000      76
    2483    93.450000      52
    2484    77.450000      71
    2485   140.650000      78
    2486   246.120000     126
    2487   123.160000     140
    2488   141.590000     173
    2489   104.050000      50
    2490   141.510000     165
    2491    90.020000     172
    2492    75.840000      70
    2493   119.470000      68
    2494   105.580000      76
    2495   112.800000      99
    2496    40.500000      45
    2497   185.060000     149
    2498   101.100000      92
    2499   109.560000     109
    2500    52.140000      50
    2501    54.780000      55
    2502    59.750000      31
    2503   107.590000     159
    2504    51.540000      60
    2505   204.430000     154
    2506   126.980000      82
    2507   136.450000     169
    2508    39.200000      40
    2509   126.830000     189
    2510   150.620000     111
    2511   141.760000     100
    2512   135.430000     200
    2513   130.940000      87
    2514    65.360000      75
    2515    59.620000     119
    2516   135.800000      99
    2517   165.530000     100
    2518   123.170000      54
    2519    97.340000      70
    2520    76.400000      31
    2521   120.650000      85
    2522   150.070000     200
    2523    41.470000      39
    2524   137.820000     100
    2525   124.450000     205
    2526   123.380000     160
    2527   114.670000     108
    2528   296.790000     249
    2529   135.410000     199
    2530    61.690000      38
    2531    41.610000      41
    2532   376.460000     122
    2533    42.170000      30
    2534   125.670000      66
    2535   111.470000      99
    2536   149.090000     210
    2537   154.010000      80
    2538   235.330000     159
    2539   110.250000     110
    2540   123.730000     225
    2541   144.840000      89
    2542   125.540000     117
    2543    80.670000      81
    2544   101.550000      85
    2545   147.260000     169
    2546   160.800000     130
    2547    42.020000      53
    2548   103.930000      79
    2549    66.460000      40
    2550   191.150000     110
    2551    99.790000     125
    2552   155.050000     127
    2553   159.790000     180
    2554   141.380000     141
    2555   186.150000     120
    2556    79.260000      50
    2557   141.260000     170
    2558   107.540000     129
    2559   285.080000     226
    2560   111.390000     150
    2561   263.920000     101
    2562   157.280000     135
    2563   281.830000      76
    2564    47.480000      36
    2565   243.180000     115
    2566   155.550000     106
    2567   246.420000     198
    2568   169.250000      99
    2569   157.450000     101
    2570    70.920000      56
    2571    52.900000      82
    2572   121.960000      85
    2573    99.290000     129
    2574  1453.580000     300
    2575    49.380000      30
    2576   177.910000     150
    2577   144.110000     202
    2578   183.240000     121
    2579   182.790000     102
    2580   227.730000     183
    2581   200.300000      75
    2582   338.810000      90
    2583   163.160000     243
    2584   140.350000      70
    2585   255.880000     128
    2586   120.290000      65
    2587   153.470000     107
    2588   142.120000     199
    2589   128.520000     139
    2590   129.700000     109
    2591   229.670000     169
    2592   184.930000     119
    2593    65.860000      73
    2594    91.300000     135
    2595    95.850000     170
    2596    80.880000      67
    2597   122.180000      90
    2598   131.580000     109
    2599    51.980000      80
    2600   102.060000     110
    2601   118.780000     124
    2602   111.720000     150
    2603   113.290000     108
    2604   121.090000     149
    2605    94.810000     100
    2606   118.940000     250
    2607   167.950000     158
    2608   115.840000      52
    2609   281.070000     169
    2610   155.380000     162
    2611   103.100000      92
    2612   136.760000     750
    2613   162.120000      86
    2614   327.830000      99
    2615   172.020000      97
    2616   198.810000     125
    2617    17.400000      16
    2618    94.070000      89
    2619   102.190000      94
    2620   393.490000     179
    2621    88.120000     109
    2622   280.300000     101
    2623   135.080000     101
    2624   108.330000     132
    2625   211.320000     525
    2626   146.510000     300
    2627    80.070000     119
    2628    67.690000      20
    2629   177.760000     128
    2630    57.550000      35
    2631   224.950000     223
    2632   107.950000     150
    2633    14.860000      14
    2634   111.910000      80
    2635    73.610000     105
    2636   189.610000     128
    2637   117.840000     100
    2638    61.530000      65
    2639   133.020000     235
    2640    84.360000      40
    2641    88.690000     130
    2642   123.070000     165
    2643    63.710000      66
    2644    90.370000      83
    2645    38.870000      35
    2646   203.660000     200
    2647   202.590000     145
    2648    38.130000      20
    2649   126.230000      80
    2650   112.770000      85
    2651   202.650000      80
    2652   141.750000      95
    2653    76.530000      45
    2654   129.900000     150
    2655   141.210000     150
    2656    64.880000     120
    2657   112.250000      85
    2658   116.480000      94
    2659   160.160000     116
    2660    46.700000      36
    2661   302.370000     700
    2662    89.710000      79
    2663   153.280000      90
    2664   153.260000      99
    2665   115.320000     112
    2666    82.320000      45
    2667    14.560000      14
    2668    57.730000      30
    2669    33.740000      49
    2670   147.430000     325
    2671    37.110000      70
    2672    58.630000      45
    2673    76.850000      90
    2674   109.620000     125
    2675   105.140000      40
    2676   113.000000      91
    2677   150.650000     210
    2678   142.490000      80
    2679    50.820000     699
    2680   171.070000     141
    2681    82.000000      98
    2682    42.680000      40
    2683   131.790000      89
    2684    52.590000      89
    2685   162.940000     150
    2686    60.440000      66
    2687    41.760000      65
    2688   144.140000     176
    2689   314.650000     181
    2690    63.240000      45
    2691   124.740000     135
    2692    93.220000      40
    2693    85.380000      53
    2694    67.280000      40
    2695   100.220000      51
    2696   113.080000     135
    2697   124.360000      69
    2698    55.900000      98
    2699   113.470000     181
    2700   148.950000     110
    2701   111.210000     109
    2702   149.900000     140
    2703    49.170000      45
    2704   203.060000      93
    2705  1113.290000     135
    2706    39.450000      37
    2707   152.200000     110
    2708    40.840000      32
    2709    84.540000     127
    2710   173.010000     114
    2711    61.840000      75
    2712   216.470000      94
    2713   139.270000      75
    2714   166.650000     101
    2715    48.210000      36
    2716   184.040000     126
    2717    85.210000      78
    2718   112.340000    1000
    2719   145.350000      90
    2720   242.800000     103
    2721    46.070000     100
    2722   120.770000     213
    2723   121.280000     142
    2724   184.280000      70
    2725   107.530000     150
    2726   154.490000     171
    2727   124.740000     103
    2728   205.500000     243
    2729   158.610000     293
    2730   157.480000     110
    2731    90.170000     163
    2732   488.920000      95
    2733   127.720000     132
    2734    41.430000      38
    2735   140.770000     100
    2736   131.020000     450
    2737    60.790000      90
    2738    44.110000      27
    2739   265.140000     250
    2740    39.460000      94
    2741   129.270000      90
    2742   118.830000     185
    2743   161.770000     123
    2744   103.330000     110
    2745   100.740000     100
    2746    56.020000      45
    2747   115.320000      90
    2748    53.510000      50
    2749   189.600000      89
    2750   142.270000     235
    2751   189.860000     232
    2752    84.480000      50
    2753   100.760000      60
    2754   173.630000      47
    2755   188.440000     117
    2756    71.880000      50
    2757   132.600000     129
    2758    91.140000      81
    2759    70.610000      50
    2760   107.260000     328
    2761   139.150000     250
    2762   150.070000      75
    2763    71.410000      59
    2764   113.550000     126
    2765    42.160000      22
    2766    58.150000      65
    2767    48.030000      13
    2768    89.420000      94
    2769   115.700000      57
    2770    55.800000      42
    2771   126.130000      80
    2772    87.940000      48
    2773   140.860000     170
    2774   181.100000     120
    2775   167.600000     150
    2776   323.870000     132
    2777    82.030000      68
    2778   140.710000     171
    2779    50.360000      50
    2780   175.460000      69
    2781   190.050000      96
    2782   145.440000     106
    2783    62.340000     120
    2784    46.510000      38
    2785   189.040000      65
    2786   139.150000     125
    2787    94.830000      62
    2788   214.570000      79
    2789    78.330000      39
    2790   196.670000     125
    2791    60.210000      90
    2792    72.270000      61
    2793    14.400000      15
    2794    75.930000      50
    2795    67.540000      90
    2796   126.410000      90
    2797   181.480000     102
    2798   131.670000     140
    2799    79.470000      60
    2800    83.510000      25
    2801   157.970000     100
    2802    61.700000      65
    2803   245.080000     100
    2804   221.280000     384
    2805   259.110000     130
    2806   138.630000     150
    2807    95.720000     140
    2808   197.080000     161
    2809   333.250000     123
    2810   131.450000     115
    2811   149.860000     115
    2812    56.180000     130
    2813    37.330000      31
    2814   125.030000     112
    2815   119.160000     116
    2816   138.170000      99
    2817   261.880000     106
    2818   107.500000     170
    2819   147.900000     200
    2820   129.540000     120
    2821    91.790000     175
    2822   110.150000     150
    2823    82.930000      81
    2824   120.310000     109
    2825   177.800000     220
    2826    45.200000      48
    2827   159.890000     120
    2828    41.600000      42
    2829   127.530000      79
    2830    99.080000     500
    2831    85.380000      48
    2832    46.010000      59
    2833    99.700000     116
    2834    83.610000     124
    2835   126.100000      45
    2836   142.880000     284
    2837   189.290000     108
    2838   171.460000     102
    2839   121.910000      80
    2840   175.920000     150
    2841   131.010000     250
    2842   150.770000     119
    2843   143.810000     149
    2844   112.020000     247
    2845   191.640000     180
    2846   164.470000      79
    2847   240.090000     178
    2848    81.530000     136
    2849    97.230000      55
    2850    75.080000      80
    2851   262.150000     120
    2852   196.540000      70
    2853   160.360000     110
    2854    80.160000      90
    2855   191.440000      89
    2856   119.190000      89
    2857    99.940000     150
    2858   212.640000     250
    2859   408.950000     150
    2860   114.800000      91
    2861   136.430000      60
    2862    78.010000     237
    2863    92.480000      83
    2864   142.340000     100
    2865   175.340000      65
    2866   140.220000     203
    2867   143.340000      95
    2868   146.420000      88
    2869   215.060000      60
    2870    70.150000      40
    2871   141.650000      68
    2872    65.050000     110
    2873    53.690000      60
    2874    58.700000      65
    2875    45.270000      38
    2876   105.430000     130
    2877    50.240000     150
    2878    69.260000      55
    2879    91.910000      69
    2880   120.890000      85
    2881    92.430000     150
    2882   187.100000     249
    2883    49.270000      46
    2884    77.720000      57
    2885    56.440000      50
    2886    56.330000      84
    2887   105.710000     125
    2888   199.750000     300
    2889    50.780000      60
    2890    58.180000     900
    2891   361.950000     150
    2892   817.670000     249
    2893   100.740000      93
    2894    91.570000     230
    2895    61.940000     213
    2896   123.740000     125
    2897   130.320000     110
    2898    65.200000     149
    2899   194.230000    2000
    2900   292.700000     130
    2901   202.480000     113
    2902    77.600000      70
    2903    95.910000     100
    2904   157.810000     161
    2905   416.870000      60
    2906   101.730000      79
    2907    82.310000     107
    2908    45.990000      58
    2909   158.790000     189
    2910    22.990000      16
    2911   128.650000     599
    2912    35.890000      60
    2913    53.090000      35
    2914    44.690000     600
    2915   101.160000     100
    2916   148.560000      78
    2917    19.080000      16
    2918   172.600000     190
    2919    66.960000      45
    2920    46.430000      55
    2921   269.070000     599
    2922   216.640000     100
    2923   213.030000      50
    2924    43.150000      59
    2925    95.540000     160
    2926   125.260000     175
    2927    55.650000      61
    2928    49.150000      58
    2929    59.080000      61
    2930   162.250000      85
    2931   165.400000     172
    2932   134.860000     125
    2933   126.880000      81
    2934    95.430000      68
    2935    87.950000     100
    2936   136.550000     159
    2937   218.330000     350
    2938    95.790000      89
    2939   120.380000      65
    2940   138.130000     129
    2941   134.910000     255
    2942   117.020000      95
    2943   276.350000      70
    2944    76.870000      70
    2945    91.680000      96
    2946   144.170000     100
    2947    53.010000      75
    2948   102.300000      63
    2949   148.880000      99
    2950    81.980000      90
    2951   144.880000     155
    2952    53.190000      50
    2953   167.020000      90
    2954    39.920000      31
    2955   138.750000      81
    2956    62.250000      59
    2957    88.110000     399
    2958   176.840000     999
    2959   146.500000      86
    2960    95.930000      66
    2961    86.120000      68
    2962   108.170000     134
    2963   153.130000     109
    2964    64.700000      50
    2965   109.830000      90
    2966   104.060000     100
    2967   161.320000      65
    2968    37.600000      55
    2969   121.410000      97
    2970   119.890000     254
    2971   149.760000      90
    2972   178.400000     180
    2973   180.370000     200
    2974    45.180000      40
    2975   162.950000     259
    2976   189.360000      75
    2977   233.570000     100
    2978   149.960000     105
    2979   150.120000      88
    2980   188.880000     125
    2981    83.360000      61
    2982   123.600000     100
    2983   114.220000      82
    2984   548.460000     125
    2985   167.720000     125
    2986   273.300000      45
    2987   127.810000     121
    2988   110.300000     149
    2989   103.490000     203
    2990    14.880000      15
    2991   111.870000     192
    2992   157.220000     118
    2993    75.860000      80
    2994   121.860000     200
    2995    44.060000      55
    2996    88.350000      46
    2997   101.140000     110
    2998    75.630000      65
    2999   397.090000     250
    3000   104.530000      50
    3001   145.900000      50
    3002   200.890000     120
    3003   219.120000     271
    3004   252.450000     112
    3005   120.490000     700
    3006   217.830000     115
    3007   117.590000     154
    3008   136.100000     145
    3009    95.850000     148
    3010   132.190000     405
    3011    60.350000      58
    3012   147.070000      99
    3013   118.400000     185
    3014    47.120000      35
    3015    98.840000     110
    3016   117.220000      75
    3017    35.610000      55
    3018   151.330000     175
    3019   108.540000     299
    3020   168.500000     115
    3021   145.360000      90
    3022   101.850000     169
    3023    93.060000      49
    3024   127.750000     535
    3025   397.090000     100
    3026    46.600000      40
    3027   145.740000     140
    3028   134.930000     300
    3029   137.580000     179
    3030   202.590000     750
    3031    62.260000      50
    3032   164.750000     126
    3033    45.700000      47
    3034   155.040000      75
    3035    69.680000      70
    3036   120.480000     155
    3037    59.930000      40
    3038   182.780000     189
    3039   173.060000     102
    3040    96.080000      80
    3041   101.090000      79
    3042   127.390000      63
    3043    72.580000     100
    3044   123.190000      79
    3045    52.620000      32
    3046   237.770000     129
    3047    97.730000     132
    3048   110.760000      55
    3049    49.140000      75
    3050   145.490000     110
    3051    68.330000      51
    3052    48.480000      50
    3053    61.970000      33
    3054    81.760000     102
    3055   152.390000      82
    3056    58.710000      25
    3057   149.320000     270
    3058   176.007000     105
    3059   159.210000     104
    3060   156.510000     183
    3061   150.900000     175
    3062   133.850000     116
    3063   130.560000      66
    3064   109.460000      53
    3065    80.760000      75
    3066    58.440000      50
    3067   179.170000     149
    3068   165.360000     199
    3069    96.910000      66
    3070   102.030000     126
    3071   122.480000     110
    3072   341.210000      65
    3073   162.410000     145
    3074   206.960000     285
    3075    57.430000      50
    3076   175.760000      97
    3077    34.680000      23
    3078    60.580000     110
    3079   126.280000      69
    3080    78.380000      37
    3081   717.460000     229
    3082   177.680000     249
    3083   127.470000     145
    3084    70.400000      88
    3085    41.320000      70
    3086   149.090000     133
    3087   193.570000      61
    3088   134.620000      93
    3089   106.870000     239
    3090   382.670000      80
    3091   135.880000     231
    3092    82.670000     158
    3093    67.240000      50
    3094    82.870000      55
    3095   141.500000      94
    3096   146.430000     109
    3097    90.580000      45
    3098   195.070000     100
    3099   100.070000      89
    3100   112.970000     250
    3101    90.570000      65
    3102   185.740000     240
    3103    14.420000      14
    3104    51.900000      59
    3105    43.250000      45
    3106   117.840000      50
    3107   204.770000      74
    3108   121.930000     350
    3109   137.750000     350
    3110    54.490000      61
    3111    54.210000     110
    3112   153.500000     179
    3113   118.790000      85
    3114    54.870000      40
    3115    51.820000      48
    3116   189.880000     140
    3117    62.760000      51
    3118   118.950000     121
    3119    84.270000      16
    3120    93.380000      50
    3121    56.400000      48
    3122   151.740000     151
    3123   117.110000     118
    3124   543.930000      78
    3125   144.170000     197
    3126   219.220000     105
    3127    62.970000      65
    3128    86.250000      50
    3129   155.750000     110
    3130   160.690000     280
    3131   178.270000     160
    3132   156.830000     399
    3133   113.500000      78
    3134   143.030000     164
    3135   178.930000     171
    3136   319.160000     259
    3137   110.640000     160
    3138   107.970000     155
    3139   127.430000     136
    3140    79.900000     150
    3141   116.490000     106
    3142   167.240000     172
    3143   154.890000     111
    3144    71.180000      90
    3145   119.300000     101
    3146    40.350000      35
    3147   158.650000     141
    3148   185.240000     129
    3149   124.910000     100
    3150   265.810000     132
    3151    78.530000      57
    3152   102.410000     199
    3153    49.430000      65
    3154    81.260000      30
    3155    52.000000      49
    3156   145.150000     100
    3157    98.300000      85
    3158    48.060000      55
    3159   205.990000      80
    3160    65.390000      80
    3161    60.880000      53
    3162    55.500000     150
    3163   147.100000     234
    3164   131.080000     121
    3165   116.800000     400
    3166    50.160000      40
    3167   101.400000      33
    3168   113.730000      88
    3169   108.560000      68
    3170   201.170000      81
    3171   160.780000     330
    3172   182.630000     148
    3173    47.870000      60
    3174    98.530000      90
    3175   133.300000     168
    3176   107.980000     271
    3177   187.270000     145
    3178    50.410000      31
    3179   130.780000      70
    3180   165.320000      70
    3181   162.620000      85
    3182    56.290000      51
    3183   149.540000     250
    3184    42.860000      25
    3185    47.040000      48
    3186   125.890000      59
    3187   147.380000     119
    3188    52.960000     150
    3189   145.400000      82
    3190   216.160000     110
    3191   434.700000      40
    3192    52.620000      55
    3193   169.790000     120
    3194   148.730000      99
    3195   160.070000     180
    3196   176.850000      89
    3197    89.150000     151
    3198   234.540000      97
    3199   123.910000     160
    3200   100.440000      80
    3201   153.620000     186
    3202    49.630000      55
    3203   145.830000     107
    3204    56.630000     107
    3205   100.630000      99
    3206   110.360000     975
    3207    51.440000      76
    3208   128.350000     138
    3209   100.820000      89
    3210   201.390000     389
    3211    81.990000     150
    3212   136.750000     132
    3213    85.430000      80
    3214    81.210000      45
    3215   171.940000     139
    3216   148.380000     135
    3217   191.630000     120
    3218   114.760000     140
    3219   139.010000      99
    3220    49.230000     120
    3221   107.380000      55
    3222    88.680000      80
    3223   198.940000     145
    3224    66.300000      50
    3225   121.690000      78
    3226   109.430000     117
    3227    44.650000      32
    3228   113.750000      54
    3229    99.570000      99
    3230   146.590000     164
    3231   218.900000     123
    3232    73.240000      90
    3233    64.360000      45
    3234   159.390000     105
    3235    59.970000      23
    3236   100.240000     111
    3237    74.320000      59
    3238    62.330000      55
    3239   106.940000     150
    3240   121.410000      64
    3241   185.270000     154
    3242   114.930000      99
    3243   260.300000     106
    3244   142.440000     134
    3245    51.000000      55
    3246   127.370000     150
    3247   113.080000     101
    3248   186.000000     150
    3249   204.030000     250
    3250   203.500000     111
    3251   127.570000     193
    3252   188.570000     243
    3253   203.460000     216
    3254   132.500000     110
    3255    45.260000      59
    3256   121.300000      88
    3257   105.570000      90
    3258   250.450000     130
    3259   183.450000     198
    3260   124.610000     189
    3261   185.100000     169
    3262   119.200000     109
    3263   324.800000     499
    3264   135.060000     213
    3265   175.150000     418
    3266   153.230000      53
    3267   162.840000      45
    3268   207.180000      99
    3269    91.640000      52
    3270   144.130000     121
    3271   119.930000      45
    3272    39.110000      34
    3273    57.790000      69
    3274   226.050000      79
    3275   142.870000      76
    3276    86.560000      80
    3277   133.870000     135
    3278    18.590000      14
    3279   138.090000      78
    3280   111.370000      79
    3281    68.360000      70
    3282   190.480000     111
    3283    43.470000      55
    3284   165.408000     140
    3285    69.640000     100
    3286   104.910000     125
    3287    73.850000      60
    3288   250.710000     178
    3289   225.890000     158
    3290   114.170000     100
    3291   130.570000      80
    3292   103.770000      75
    3293    49.310000      80
    3294    72.070000      95
    3295   358.800000     400
    3296    93.840000     100
    3297   196.760000     105
    3298    50.450000     225
    3299   283.670000     293
    3300   159.440000      87
    3301    54.630000     100
    3302    55.860000     285
    3303    43.430000      55
    3304   109.810000     106
    3305    31.960000      35
    3306   145.890000      90
    3307    94.500000      70
    3308    74.260000      83
    3309   159.530000     109
    3310   193.010000     278
    3311   547.850000      92
    3312   150.960000     160
    3313    94.750000      80
    3314   142.660000     125
    3315   118.800000      97
    3316    35.240000      25
    3317   159.880000     181
    3318    56.180000      50
    3319   133.320000     176
    3320    67.830000      88
    3321   301.170000     125
    3322   178.290000     110
    3323   141.910000      79
    3324   136.830000     100
    3325   108.500000      80
    3326    40.110000      55
    3327    92.070000     120
    3328   196.620000     134
    3329    57.750000      60
    3330   222.610000     190
    3331   102.130000     120
    3332    70.200000      65
    3333   124.070000    1200
    3334   173.350000     150
    3335    26.930000      28
    3336   134.700000     271
    3337   285.020000      82
    3338   139.350000     106
    3339    45.620000      37
    3340    64.490000      79
    3341    87.210000     180
    3342   201.470000      80
    3343   467.280000     175
    3344   124.290000     175
    3345    56.990000      52
    3346   117.520000     149
    3347    99.060000      80
    3348   178.810000     120
    3349   111.310000      80
    3350    54.950000      85
    3351    80.890000     110
    3352   410.700000     250
    3353    93.040000      91
    3354   176.830000      77
    3355   125.170000     143
    3356   118.230000     119
    3357   206.920000     153
    3358   131.450000      99
    3359   228.010000     249
    3360   139.940000     143
    3361   126.380000      90
    3362   103.870000     950
    3363   353.700000      66
    3364    79.830000     800
    3365   240.390000     300
    3366    92.260000      31
    3367   206.820000     356
    3368    62.600000      55
    3369   147.130000     132
    3370   237.490000      66
    3371    59.210000      58
    3372   174.220000     114
    3373    98.850000     128
    3374    59.590000      64
    3375   152.510000     250
    3376   162.100000     115
    3377   131.920000     286
    3378    78.130000     110
    3379   135.100000     129
    3380   184.710000     147
    3381   340.300000     450
    3382    93.450000     115
    3383   188.860000     101
    3384   166.550000     135
    3385    57.290000      38
    3386    61.180000     197
    3387   127.630000     180
    3388   194.030000     400
    3389   122.590000     144
    3390   431.770000     453
    3391   142.330000      82
    3392    44.080000      36
    3393    48.400000     150
    3394   270.710000      70
    3395   153.990000     142
    3396   137.100000     129
    3397   115.460000      70
    3398   130.160000      93
    3399    61.070000      90
    3400   166.130000     692
    3401   189.790000      90
    3402    74.490000     125
    3403   109.990000     169
    3404   174.710000     650
    3405   165.600000      89
    3406   141.480000     200
    3407    63.010000      45
    3408    60.020000      49
    3409    51.280000      51
    3410   162.750000     449
    3411   153.240000      85
    3412    49.940000      40
    3413   145.730000     103
    3414   100.360000     129
    3415    80.440000     158
    3416    92.910000      45
    3417   147.400000     417
    3418   164.450000     695
    3419    51.590000      75
    3420   149.680000     188
    3421   154.410000     159
    3422    84.270000      59
    3423   146.650000      65
    3424    80.050000      70
    3425   431.440000     117
    3426   108.230000     105
    3427    64.190000      60
    3428   216.150000     170
    3429    85.160000      63
    3430   117.760000      50
    3431   141.390000     155
    3432   115.290000     100
    3433    42.870000      41
    3434   523.120000     250
    3435   136.080000     350
    3436   119.990000      50
    3437   133.980000     200
    3438    81.700000     149
    3439   127.250000      62
    3440    96.230000     350
    3441   111.990000     180
    3442   143.500000      57
    3443    61.830000      50
    3444    96.840000     132
    3445   156.470000     130
    3446    59.890000      55
    3447   210.370000      41
    3448   152.560000     148
    3449   209.500000     198
    3450   102.590000      80
    3451    68.950000      66
    3452    47.830000      45
    3453   107.420000     175
    3454   141.800000     108
    3455    78.350000      88
    3456   135.590000      60
    3457   108.920000     120
    3458    88.160000      80
    3459    45.080000      30
    3460    83.430000      90
    3461    47.840000      70
    3462   119.220000     110



```python
print(RF.feature_importances_)
```

    [3.49485065e-02 5.52795588e-01 3.06983597e-01 8.47451382e-04
     2.08692245e-02 6.69762745e-05 1.60611662e-03 1.37117836e-05
     1.61847254e-04 2.06447380e-06 9.89014565e-03 1.75604721e-04
     7.64774336e-05 1.97518907e-03 6.16174719e-05 1.13805908e-02
     4.89340896e-04 6.48525444e-05 1.64691540e-05 1.14722246e-05
     6.71617090e-04 4.52839928e-05 9.91469591e-06 2.64478415e-04
     2.77154539e-05 1.52885236e-05 2.48097589e-03 1.03697932e-06
     9.95086850e-05 9.62443064e-06 3.04869725e-04 2.64413927e-05
     7.93294231e-03 3.59262738e-05 1.65911602e-05 9.71925765e-06
     1.04791309e-04 5.91318198e-04 1.37516522e-03 2.84050905e-04
     5.80193906e-05 1.23990155e-04 7.00465836e-05 4.15865868e-04
     3.19069582e-06 1.52873701e-06 1.14004002e-03 9.88832366e-05
     6.31753419e-04 9.56690785e-07 1.71196909e-05 4.49019815e-03
     1.54181666e-06 1.73608331e-04 4.01510772e-05 2.86900322e-04
     5.87600720e-04 2.66723472e-03 2.92284322e-06 1.02310046e-05
     2.62753399e-07 2.81513525e-06 1.93843441e-06 4.08217316e-05
     1.34861899e-05 1.48255250e-05 4.05948112e-05 3.09427906e-06
     9.72882549e-05 6.96511812e-04 1.20627313e-06 9.42895075e-06
     8.99514600e-05 1.03021673e-05 2.17010226e-05 5.09567098e-03
     1.30378165e-04 1.37294779e-03 1.77779055e-04 3.81644072e-06
     5.74957531e-05 4.01755068e-04 6.94265340e-06 5.96310787e-06
     5.97095821e-05 8.28610249e-04 5.65372985e-04 1.11272638e-05
     1.12656186e-06 2.35583569e-04 1.20123992e-04 5.20080600e-04
     1.78108918e-04 5.44459670e-05 3.68095438e-04 3.53091977e-05
     1.02098585e-04 1.31204709e-05 5.13113339e-05 2.37653591e-05
     4.75475674e-05 1.24131336e-04 3.16300971e-04 9.66795693e-06
     1.53781149e-05 1.59096060e-05 6.22298646e-07 6.22393487e-04
     4.45457606e-06 9.47076348e-06 2.20860142e-04 2.98818929e-03
     4.82812937e-06 2.31630098e-04 1.32685856e-06 6.60648652e-06
     9.17417330e-05 4.23699421e-04 1.67674757e-03 3.75887758e-06
     7.72808986e-05 1.32885707e-05 1.55697409e-04 5.67067633e-04
     3.79455010e-06 6.07867694e-07 8.08694490e-04 1.86710474e-03
     2.55385523e-05 5.22976775e-03 6.57577537e-06 1.66939406e-05
     1.19944662e-05 4.94373885e-04 6.17427890e-06 5.09612556e-04
     1.20399180e-03 1.12933929e-04 4.40163567e-06 3.34417320e-05
     9.61420625e-05 1.59435794e-05 7.61685808e-05 8.23958379e-05
     2.67974853e-03 2.91218675e-05 2.53220672e-05]



```python
RFAssesment_dict = {"Feature_Importance":RF.feature_importances_, "Feature_Name": selected_df.iloc[: , 7:-1].columns}
RFAssesment_df = pd.DataFrame(RFAssesment_dict)
print(RFAssesment_df)
```

         Feature_Importance                         Feature_Name
    0          3.494851e-02                    reviews_per_month
    1          5.527956e-01                             latitude
    2          3.069836e-01                            longitude
    3          8.474514e-04                          Shared room
    4          2.086922e-02                      Entire home/apt
    5          6.697627e-05                           Hotel room
    6          1.606117e-03                         Private room
    7          1.371178e-05                      Agincourt North
    8          1.618473e-04         Agincourt South-Malvern West
    9          2.064474e-06                            Alderwood
    10         9.890146e-03                                Annex
    11         1.756047e-04                    Banbury-Don Mills
    12         7.647743e-05                       Bathurst Manor
    13         1.975189e-03                  Bay Street Corridor
    14         6.161747e-05                      Bayview Village
    15         1.138059e-02                Bayview Woods-Steeles
    16         4.893409e-04                 Bedford Park-Nortown
    17         6.485254e-05              Beechborough-Greenbrook
    18         1.646915e-05                              Bendale
    19         1.147222e-05                Birchcliffe-Cliffside
    20         6.716171e-04                          Black Creek
    21         4.528399e-05                          Blake-Jones
    22         9.914696e-06                 Briar Hill-Belgravia
    23         2.644784e-04    Bridle Path-Sunnybrook-York Mills
    24         2.771545e-05                      Broadview North
    25         1.528852e-05                  Brookhaven-Amesbury
    26         2.480976e-03      Cabbagetown-South St.James Town
    27         1.036979e-06                   Caledonia-Fairbank
    28         9.950869e-05                            Casa Loma
    29         9.624431e-06               Centennial Scarborough
    30         3.048697e-04                Church-Yonge Corridor
    31         2.644139e-05                  Clairlea-Birchmount
    32         7.932942e-03                         Clanton Park
    33         3.592627e-05                           Cliffcrest
    34         1.659116e-05               Corso Italia-Davenport
    35         9.719258e-06                             Danforth
    36         1.047913e-04                   Danforth East York
    37         5.913182e-04                   Don Valley Village
    38         1.375165e-03                          Dorset Park
    39         2.840509e-04  Dovercourt-Wallace Emerson-Junction
    40         5.801939e-05                 Downsview-Roding-CFB
    41         1.239902e-04                       Dufferin Grove
    42         7.004658e-05                    East End-Danforth
    43         4.158659e-04             Edenbridge-Humber Valley
    44         3.190696e-06                        Eglinton East
    45         1.528737e-06                     Elms-Old Rexdale
    46         1.140040e-03                  Englemount-Lawrence
    47         9.888324e-05       Eringate-Centennial-West Deane
    48         6.317534e-04                  Etobicoke West Mall
    49         9.566908e-07                      Flemingdon Park
    50         1.711969e-05                    Forest Hill North
    51         4.490198e-03                    Forest Hill South
    52         1.541817e-06               Glenfield-Jane Heights
    53         1.736083e-04                    Greenwood-Coxwell
    54         4.015108e-05                            Guildwood
    55         2.869003e-04                           Henry Farm
    56         5.876007e-04                      High Park North
    57         2.667235e-03                    High Park-Swansea
    58         2.922843e-06                       Highland Creek
    59         1.023100e-05                    Hillcrest Village
    60         2.627534e-07             Humber Heights-Westmount
    61         2.815135e-06                        Humber Summit
    62         1.938434e-06                           Humbermede
    63         4.082173e-05                   Humewood-Cedarvale
    64         1.348619e-05                              Ionview
    65         1.482552e-05           Islington-City Centre West
    66         4.059481e-05                        Junction Area
    67         3.094279e-06             Keelesdale-Eglinton West
    68         9.728825e-05                         Kennedy Park
    69         6.965118e-04                 Kensington-Chinatown
    70         1.206273e-06        Kingsview Village-The Westway
    71         9.428951e-06                       Kingsway South
    72         8.995146e-05                           L'Amoreaux
    73         1.030217e-05                   Lambton Baby Point
    74         2.170102e-05                     Lansing-Westgate
    75         5.095671e-03                  Lawrence Park North
    76         1.303782e-04                  Lawrence Park South
    77         1.372948e-03                   Leaside-Bennington
    78         1.777791e-04                      Little Portugal
    79         3.816441e-06                          Long Branch
    80         5.749575e-05                              Malvern
    81         4.017551e-04                           Maple Leaf
    82         6.942653e-06                        Markland Wood
    83         5.963108e-06                             Milliken
    84         5.970958e-05  Mimico (includes Humber Bay Shores)
    85         8.286102e-04                          Morningside
    86         5.653730e-04                            Moss Park
    87         1.112726e-05                         Mount Dennis
    88         1.126562e-06    Mount Olive-Silverstone-Jamestown
    89         2.355836e-04                  Mount Pleasant East
    90         1.201240e-04                  Mount Pleasant West
    91         5.200806e-04                          New Toronto
    92         1.781089e-04                     Newtonbrook East
    93         5.444597e-05                     Newtonbrook West
    94         3.680954e-04                              Niagara
    95         3.530920e-05                      North Riverdale
    96         1.020986e-04                  North St.James Town
    97         1.312047e-05                    O'Connor-Parkview
    98         5.131133e-05                             Oakridge
    99         2.376536e-05                      Oakwood Village
    100        4.754757e-05                        Old East York
    101        1.241313e-04              Palmerston-Little Italy
    102        3.163010e-04                    Parkwoods-Donalda
    103        9.667957e-06                 Pelmo Park-Humberlea
    104        1.537811e-05             Playter Estates-Danforth
    105        1.590961e-05                        Pleasant View
    106        6.222986e-07                   Princess-Rosethorn
    107        6.223935e-04                          Regent Park
    108        4.454576e-06                      Rexdale-Kipling
    109        9.470763e-06                    Rockcliffe-Smythe
    110        2.208601e-04                         Roncesvalles
    111        2.988189e-03                  Rosedale-Moore Park
    112        4.828129e-06                                Rouge
    113        2.316301e-04         Runnymede-Bloor West Village
    114        1.326859e-06                               Rustic
    115        6.606487e-06                  Scarborough Village
    116        9.174173e-05                       South Parkdale
    117        4.236994e-04                      South Riverdale
    118        1.676748e-03                 St.Andrew-Windfields
    119        3.758878e-06                              Steeles
    120        7.728090e-05                  Stonegate-Queensway
    121        1.328857e-05               Tam O'Shanter-Sullivan
    122        1.556974e-04                        Taylor-Massey
    123        5.670676e-04                          The Beaches
    124        3.794550e-06         Thistletown-Beaumond Heights
    125        6.078677e-07                     Thorncliffe Park
    126        8.086945e-04                    Trinity-Bellwoods
    127        1.867105e-03                           University
    128        2.553855e-05                     Victoria Village
    129        5.229768e-03    Waterfront Communities-The Island
    130        6.575775e-06                            West Hill
    131        1.669394e-05               West Humber-Clairville
    132        1.199447e-05                  Westminster-Branson
    133        4.943739e-04                               Weston
    134        6.174279e-06                   Weston-Pellam Park
    135        5.096126e-04                     Wexford/Maryvale
    136        1.203992e-03                      Willowdale East
    137        1.129339e-04                      Willowdale West
    138        4.401636e-06     Willowridge-Martingrove-Richview
    139        3.344173e-05                               Woburn
    140        9.614206e-05                    Woodbine Corridor
    141        1.594358e-05                     Woodbine-Lumsden
    142        7.616858e-05                             Wychwood
    143        8.239584e-05                       Yonge-Eglinton
    144        2.679749e-03                       Yonge-St.Clair
    145        2.912187e-05              York University Heights
    146        2.532207e-05                   Yorkdale-Glen Park


# Random Forest Model Evaluation


```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_RFpred)
```




    63186.99898229457




```python
from sklearn.metrics import r2_score
r2_score(y_test, y_RFpred)
```




    -0.07660328151077911




```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_RFpred)
```




    80.77298512850129



# Decision Tree Classifier 


```python

```


```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
for y
clf.fit(X_train, y_train)
y_clfpred = clf.predict(X_test)
prediction_dictclf = {"y_pred":y_clfpred, "y_test":y_test}
prediction_clfdf = pd.DataFrame(prediction_dictclf)
print(predictions_clfdf) 




```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-46-3ac1d0a87b11> in <module>
          5 prediction_dictclf = {"y_pred":y_clfpred, "y_test":y_test}
          6 prediction_clfdf = pd.DataFrame(prediction_dictclf)
    ----> 7 print(predictions_clfdf)
          8 
          9 


    NameError: name 'predictions_clfdf' is not defined



```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_clfpred)
print(cm)
accuracy_score(y_test, y_clfpred)
```


```python
graph_x = X_train[:, 139]
plt.scatter(graph_x, y_train, color = 'red')

#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price Vs Predictors')
plt.xlabel('Predictors')
plt.ylabel('Price per night')
plt.show()
#(y_pred.reshape(len(y_pred),1)
```

##  Function that takes an input sample vector (one-hot encoded for locations) and gives us back the corresponding location




```python

def vec_to_name(vector, index_to_name_dict):

    for index, value in enumerate(vector):
        if value == 1:
            return index_to_name_dict[index]

neighbourhood_dictionary = {}
index_neighbourhood_dict = {}
for index, name in enumerate(neighbourhood_names):
    neighbourhood_dictionary[name] = index
    index_neighbourhood_dict[index] = name


```

## Visualization of the training results


```python
#index_neighbourhood_dict
#ax = fig.add_axes(len(X_svr))
location_svr = [vec_to_name(sample, ) for sample in X_svr]
fig=plt.figure() #Creates a new figure
ax1=fig.add_subplot(111);
line1 = ax1.plot(location_svr, y_svr)
plt.show() 
```


```python

indices_svr = [neighbourhood_dictionary[n] for n in location_svr]
plt.scatter(indices_svr, y_svr)
plt.show()
```


```python
print( index_neighbourhood_dict)
```


```python

plt.scatter(X_train, y_train, color = 'magenta')
plt.plot(X_test, classifier.predict(X_test), color = 'green')
plt.title('Support Vector Regression Model')
plt.xlabel('location')
plt.ylabel('peice per night')
plt.show()
```

# Mapping the listings


```python
from bokeh.io import output_file, output_notebook, show
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, LogColorMapper, BasicTicker, ColorBar,
    DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)
from bokeh.models.mappers import ColorMapper, LinearColorMapper
from bokeh.palettes import Viridis5


map_options = GMapOptions(lat=selected_df.iloc[: , 8].values, lng=selected_df.iloc[: , 9].values, map_type="roadmap", zoom=6)

plot = GMapPlot(
    x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options
)
plot.title.text = "Hey look! It's a scatter plot on a map!"

# For GMaps to function, Google requires you obtain and enable an API key:
#
#     https://developers.google.com/maps/documentation/javascript/get-api-key
#
# Replace the value below with your personal API key:
plot.api_key = "AIzaSyBYrbp34OohAHsX1cub8ZeHlMEFajv15fY"

source = ColumnDataSource(
    data=dict(
        lat=housing.latitude.tolist(),
        lon=housing.longitude.tolist(),
        size=housing.median_income.tolist(),
        color=housing.median_house_value.tolist()
    )
)
max_median_house_value = housing.loc[housing['median_house_value'].idxmax()]['median_house_value']
min_median_house_value = housing.loc[housing['median_house_value'].idxmin()]['median_house_value']

#color_mapper = CategoricalColorMapper(factors=['hi', 'lo'], palette=[RdBu3[2], RdBu3[0]])
#color_mapper = LogColorMapper(palette="Viridis5", low=min_median_house_value, high=max_median_house_value)
color_mapper = LinearColorMapper(palette=Viridis5)

circle = Circle(x="lon", y="lat", size="size", fill_color={'field': 'color', 'transform': color_mapper}, fill_alpha=0.5, line_color=None)
plot.add_glyph(source, circle)

color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0))
plot.add_layout(color_bar, 'right')

plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
#output_file("gmap_plot.html")
output_notebook()

show(plot)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-47-ddf39b52cf80> in <module>
          8 
          9 
    ---> 10 map_options = GMapOptions(lat=selected_df.iloc[: , 8].values, lng=selected_df.iloc[: , 9].values, map_type="roadmap", zoom=6)
         11 
         12 plot = GMapPlot(


    /opt/conda/lib/python3.8/site-packages/bokeh/model.py in __init__(self, **kwargs)
        232         kwargs.pop("id", None)
        233 
    --> 234         super().__init__(**kwargs)
        235         default_theme.apply_to_model(self)
        236 


    /opt/conda/lib/python3.8/site-packages/bokeh/core/has_props.py in __init__(self, **properties)
        245 
        246         for name, value in properties.items():
    --> 247             setattr(self, name, value)
        248 
        249     def __setattr__(self, name, value):


    /opt/conda/lib/python3.8/site-packages/bokeh/core/has_props.py in __setattr__(self, name, value)
        272 
        273         if name in props or (descriptor is not None and descriptor.fset is not None):
    --> 274             super().__setattr__(name, value)
        275         else:
        276             matches, text = difflib.get_close_matches(name.lower(), props), "similar"


    /opt/conda/lib/python3.8/site-packages/bokeh/core/property/descriptors.py in __set__(self, obj, value, setter)
        537             raise RuntimeError("%s.%s is a readonly property" % (obj.__class__.__name__, self.name))
        538 
    --> 539         self._internal_set(obj, value, setter=setter)
        540 
        541     def __delete__(self, obj):


    /opt/conda/lib/python3.8/site-packages/bokeh/core/property/descriptors.py in _internal_set(self, obj, value, hint, setter)
        758 
        759         '''
    --> 760         value = self.property.prepare_value(obj, self.name, value)
        761 
        762         old = self.__get__(obj, obj.__class__)


    /opt/conda/lib/python3.8/site-packages/bokeh/core/property/bases.py in prepare_value(self, obj_or_cls, name, value)
        329                     break
        330             else:
    --> 331                 raise e
        332         else:
        333             value = self.transform(value)


    /opt/conda/lib/python3.8/site-packages/bokeh/core/property/bases.py in prepare_value(self, obj_or_cls, name, value)
        322         try:
        323             if validation_on():
    --> 324                 self.validate(value)
        325         except ValueError as e:
        326             for tp, converter in self.alternatives:


    /opt/conda/lib/python3.8/site-packages/bokeh/core/property/bases.py in validate(self, value, detail)
        454                 nice_join([ cls.__name__ for cls in self._underlying_type ]), value, type(value).__name__
        455             )
    --> 456             raise ValueError(msg)
        457 
        458     def from_json(self, json, models=None):


    ValueError: expected a value of type Real, got [43.64105 43.66724 43.69602 ... 43.76668 43.6733  43.64481] of type ndarray



```python

```
