# US Election Prediction

Using the data from https://github.com/MEDSL/2018-elections-unoffical, we attempt to classify whether a democrat or republican will win senate, house, and presidential elections in a given county.  This judgement is based on the demographic information from each county.

The csv file: election-context-2018.csv is the raw dataset containing the data that we used in creating our models.

The 'data' directory contains the trimmed datasets for the models to be trained on. Each one contains only features relevent to the classification problem at hand.

The python file 'data_parser.py' contains the script that is used to create the files in the data directory.  It uses the raw data from the csv file and trims each election dataset to only include helpful sets of features.

The python file 'model.py' is used to generate, train, and analyze the machine learning model.  It uses Keras and matplotlib.pyplot to accomplish these tasks.

The h5 file 'my_new_model.h5' is a copy of our final model and can be accessed using the command "tf.keras.models.load_model('/path/to/my_new_model.h5')" in python.

## Features

### cvap\_pct
- **Description**: citizen voting-age population / total population
- **Year/s**: 2012-2016 (ACS 5-Year Estimates)
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH), [Citizen Voting Age Population (CVAP) Special Tabulation From the 2012-2016 5-Year American Community Survey](https://www.census.gov/programs-surveys/decennial-census/about/voting-rights/cvap.html)

----------------

### white\_pct
- **Description**: non-Hispanic whites as a percentage of total population
- **Year/s**: 2012-2016 (ACS 5-Year Estimates)
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH)

----------------

### black\_pct
- **Description**: non-Hispanic blacks as a percentage of total population
- **Year/s**: 2012-2016 (ACS 5-Year Estimates)
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH)

----------------

### hispanic\_pct
- **Description**: Hispanics or Latinos as a percentage of total population
- **Year/s**: 2012-2016 (ACS 5-Year Estimates)
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH)

----------------

### foreignborn\_pct
- **Description**: foreign-born population as a percentage of total population
- **Year/s**: 2012-2016 (ACS 5-Year Estimates)
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH)

----------------

### female\_pct
- **Description**: females as a percentage of total population
- **Year/s**: 2012-2016 (ACS 5-Year Estimates)
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH)

----------------

### age29andunder\_pct
- **Description**: population 29 years or under as a percentage of total population
- **Year/s**: 2012-2016 (ACS 5-Year Estimates)
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH)

----------------

### age65andolder\_pct
- **Description**: population 65 years or older as a percentage of total population
- **Year/s**: 2012-2016 (ACS 5-Year Estimates)
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH)

----------------

### median\_hh\_inc
- **Description**: median household income in the past 12 months (in 2016 inflation-adjusted dollars)
- **Year/s**: 2012-2016 (ACS 5-Year Estimates)
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH)

----------------

### clf\_unemploy\_pct
- **Description**: unemployed population in labor force as a percentage of total population in civilian labor force
- **Year/s**: 2012-2016 (ACS 5-Year Estimates)
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH)

----------------

### lesshs\_pct
- **Description**: population with an education of less than a regular high school diploma as a percentage of total population
- **Year/s**: 2012-2016 (ACS 5-Year Estimates)
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH)

----------------

### lesscollege\_pct
- **Description**: population with an education of less than a bachelor's degree as a percentage of total population
- **Year/s**: 2012-2016 (ACS 5-Year Estimates)
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH)

----------------

### rural\_pct
- **Description**: rural population as a percentage of total population
- **Year/s**: 2010
- **Source**: [IPUMS NHGIS, University of Minnesota](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MLLQDH)

----------------

### is\_rural
- **Description**: is the county rural or urban, based on rural-urban continuum codes
- **Year/s**: 2013
- **Source**: [USDA Economic Research Service](https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/)
- **Coding**:

| Code | Description |
| --- | --- |
| 0 | Rural |
| 1 | Urban |

-----------------

## Label

### winner

| Code | Description |
| --- | --- |
| 0 | Democrat |
| 1 | Not Democrat |

-----------------
