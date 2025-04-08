# SCAM or REAL? A Multi-Model Approach to Detecting Fraudulent Job postings

Shiny app for additional data exploration https://polinashiny.shinyapps.io/myapp/
 
# Introduction 
Online job platforms have become a primary way for people to find employment, but they have also become a target for scammers posting fake job ads. These fraudulent postings can trick job seekers into sharing personal information or paying unnecessary fees. In this project, we analyze a dataset of job postings to explore the characteristics that distinguish real jobs from scams. 

The data was sourced from https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs

# Dataset Overview & Database Setup

The data was provided in CSV format and was loaded into a local SQLite database named Fake_or_Real_Jobs.db using Python, with 'utf-8' encoding to correctly handle special characters such as punctuation and brackets.

The database contains one main table, Job_Posts, with 17,880 job postings and 18 fields (all text or integer types). Job_id serves as the unique identifier.

Built-in sqlite3 package in Python is used to connect to the database, create SQL queries, and retrieve data for analysis.

Several columns such as Title, Department, Description, Company_profile, Benefits and Requirements are highly unique and descriptive. Columns such as Employment_type, Required_experience, Required_education, Industry and Job_function have a smaller number of categories, while others like Telecommuting, Logo, Has_questions and Fraudulent are binary features.

<img src="images/unique_values.png" alt="Unique Values" width="250" height="400">

# Data Cleaning and Processing

### Handling Missing Values

Many features had significant missing data. For the purposes of this project, missing values were retained and marked as 'unspecified', treating the absence of information as a potential signal of fraud.

<img src="images/missing_values.png" alt="Missing values" width="250" height="400">

### Location field transformation

The original Location field was split into Country, State, and City for more granular analysis. This allowed us to focus on regional patterns and reduce complexity during modeling.

<img src="images/jobs_by_city.png" alt="Jobs by city" width="400" height="250">

### Filtering for U.S. Job Postings

The dataset included postings from 90 countries, but over 10,000 entries were from the U.S. To focus on a consistent subset that is more relevant from the personal perspective, a new table called Job_Posts_US was created, containing only U.S.-based postings and all fields from the original Job_Posts table except for Department and Salary, which were excluded due to having over 70% of missing values.

### Using Descriptive Fields As Binary Indicators

‘Requirements’, ‘Description’, ‘Company_profile’, and ‘Benefits’ with highly diverse and descriptive values were not modified. Instead, they will be used as binary indicators to capture whether or not this information is present in the job posting. 


