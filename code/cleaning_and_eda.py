# Load the necessary libraries
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import re

# Establish a connection to the SQLite database (or create it if it doesn't exist)
con = sqlite3.connect("Fake_or_Real_Jobs.db")
# Create a cursor object to interact with the database
cur = con.cursor()


# Execute the query to see the first 5 rows of the data from Job_Posts table 
cur.execute('''
    SELECT * FROM Job_Posts
    LIMIT 5;
''')
rows = cur.fetchall() # Fetch the rows
# Get column names from the cursor description
columns = [description[0] for description in cur.description]
# Create a DataFrame
df = pd.DataFrame(rows, columns=columns)
# Display the DataFrame
print(df)


# Get column names and types from Job_Posts table
cur.execute("PRAGMA table_info(Job_Posts);")
columns_info = cur.fetchall()


# Loop through each column and count unique values along with the data type
for col_id, col_name, col_type, *_ in columns_info:
    query = f"SELECT COUNT(DISTINCT {col_name}) FROM Job_Posts;"
    cur.execute(query)
    count = cur.fetchone()[0]
    print(f"'{col_name}'({col_type}): {count} unique values ")
       
# Loop through each column and count missing values (NULLs or empty strings)
for col in columns:
    query = f"SELECT COUNT(*) FROM Job_Posts WHERE {col} = '';"
    cur.execute(query)
    null_count = cur.fetchone()[0]
    print(f"'{col}': {null_count} missing values")
  
    
# Loop through each column and replace empty strings with 'unspecified'
for col in columns: 
    cur.execute(f"UPDATE Job_Posts SET {col} = 'unspecified' WHERE {col} = '';")
# Commit changes to the database
con.commit()


# Add a new column to store Country 
cur.execute("ALTER TABLE Job_Posts ADD COLUMN Country TEXT;")
# Add a new column to store State 
cur.execute("ALTER TABLE Job_Posts ADD COLUMN State TEXT;")
# Add a new column to store City 
cur.execute("ALTER TABLE Job_Posts ADD COLUMN City TEXT;")
# Commit the changes
con.commit()


# Extract the Country: everything before the first comma
cur.execute('''
    UPDATE Job_Posts
    SET Country = TRIM(SUBSTR(Location, 1, INSTR(Location, ',') - 1))
    WHERE INSTR(Location, ',') > 0;
''')

# Extract the State: the text between the first and second commas
cur.execute('''
    UPDATE Job_Posts
    SET State = TRIM(
        SUBSTR(
            Location,
            INSTR(Location, ',') + 1,
            INSTR(SUBSTR(Location, INSTR(Location, ',') + 1), ',') - 1
        )
    )
    WHERE INSTR(Location, ',') > 0;
''')

# Extract the City: the text after the second comma
cur.execute('''
    UPDATE Job_Posts
    SET City = TRIM(
        SUBSTR(
            Location,
            INSTR(Location, ',') + INSTR(SUBSTR(Location, INSTR(Location, ',') + 1), ',') + 1
        )
    )
    WHERE INSTR(Location, ',') > 0;
''')

# Save the updates to the table
con.commit()


# Get column names and their data types from the Job_Posts table
cur.execute("PRAGMA table_info(Job_Posts);")
columns_info = cur.fetchall()


# Print column name and type for each column
print("Columns and their data types in Job_Posts table:\n")
for col in columns_info:
    print(f"{col[1]}: {col[2]}")
# Get the names of all columns in the Job_Posts table
cur.execute("PRAGMA table_info(Job_Posts);")
columns = [info[1] for info in cur.fetchall()]
# Loop through each column and replace empty strings with 'unspecified'
for col in columns: 
    cur.execute(f"UPDATE Job_Posts SET {col} = 'unspecified' WHERE TRIM({col}) = '';")    
con.commit


# Query to count the number of unique countries in the Country column
cur.execute('''
    SELECT COUNT(DISTINCT Country)
    FROM Job_Posts;
''')
# Fetch the result of the query
unique_country_count = cur.fetchone()[0]
# Print the number of unique countries
print("Number of unique countries:", unique_country_count)

# Query to get the top 5 countries with the most job postings
cur.execute('''
    SELECT TRIM(Country) AS Country, COUNT(*) AS Count
    FROM Job_Posts
    GROUP BY Country
    ORDER BY Count DESC
    LIMIT 5;
''')
# Fetch and print the top 5 countries
top_countries = cur.fetchall()
print("\nTop 5 countries by number of job postings:")
for country, count in top_countries:
    print(f"{country}: {count}")


# Create a new table Job_Posts_US with only US-based job postings and selected columns
cur.execute('''
    CREATE TABLE Job_Posts_US AS
    SELECT
        Job_id,
        Title,
        Company_profile,
        Description,
        Requirements,
        Benefits,
        Telecommuting,
        Logo,
        Has_questions,
        Employment_type,
        Required_experience,
        Required_education,
        Industry,
        Job_function,
        Fraudulent,
        State,
        City
    FROM Job_Posts
    WHERE Country = 'US';
''')
# Commit the changes to the database
con.commit()


# Verify that the new table was created and check how many rows it contains
cur.execute('SELECT COUNT(*) FROM Job_Posts_US;')
us_count = cur.fetchone()[0]
print("Number of US job postings in Job_Posts_US:", us_count)


# List of columns to convert to lowercase for consistency
columns_to_lower = [
    'Title', 'Employment_type', 'Required_experience',
    'Required_education', 'Industry', 'Job_function', 'City'
]
# Loop through each column and apply the LOWER function to convert text to lowercase
for col in columns_to_lower:
    query = f'''
        UPDATE Job_Posts_US
        SET {col} = LOWER({col})
    '''
    cur.execute(query)
# Commit the changes to the database
con.commit()


# Get all job titles from the US-only table
cur.execute('''
    SELECT Title FROM Job_Posts_US;
''')
titles = cur.fetchall()


# Extract individual words from titles and count how often each word appears
words = []
for (title,) in titles:
    title_words = re.findall(r'\b\w+\b', title.lower())  # lowercase + extract words
    words.extend(title_words)

# Count frequency of each word across all titles
word_counts = Counter(words)
# Get the top 30 most common words found in job titles
top_words = word_counts.most_common(30)
# Print the most frequent words
print("Top 30 Most Common Words in Job Titles:")
for word, count in top_words:
    print(f"{word}: {count}")


# Get the most common full job titles with their counts and percentage of total
cur.execute('''
    SELECT 
        Title,
        COUNT(*) AS Title_Count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS Percentage
    FROM Job_Posts_US
    GROUP BY Title
    ORDER BY Title_Count DESC
    LIMIT 30;
''')
rows = cur.fetchall()
# Print top 30 most frequent job titles
print("\nTop 30 Titles by Job Count and Percentage:\n")
for title, count, pct in rows:
    print(f"{title}: {count} jobs ({pct:.2f}%)")


# Dictionary mapping keywords to standardized job title categories
title_updates = {
    'teacher': 'teacher',
    'customer service': 'customer service',
    'engineer': 'engineer',
    'sales': 'sales',
    'sale': 'sales',
    'manager': 'manager',
    'developer': 'developer',
    'data': 'data specialist',
    'supervisor': 'supervisor',
    'technician': 'technician',
    'marketing': 'marketing associate',
    'assistant': 'assistant',
    'analyst': 'analyst',
    'designer': 'designer',
    'director': 'director',
    'account': 'accountant',
    'driver': 'driver',
    'consultant': 'consultant'
}
# Loop through each keyword and update titles that contain it
for keyword, new_title in title_updates.items():
    cur.execute(f'''
        UPDATE Job_Posts_US
        SET Title = ?
        WHERE LOWER(Title) LIKE ?;
    ''', (new_title, f'%{keyword}%'))
# Save the changes to the database
con.commit()


# Query to get the top 20 most frequent job titles with their count and percentage
cur.execute('''
    SELECT 
        Title,
        COUNT(*) AS Title_Count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS Percentage
    FROM Job_Posts_US
    GROUP BY Title
    ORDER BY Title_Count DESC
    LIMIT 20
''')
# Fetch the result
rows = cur.fetchall()
# Print the top 20 titles and their frequencies
print("Top 20 Titles by Job Count and Percentage:\n")
for dept, count, pct in rows:
    print(f"{dept}: {count} jobs ({pct:.2f}%)")


# Store the top titles we want to keep in a list
top_titles = []
for title, count, pct in rows:
    top_titles.append(title)
# Create placeholders (?,?,?,...) for use in the SQL WHERE clause
placeholders = ','.join('?' for _ in top_titles)
# Update all other titles not in the top list to "Other"
cur.execute(f'''
    UPDATE Job_Posts_US
    SET Title = 'Other'
    WHERE LOWER(Title) NOT IN ({placeholders})
''', top_titles)
# Save the changes
con.commit()


def show_value_counts(column_name):
    # Query to get counts and percentages
    cur.execute(f'''
        SELECT 
            {column_name},
            COUNT(*) AS Count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS Percentage
        FROM Job_Posts_US
        GROUP BY {column_name}
        ORDER BY Count DESC;
    ''')
    rows = cur.fetchall()
    # Print results
    print(f"\nUnique values in '{column_name}' with frequency and percentage:\n")
    for value, count, pct in rows:
        print(f"{value}: {count} jobs ({pct:.2f}%)")
    
    return rows


# Employment type
employment_rows = show_value_counts("Employment_type")

# Required experience
experience_rows = show_value_counts("Required_experience")

# Required education
education_rows = show_value_counts("Required_education")


# Group all variations of 'vocational' under a single category
cur.execute('''
    UPDATE Job_Posts_US
    SET Required_education = 'vocational'
    WHERE Required_education LIKE '%vocational%';
''')
# Save the changes to the database
con.commit()


# Rename longer education level descriptions to simplified versions
cur.execute('''
    UPDATE Job_Posts_US
    SET Required_education = 
        CASE 
            WHEN Required_education = 'some college coursework completed' THEN 'some college'
            WHEN Required_education = 'some high school coursework' THEN 'some high school'
            ELSE Required_education
        END;
''')
# Commit the changes to the database
con.commit()

# Job function
function_rows = show_value_counts("Job_function")