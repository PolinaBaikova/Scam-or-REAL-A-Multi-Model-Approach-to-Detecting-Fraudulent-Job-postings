# Load the necessary libraries
import sqlite3
import csv


# Establish a connection to the SQLite database (or create it if it doesn't exist)
con = sqlite3.connect("Fake_or_Real_Jobs.db")

# Create a cursor object to interact with the database
cur = con.cursor()

# Create Job_Posts table 
cur.execute('''
    CREATE TABLE IF NOT EXISTS Job_Posts (
        Job_id INTEGER PRIMARY KEY,
        Title TEXT, 
        Location TEXT,
        Department TEXT, 
        Salary TEXT,
        Company_profile TEXT, 
        Description TEXT,
        Requirements TEXT,
        Benefits TEXT,
        Telecommuting INTEGER,
        Logo INTEGER,
        Has_questions INTEGER,
        Employment_type TEXT,
        Required_experience TEXT,
        Required_education TEXT,
        Industry TEXT,
        Job_function TEXT,
        Fraudulent INTEGER
    );
''')

# Commit the changes (new table created) to the database
con.commit()


# Using 'utf-8' encoding to handle special characters (punctuation, brackets ext)
with open("fake_job_postings.csv", encoding='utf-8') as table_file:
    reader = csv.reader(table_file)
    
    next(reader, None)  # Skip the header row

    # Insert rows into the Job_Posts table
    cur.executemany('''
        INSERT INTO Job_Posts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', reader)
    

# Commit changes to the database
con.commit()


# Execute a  query to count the total number of rows in the Job_Posts table
cur.execute('SELECT COUNT(*) FROM Job_Posts;')

# Get the result of the query
total_jobs = cur.fetchone()[0]

# Print the total number of job postings
print("Total number of job posts:", total_jobs)