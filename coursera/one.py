import requests
from bs4 import BeautifulSoup
import pandas as pd

# Send an HTTP request to the webpage
url = 'https://en.wikipedia.org/wiki/Cloud-computing_comparison'  
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Print the title of the webpage to verify
print("Title: " + soup.title.text)

# Find the table containing the data (selecting the first table by default)
table = soup.find('table')

# Extract table rows
rows = table.find_all('tr')

# Extract headers from the first row (using <th> tags)
headers = [header.text.strip() for header in rows[0].find_all('th')]

# Loop through the rows and extract data (skip the first row with headers)
data = []
for row in rows[1:]:  # Start from the second row onwards
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    data.append(cols)

# Convert the data into a pandas DataFrame, using the extracted headers as column names
df = pd.DataFrame(data, columns=headers)

# Display the first few rows of the DataFrame to verify
print(df.head())  

# Save the DataFrame to a CSV file
df.to_csv('scraped_data.csv', index=False)