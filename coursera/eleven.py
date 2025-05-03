import pandas as pd

data = {
    'Name': [],
    'Amount': [],
    'Date': []
}

df = pd.DataFrame(data)

df.to_csv('people.txt', sep='\t', index=False)

# Read the text file
df = pd.read_csv('people.txt', sep='\t')

name = input('Enter the name: ')
amount = input('Enter the amount: ')
date = input('Enter the date: ')

# Add a new row
new_row = pd.DataFrame([{'Name': f'{name}', 'Amount': amount, 'Date': f' {date}'}])
df = pd.concat([df, new_row], ignore_index=True)

# Save the updated file
df.to_csv('people.txt', sep='\t', index=False)