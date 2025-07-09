import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def scrape_honeywell_patents_table():
    """
    Scrape the patent-product table from Honeywell patents page and save as CSV.
    """
    # URL of the Honeywell patents page
    url = "https://www.honeywell.com/us/en/patents"
    
    # Headers to mimic browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    print(f"Accessing {url}...")
    
    try:
        # Get the page content
        response = requests.get(url, headers=headers)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"Failed to access the page. Status code: {response.status_code}")
            return None
            
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table containing patents and products
        # This is a general approach since we don't know the exact table structure
        tables = soup.find_all('table')
        
        if not tables:
            print("No tables found on the page. The page structure might have changed.")
            return None
            
        print(f"Found {len(tables)} tables on the page.")
        
        # Assuming the target table has Product and Patent columns
        # We'll examine each table to find the right one
        matching_tables = []
        
        for i, table in enumerate(tables):
            headers_row = table.find('tr')
            if not headers_row:
                continue
                
            headers = [th.text.strip() for th in headers_row.find_all(['th', 'td'])]
            print(f"Table {i+1} headers: {headers}")
            
            # Check if this table has both Product and Patent columns
            if 'Product Name' in headers and 'Associated Patent(s)' in headers:
                matching_tables.append((i, table, headers))
        
        if not matching_tables:
            print("Couldn't find a table with Product and Patent columns.")
            return None
        
        # Create Data directory if it doesn't exist
        if not os.path.exists("Data"):
            os.makedirs("Data")
        
        all_data = []
        
        for table_index, table, headers in matching_tables:
            print(f"\nProcessing table {table_index+1} with headers: {headers}")
            
            # Extract data from this table
            data = []
            rows = table.find_all('tr')
            
            # Extract data from each row
            for row in rows[1:]:  # Skip header row
                cols = row.find_all(['td', 'th'])
                if cols:
                    row_data = {headers[i]: col.text.strip() for i, col in enumerate(cols) if i < len(headers)}
                    # Add a source column to track which table this came from
                    row_data['Table Number'] = table_index + 1
                    data.append(row_data)
            
            print(f"Extracted {len(data)} rows from table {table_index+1}")
            
            # Add to combined data
            all_data.extend(data)
        
        # Create DataFrame
        patents_df = pd.DataFrame(all_data)
        
        # Display results
        print(f"\nSuccessfully extracted {len(patents_df)} rows of data.")
        
        # Save to CSV
        output_file = "../Data/honeywell_patents_products.tsv"
        patents_df.to_csv(output_file, index=False, sep='\t')
        print(f"Data saved to {output_file}")
        
        return patents_df
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    print("Starting Honeywell patents table scraper...")
    scrape_honeywell_patents_table()
    print("Scraping completed.")