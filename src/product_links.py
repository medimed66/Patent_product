import pandas as pd
from typing import List, Dict, Any, Optional
import time
import os
import random
from playwright.sync_api import sync_playwright, Page
from playwright_stealth import stealth_sync

# Configuration constants
SAVE_FREQUENCY = 3  # Save after processing this many products
QUERY_SUFFIXES = ["", "datasheet", "technical specifications", "user manual"]
COLUMN_NAMES = ["Product Link", "Datasheet Link", "Specifications Link", "User Manual Link"]

# Headers for requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

def get_google_result(page: Page, query: str, first_search: bool = False) -> Optional[str]:
    """
    Get the first Google search result using an existing Playwright page.
    Retries up to max_tries times if an exception occurs.
    
    Parameters:
        page: The Playwright page to use
        query: The search query
        first_search: Flag indicating if this is the first search (go to Google homepage) or 
                    a subsequent search (use existing search bar)
    """
    # For first search only, navigate to Google homepage
    if first_search:
        page.goto("https://www.google.com/?hl=en")
    
    # Find the search input element
    search_selectors = [
        "input[name='q']",
        "textarea[name='q']",
        "input[title='Search']",
        "[aria-label='Search']"
    ]
    
    search_input = None
    for selector in search_selectors:
        try:
            if page.locator(selector).count() > 0:
                search_input = page.locator(selector)
                break
        except Exception:
            continue
    
    if not search_input:
        # If we can't find the search bar, as a fallback, go to Google homepage
        if not first_search:
            page.goto("https://www.google.com/?hl=en")
            for selector in search_selectors:
                try:
                    if page.locator(selector).count() > 0:
                        search_input = page.locator(selector)
                        break
                except Exception:
                    continue
        if not search_input:
            # If we still can't find it, raise an error
            print("Error: Unable to find search input element on Google homepage.")
            return None
    
    # Clear and type with delay between keystrokes to simulate typing
    search_input.click()
    search_input.fill("")
    for char in query:
        search_input.type(char, delay=random.randint(10, 40))
    search_input.press("Enter")
    
    # Find the first result heading and then locate its parent <a> tag to get the href
    try:
        page.wait_for_selector("h3", timeout=5000)
        
        # First attempt: Look for the wrapping <a> tag that contains h3
        first_result_url = page.evaluate('''() => {
            const h3Element = document.querySelector('h3');
            if (h3Element) {
                // Find the closest ancestor <a> element
                const linkElement = h3Element.closest('a');
                if (linkElement && linkElement.href) {
                    return linkElement.href;
                }
            }
            return null;
        }''')
        
        # If first method fails, try alternative selector
        if not first_result_url:
            # Try to find search result link directly
            first_result_url = page.evaluate('''() => {
                const resultLinks = Array.from(document.querySelectorAll('a[href]')).filter(
                    link => link.href &&
                            link.href.startsWith('http') &&
                            !link.href.includes('google.com') &&
                            link.querySelector('h3')
                );
                return resultLinks.length > 0 ? resultLinks[0].href : null;
            }''')
    except Exception as e:
        print(f"Error extracting URL: {str(e)}")
        return None
    if not first_result_url:
        print("Could not extract URL from first search result")
        return None
    return first_result_url

def process_products(products_data: List[Dict], company: str, df: pd.DataFrame, output_file: str) -> List[Dict]:
    """Process multiple products using a single browser instance."""
    results = []
    
    with sync_playwright() as p:
        try:
            # Launch a single browser for all queries
            browser = p.chromium.launch(headless=False)
            
            # Create a single page (tab) for all searches
            page = browser.new_page()
            stealth_sync(page)
            first_search = True
            
            # Process each product
            for i, product_data in enumerate(products_data):
                product_name = product_data['Product Name']
                missing_columns = product_data['missing_columns']
                
                print(f"\nProcessing product: {product_name}")
                result = {'Product Name': product_name}
                
                # Process each missing column
                for suffix, column in zip(QUERY_SUFFIXES, COLUMN_NAMES):
                    if column in missing_columns:
                        # Create query
                        query = f"{company} {product_name}"
                        if suffix:
                            query += f" {suffix}"
                        
                        try:
                            link = get_google_result(page, query, first_search=first_search)
                            first_search = False
                            result[column] = link
                        except Exception as e:
                            print(f"Error getting {column} for '{product_name}': {str(e)}")
                            result[column] = None
                            if not page.is_closed():
                                page.close()
                            browser.close()
                            time.sleep(5)
                            browser = p.chromium.launch(headless=False)
                            page = browser.new_page()
                            stealth_sync(page)
                            first_search = True
                        
                        # Small delay between searches
                        time.sleep(random.uniform(0.5, 2))
                
                results.append(result)
                
                if (i + 1) % SAVE_FREQUENCY == 0:
                    # Save progress every SAVE_FREQUENCY products
                    print(f"Saving progress after processing {i + 1} products...")
                    df = update_dataframe(df, results)
                    df.to_csv(output_file, sep='\t', index=False)
            
            # Close the page and browser when all products are processed
            if not page.is_closed():
                page.close()
            browser.close()
            
        except Exception as e:
            print(f"Browser error: {str(e)}")
            # Return the results we have so far
            
    return results

def update_dataframe(df: pd.DataFrame, results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Update the dataframe with the new results."""
    for result in results:
        product_name = result['Product Name']
        # Find the index of this product in the dataframe
        idx = df.index[df['Product Name'] == product_name].tolist()
        if idx:
            # Update each column in the result
            for col, value in result.items():
                if col != 'Product Name' and col != 'error':  # Skip the product name and error columns
                    df.at[idx[0], col] = value
    return df

def main():
    # Get company name from user
    company = input("Enter company name to search for (e.g., Honeywell): ").strip()
    if not company:
        company = "Honeywell"  # Default
        print(f"Using default company name: {company}")

    # Define input and output file paths
    input_file = input("Enter input file path (e.g., Data/products.tsv): ").strip()
    if not input_file:
        input_file = '../Data/honeywell_patents_products.tsv'
        print(f"Using default input file: {input_file}")
        
    output_dir = os.path.dirname(input_file)
    if not output_dir:
        output_dir = '.'
    
    output_file = os.path.join(output_dir, f"{company.lower()}_products_links.tsv")
    
    # Check if the output file exists, if not create it from the original product list
    if not os.path.exists(output_file):
        print(f"Creating new output file: {output_file}")
        
        try:
            # Load product data from the original source
            product_patent_df = pd.read_csv(input_file, sep='\t')
            product_list = product_patent_df['Product Name'].unique()
            
            # Create a new dataframe with just the product names
            df = pd.DataFrame({'Product Name': product_list})
            
            # Add empty columns for the links
            for column in COLUMN_NAMES:
                df[column] = None
                
        except Exception as e:
            print(f"Error creating new dataframe: {str(e)}")
            return
    else:
        print(f"Loading existing output file: {output_file}")
        # Load existing dataframe
        try:
            df = pd.read_csv(output_file, sep='\t')
            
            # Make sure all required columns exist
            for column in COLUMN_NAMES:
                if column not in df.columns:
                    df[column] = None
                    
        except Exception as e:
            print(f"Error loading existing dataframe: {str(e)}")
            return
    
    # Find rows with missing values in any of the link columns
    missing_data = df[df[COLUMN_NAMES].isna().any(axis=1)].copy()
    
    # If no missing data, exit
    if missing_data.empty:
        print("No missing data to process.")
        return
    
    # For each row, determine which specific columns are missing
    missing_data['missing_columns'] = missing_data.apply(
        lambda row: [col for col in COLUMN_NAMES if pd.isna(row[col])], axis=1
    )
    
    print(f"Found {len(missing_data)} products with missing data.")
    
    # Ask user how many products to process
    try:
        limit = input("Enter number of products to process (leave blank for all): ").strip()
        if limit:
            limit = int(limit)
            missing_data = missing_data.head(limit)
            print(f"Will process {len(missing_data)} products.")
    except ValueError:
        print("Invalid input. Processing all products.")
    
    total_processed = 0
    products_to_process = missing_data.to_dict('records')
    
    try:
        # Process all products with a single browser instance
        print(f"Starting processing of {len(products_to_process)} products...")
        results = process_products(products_to_process, company, df, output_file)
        
        # Update the dataframe with all results
        df = update_dataframe(df, results)
        total_processed = len(results)
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
        df.to_csv(output_file, sep='\t', index=False)
        print(f"Progress saved to {output_file}. Processed {total_processed} products.")
        return
    
    # Final save
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\nProcessing completed. Data saved to {output_file}")

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        elapsed_time = time.time() - start_time
        print(f"Total execution time: {elapsed_time:.2f} seconds")