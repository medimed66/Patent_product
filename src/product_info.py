from urllib.parse import urlparse
from bs4 import BeautifulSoup
from io import BytesIO
import pandas as pd
import requests
import pdfplumber
from PyPDF2 import PdfReader
import json
import re
import os
import time
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import torch
import gc

SAVE_FREQ = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 16384,
    device_map = "auto",
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)
model.eval()

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPU(s)")
    
    for i in range(device_count):
        # Get memory allocated and total memory
        allocated = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        print(f"\nGPU {i}:")
        print(f"  Allocated Memory: {allocated:.2f} GB")
        print(f"  Total VRAM: {total:.2f} GB")
        print(f"  Utilization: {allocated/total*100:.1f}%")
else:
    print("No GPU available, using CPU")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

description_system_prompt = f"""You are a text analysis and summarization system.
You will be given a list of bullet points about a certain product, extracted from various sources.

Your task is :
1. Analyze the extracted bullet points about the specified product and filter out any repeated, redundant, or contradictory information.
2. Create a single and complete description about the product that includes all remaining relevant bullet points about the product
3. The description must clearly describe what the product is and what it does.
4. Do no hallucinate or add any other information or subjective assessment that wasn't explicitly provided in the bullet points.

Formatting instructions:
- The output should be a continuous block of text that describes and gives an overview of the product.
- The description must begin with : "The [product name] is/are ..."
- The description must incorporate all unique and relevant information from the provided bullet points.
- The description must be clear, accurate, and factual without any unnecessary repetition.
- Do not include any meta comments like introductions, conclusions or suggestions.
- If the provided text is empty, don't return anything."""

bulletpoint_system_prompt = """You are a text analysis and information extraction system.
Your job is to analyze the provided raw text scraped from some website, and extract any descriptive information about the specified products

Your task:
1. Extract descriptive information about the specified products in the form of bullet points.
2. Focus the extraction process on:
    - What the product is.
    - What the product does.
    - What the product can be used for.
    - Any specifications or features about the product.
    - Anything relevant to identify related patents or technologies used to developp the product.
3. Identify and discard text that is irrelevant to the specified products.
4. Discard information that is related to the website itself
5. do not hallucinate or add any information that is not present in the text.
6. do not make subjective assessments or claims.
7. Ignore any other product that isn't explicitly cited in the list of relevant products.

**Formatting instructions:**
- For each product from the specified list, create a list of bullet points, each containing a different set of informations.
- Start each product section with "## PRODUCT: [product name]"
- Then, list bullet points with factual information. Each bullet point MUST:
    - Start with the "*" symbol and be on a new line.
    - Contain some unique information extracted from the source text.
- If the provided text is empty or irrelevant to one of the specified products, write:
    ## PRODUCT: [product name]
    * NO RELEVANT INFORMATION
- Do not include any meta comments like introductions, conclusions or suggestions."""

def is_pdf_link(url):
    """Check if the URL points to a PDF file"""
    return url.lower().endswith('.pdf') or 'pdf' in urlparse(url).path.lower()

def extract_text_from_pdf(content, password=None):
    # Try pdfplumber first (faster for most cases)
    try:
        with pdfplumber.open(BytesIO(content)) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        print(f"pdfplumber failed (falling back to PyPDF2): {e}")
    # Fallback to PyPDF2 for encrypted PDFs
    try:
        pdf_reader = PdfReader(BytesIO(content))
        if pdf_reader.is_encrypted:
            pdf_reader.decrypt(password or "")
        return "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    except Exception as e:
        print(f"PyPDF2 failed: {e}")
        return ""

def extract_text_from_html(html_content):
    """Extract meaningful text from HTML content"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Get text
        text = soup.get_text(separator=' ', strip=True)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception as e:
        print(f"Error extracting text from HTML: {e}")
        return ""

def scrape_url(url):
    """Scrape content from URL, whether HTML or PDF"""
    if not url or pd.isna(url) or url.strip() == "":
        return ""
    response = requests.get(url, headers=HEADERS, timeout=30)
    if response.status_code == 200:
        if is_pdf_link(url):
            return extract_text_from_pdf(response.content)
        else:
            return extract_text_from_html(response.content)
    else:
        print(f"Failed to fetch {url}: Status code {response.status_code}")
        return ""

def chunk_text(text, chunk_size=4096, overlap=100):
    chars_per_token = 4
    char_chunk_size = chunk_size * chars_per_token
    char_overlap = overlap * chars_per_token

    if len(text) <= char_chunk_size:
        return [text]

    # Split into sentences, handling edge cases
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    
    if not sentences:
        return [text]

    chunks = []
    i = 0
    
    while i < len(sentences):
        current_chunk_sentences = []
        current_length = 0
        
        # Add sentences until we reach the chunk size limit
        while i < len(sentences):
            sentence = sentences[i]
            # Calculate length if we add this sentence (including space separator)
            additional_length = len(sentence) + (1 if current_chunk_sentences else 0)
            
            if current_length + additional_length <= char_chunk_size:
                current_chunk_sentences.append(sentence)
                current_length += additional_length
                i += 1
            else:
                # If we haven't added any sentences, the sentence is too long - add it anyway
                if not current_chunk_sentences:
                    current_chunk_sentences.append(sentence)
                    i += 1
                break
        
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))
            
            # Calculate overlap for next chunk
            if i < len(sentences):  # More sentences to process
                overlap_length = 0
                overlap_count = 0
                
                # Count sentences from the end that fit in overlap
                for j in range(len(current_chunk_sentences) - 1, -1, -1):
                    sentence_length = len(current_chunk_sentences[j])
                    test_length = overlap_length + sentence_length + (1 if overlap_count > 0 else 0)
                    
                    if test_length <= char_overlap:
                        overlap_length = test_length
                        overlap_count += 1
                    else:
                        break
                
                # Backtrack to include overlap sentences in next iteration
                i -= overlap_count
    
    return chunks

def extract_bulletpoints(chunks, product_names):
    product_list = "\n- " + "\n- ".join(product_names)
    all_descriptions = {product: [] for product in product_names}
    tokens_per_product = 256
    max_new_tokens = tokens_per_product * len(product_names)
    print(f"Processing {len(chunks)} chunks")
    for chunk in chunks:
        if not chunk.strip():
            continue
        gc.collect()
        for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        with torch.inference_mode():
            user_prompt = f"List of Relevant Products:{product_list}\n\nText to analyze:\n\"\"\"\n{chunk}\n\"\"\"\n\nExtracted bullet points:\n"
            messages = [
                {"role": "system", "content": bulletpoint_system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, padding=False, truncation=False, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            print(f"Processing chunk with {input_len} input tokens")
            if torch.cuda.is_available():
                print(f"GPU memory before chunk {i+1}: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            generation = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                cache_implementation="offloaded",
            )
            gen_cpu = generation.cpu()[0]
            tokens = gen_cpu[input_len:].tolist()
            response = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            del generation, inputs, input_text, messages, user_prompt, tokens, gen_cpu
        print(response)
        # Parse the response to extract information for each product
        current_product = None
        for line in response.splitlines():
            line = line.strip()
            
            # Detect new product section
            if line.lower().startswith("## product:"):
                product_name_raw = line[len("## PRODUCT:"):].strip()
                matched_product = next((p for p in product_names if product_name_raw.lower() == p.lower()), None)
                current_product = matched_product if matched_product else None
                continue
            
            if not current_product:
                continue
            
            # Collect bullet points
            if line.startswith("*") and "NO RELEVANT INFORMATION" not in line.upper():
                all_descriptions[current_product].append(line)
    
    # Compile results
    results = {}
    for product, descriptions in all_descriptions.items():
        if descriptions:
            results[product] = "\n".join(descriptions)
        else:
            results[product] = ""
    return results

def create_description(bulletpoints, product_name):
    gc.collect()
    try:
        messages = [
            {"role": "system", "content": description_system_prompt},
            {"role": "user", "content": f"Product: {product_name}\n\nBulletpoints to analyze:\n\"\"\"\n{bulletpoints}\n\"\"\"\n\nDescription:\n"}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, truncation=True, max_length=16384, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {k: v.contiguous().to(model_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=2048,
                cache_implementation="offloaded"
            )
            gen_cpu = generation.cpu()[0]
            del generation
            tokens = gen_cpu[input_len:].tolist()
            response = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            del gen_cpu, tokens, inputs
        print(f"Generated description for {product_name}: {response}")
        return response
    except Exception as e:
        print(f"Error generating final description for product {product_name}: {e}")
        return ""

def load_links_data(filepath):
    """Load and prepare product links from TSV file."""
    links_df = pd.read_csv(filepath, sep='\t')
    links_df['Links'] = links_df['Links'].apply(eval)
    return links_df

def load_product_data(json_path, products):
    """Load existing product data or initialize if not exists."""
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    
    # Ensure all products exist with non-empty fields
    for product in products:
        if product not in data or not data[product].get("Description") or not data[product].get("Bulletpoints"):
            data[product] = {
                "Bulletpoints": data.get(product, {}).get("Bulletpoints", "") or "",
                "Description": data.get(product, {}).get("Description", "") or ""
            }
    
    return data

def prepare_links_for_processing(links_df, links_tracking_path):
    """Prepare links data and tracking information."""
    # Create exploded DataFrame of links
    exploded_df = links_df.explode('Links')
    exploded_df = exploded_df[exploded_df['Links'].notna() & (exploded_df['Links'] != "")]
    
    # Load or initialize the processed links tracking
    if os.path.exists(links_tracking_path):
        processed_links_df = pd.read_csv(links_tracking_path, sep='\t')
    else:
        # Initialize a new dataframe to track processed links
        processed_links_df = exploded_df.copy()
        processed_links_df['Processed'] = False
        processed_links_df.to_csv(links_tracking_path, index=False, sep='\t')
    
    # Merge to identify which links still need processing
    merged_df = exploded_df.merge(
        processed_links_df[['Product', 'Links', 'Processed']], 
        on=['Product', 'Links'], 
        how='left'
    )
    
    # Fill NaN values in 'Processed' column with False
    merged_df['Processed'] = merged_df['Processed'].fillna(False)
    
    # Filter for links that haven't been processed
    unprocessed_links_df = merged_df[~merged_df['Processed']].copy()
    
    return unprocessed_links_df, processed_links_df

def generate_missing_descriptions(empty_products, json_path):
    """Generate descriptions for products with missing descriptions."""
    with open(json_path, "r", encoding="utf-8") as f:
        product_data = json.load(f)
    for i, product in enumerate(empty_products):
        print(f"Generating description for {product}")
        bulletpoints = product_data[product]["Bulletpoints"]
        if bulletpoints:
            description = create_description(bulletpoints, product)
            product_data[product]["Description"] = description
        else:
            print(f"No bulletpoints found for {product}")
        if (i + 1)% SAVE_FREQ == 0:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(product_data, f, indent=4)
    
    # Save the updated descriptions
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(product_data, f, indent=4)
    print("All missing descriptions have been generated.")

def process_links(unprocessed_grouped, data, processed_links_df, json_path, links_tracking_path):
    """Process each unprocessed URL and update data."""
    # Process each unique unprocessed URL
    total_links = len(unprocessed_grouped)
    for i, (url, product_list) in enumerate(zip(unprocessed_grouped['Links'], unprocessed_grouped['Product'])):
        gc.collect()
        print(f"Processing URL {i+1}/{total_links}: {url}")
        print(f"This URL is relevant for {len(product_list)} products: {product_list}")
        # Scrape and chunk the URL content
        try:
            text = scrape_url(url)
        except Exception as e:
            print(f"Error scraping URL {url}: {e}")
            continue
        if not text:
            print(f"No content found for URL: {url}")
            continue
        
        chunks= chunk_text(text)
        del text
        if len(chunks) > 3:
            print("Too many chunks generated, skipping this URL to avoid excessive processing.")
            continue
        try:
            bulletpoints_by_product = extract_bulletpoints(chunks, product_list)
            del chunks
        except Exception as e:
            print(f"Error extracting bulletpoints for URL {url}: {e}")
            continue
        # Add results to the all_product_bulletpoints dictionary
        for product, bulletpoints in bulletpoints_by_product.items():
            if bulletpoints:
                # Get existing bulletpoints from the data
                existing_bulletpoints = data.loc[product, 'Bulletpoints'] if product in data.index else ""
                
                # Append new bulletpoints
                if existing_bulletpoints:
                    data.loc[product, 'Bulletpoints'] = existing_bulletpoints + "\n" + bulletpoints
                else:
                    data.loc[product, 'Bulletpoints'] = bulletpoints
        
        # Mark this URL as processed for all products in the list
        processed_links_df.loc[(processed_links_df['Links'] == url), 'Processed'] = True
        
        # Save progress every SAVE_FREQ URLs
        if (i + 1) % SAVE_FREQ == 0:
            # Save updated product data
            data.to_json(json_path, indent=4, orient='index')
            # Save updated processed links tracking
            processed_links_df.to_csv(links_tracking_path, index=False, sep='\t')
    
    # Save final product data and processed links
    data.to_json(json_path, indent=4, orient='index')
    processed_links_df.to_csv(links_tracking_path, index=False, sep='\t')
    
    return processed_links_df

def main():
    start_time = time.time()
    
    # File paths
    links_filepath = '../Data/filtered_product_links.tsv'
    json_path = "../Data/products.json"
    links_tracking_path = "../Data/processed_links.tsv"
    
    # Step 1: Load links data
    links_df = load_links_data(links_filepath)
    #links_df = links_df.sample(2)
    products = links_df['Product'].unique()
    
    # Step 2: Load or initialize product data
    product_data = load_product_data(json_path, products)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(product_data, f, indent=4)
    del products
    
    # Step 3: Convert to DataFrame for easier processing
    data = pd.DataFrame.from_dict(product_data, orient='index')
    empty_products = data[data['Description'] == ""].index.tolist()
    del product_data
    
    # Step 4: Prepare links for processing
    unprocessed_links_df, processed_links_df = prepare_links_for_processing(links_df, links_tracking_path)
    # Step 5: Check if all links have been processed
    if unprocessed_links_df.empty:
        if empty_products:
            print("All links have been processed. Generating missing descriptions...")
            generate_missing_descriptions(empty_products, json_path)
        else:
            print("All links have been processed and all products have descriptions.")
    else:
        # Step 6: Process the unprocessed links
        unprocessed_grouped = unprocessed_links_df.groupby('Links')['Product'].apply(list).reset_index()
        processed_links_df = process_links(
            unprocessed_grouped, 
            data, 
            processed_links_df, 
            json_path, 
            links_tracking_path
        )
        # Step 7: find products with all links processed
        products_to_process = []
        for product in empty_products:
            product_links = links_df[links_df['Product'] == product]['Links'].tolist()
            if all(processed_links_df[processed_links_df['Links'].isin(product_links)]['Processed']):
                products_to_process.append(product)
        
        # Step 8: Generate descriptions if all links are now processed
        if products_to_process:
            print("All links have been processed. Generating missing descriptions...")
            generate_missing_descriptions(products_to_process, json_path)
    
    # Calculate and display elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()