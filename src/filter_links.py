from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()

def clean_df():
    links_df = pd.read_csv("../Data/honeywell_products_links.tsv", sep="\t")
    cleaned_links = pd.DataFrame()
    excluded_links = ["https://www.honeywell.com/us/en/patents"]
    cleaned_links['Product'] = links_df['Product Name']
    cleaned_links['Links'] = links_df.apply(lambda row: list({x for x in row[1:] if pd.notna(x) and x not in excluded_links}), axis=1)
    return cleaned_links

def save_progress(results):
    """Save the current progress to the output file"""
    pd.DataFrame(results).to_csv("../Data/filtered_product_links.tsv", sep="\t", index=False)

if __name__=="__main__":
    cleaned_links = clean_df()
    output_file = "../Data/filtered_product_links.tsv"
    
    # Initialize results list and track which products have been processed
    results = []
    processed_products = set()
    
    # Check if output file exists and load the already processed data
    if os.path.exists(output_file):
        print(f"Found existing output file. Loading processed data...")
        existing_results = pd.read_csv(output_file, sep="\t")
        results = existing_results.to_dict('records')
        processed_products = set(existing_results['Product'].tolist())
        print(f"Loaded {len(results)} previously processed products.")
    
    system_prompt = f"""
    You will be given a product name and a list of URL links that are supposedly related to the product.
    Your task is to identify and filter out any links that are clearly irrelevant to the specified product. 
    A link should only be removed if you can confidently determine—based on the URL alone—that it does not pertain to the product.
    Return a list of indices corresponding to relevant links.
    example : if only the first and third links are relevant return "1 3"
    if no link is relevant return "0"
    """
    
    # Define how often to save progress (e.g., every 10 products)
    save_interval = 10
    products_since_last_save = 0
    
    for i, row in cleaned_links.iterrows():
        product = row['Product']
        
        # Skip if this product has already been processed
        if product in processed_products:
            continue
        
        if pd.isna(row['Links']) or not row['Links'] or row['Links'] == "[]":
            results.append({"Product": product, "Links": []})
            products_since_last_save += 1
            continue
        links = row['Links']
        link_list = eval(links) if isinstance(links, str) else links
        formatted_links = "\n".join([f"{j+1}. {url}" for j, url in enumerate(link_list)])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f" Product : {product} \n\n List of links : {formatted_links} \n\n relevant Links : "}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        
        try:
            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=16)
                generation = generation[0][input_len:]
            
            response = tokenizer.decode(generation, skip_special_tokens=True).strip()
            
            if response == "0":
                results.append({"Product": product, "Links": []})
            else:
                indices = response.split()
                relevant_links = [link_list[int(index)-1] for index in indices if 0 < int(index) <= len(link_list)]
                results.append({"Product": product, "Links": relevant_links})
                
            # Mark as processed
            processed_products.add(product)
            products_since_last_save += 1
            
            # Save progress periodically
            if products_since_last_save >= save_interval:
                save_progress(results)
                products_since_last_save = 0
                
        except Exception as e:
            # Log the error but continue processing
            print(f"Error processing product '{product}': {e}")
            # Still save what we have so far
            save_progress(results)
            
    # Final save of all results
    save_progress(results)
    print("Processing complete!")