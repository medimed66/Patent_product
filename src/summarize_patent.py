from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import pandas as pd
import torch
import os
import re
from typing import List, Any

SAVE_FREQ = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"
SEQ_LENGTH = 7000

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = SEQ_LENGTH,
    device_map = "auto",
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)
model.eval()

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

def truncate(tokenizer: Any, text: str, max_seq_length: int = SEQ_LENGTH) -> List[int]:
    """
    Tokenizes text and truncates at the end of the last complete sentence
    that fits within the token limit.
    
    Args:
        tokenizer: The tokenizer object (e.g., from transformers library)
        text: Input text to tokenize
        max_seq_length: Maximum number of tokens allowed
        special_tokens: Whether to add special tokens (CLS, SEP, etc.)
    
    Returns:
        List of token IDs truncated at sentence boundary
    """
    
    # Split text into sentences using multiple delimiters
    # This regex handles periods, exclamation marks, question marks
    # while avoiding common abbreviations and decimal numbers
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\!|\?)\s+'
    sentences = re.split(sentence_pattern, text.strip())
    
    # Clean up sentences - remove empty ones and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Start with empty result
    accumulated_text = ""
    
    for i , sentence in enumerate(sentences):
        # Try adding this sentence
        candidate_text = accumulated_text
        if candidate_text:
            candidate_text += " " + sentence
        else:
            candidate_text = sentence
        
        # Tokenize the candidate text
        candidate_tokens = tokenizer(
            candidate_text, 
            add_special_tokens=True,
            truncation=False,
            return_tensors="pt"
        )
        # Check if it fits within the limit
        if candidate_tokens["input_ids"].shape[-1] <= max_seq_length:
            # This sentence fits, keep it
            accumulated_text = candidate_text
        else:
            # This sentence would exceed the limit
            break
    
    return accumulated_text.strip()

system_prompt = """You are a patent summarization and information compression model.
Your will be given the title, abstract, claims, and description of a patent.
Your task is to summarize this lengthy patent text into a short, concise, and informative summary.
The summary must be highly informative and should include the most important aspects of the patent.
Instead of focusing on the technical details, focus on the application, use cases, and the problem it solves.
Keep in mind that the provided text might be cut in the middle of a sentence, so do not focus on completing the sentence, focus only on summarizing given the provided information.

Formatting instructions:
- The summary should be a continuous block of plain text.
- The summary should be no longer than 400 words.
- Don't include any information that is not present in the patent text.
- Do not include any meta-commentary like "This patent is about..." or "The summary of this patent is...".
"""

def merge_text(row):
    texts = []
    for col in text_columns:
        text = row[col]
        if pd.notna(text) and isinstance(text, str) and len(text) > 0:
            texts.append(col + ": " + text.strip())
    if texts:
        return '\n'.join(texts)
    else:
        return ""

if __name__ == "__main__":
    output_file = "../Data/summarized_patents.json"
    if os.path.exists(output_file):
        patent_df = pd.read_json(output_file, lines=True, dtype=str)
    else:
        patent_df = pd.read_json("../Data/patents.json", lines=True, dtype=str)
        patent_df = patent_df[patent_df['patent_id'].str.match(r'^\d')]
        text_columns = ["title", "abstract", "claims", "description"]
        patent_df['patent_text'] = patent_df.apply(merge_text, axis=1)
        patent_df.drop(columns=[ 'abstract', 'claims', 'description', 'publication_date'], inplace=True)
        patent_df["summary"] = ""
        patent_df.reset_index(drop=True, inplace=True)
        patent_df.to_json(output_file, orient='records', lines=True)
    rows_to_process = patent_df[patent_df["summary"] == ""]
    i=0
    for idx, row in rows_to_process.iterrows():
        print("\nprocessing:", row['title'])
        try:
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
            with torch.inference_mode():
                truncated_text = truncate(tokenizer, row['patent_text'])
                print('\n', truncated_text, '\n')
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": truncated_text}
                ]
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(input_text, return_tensors="pt", truncation=False)
                input_len = inputs["input_ids"].shape[-1]
                generation = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    cache_implementation="offloaded",
                )
                gen_cpu = generation.cpu()[0]
            del generation, inputs
            tokens = gen_cpu[input_len:].tolist()
            response = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            patent_df.at[idx, "summary"] = response
            print(response)
            if i+1 % SAVE_FREQ == 0:
                patent_df.to_json(output_file, orient='records', lines=True)
            i += 1
        except Exception as e:
            print(f"Error generating summary: {e}")
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            continue
    patent_df.to_json(output_file, orient='records', lines=True)