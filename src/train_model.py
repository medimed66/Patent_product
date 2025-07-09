import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from helpers import split_data, get_embeddings_frozen, evaluate_performance, plot_metrics, save_embeddings, plot_policy_heatmap
from classes import PatentProductDataset, Encoder, Adversary, SupervisedContrastiveLoss, ModelCheckpoint, collate_fn
from zclip import ZClip
import os

LEARNING_RATE = 1e-5
ADV_LEARNING_RATE = 1e-5
EMBEDDING_DIM = 4096
TEMPERATURE = 0.02
NUM_PROD_HEADS = 32
NUM_PAT_HEADS = 32
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 64
NUM_EPOCHS = 3
MAX_LENGTH = 512
SEED=42
model_name = 'mpi-inno-comp/paecter'

# Set random seed for reproducibility
def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

def train_epoch(product_encoder, patent_encoder, dataloader,
                prod_adversary, pat_adversary, adv_optimizer, encoder_optimizer,
                encoder_scheduler, criterion, product_device, patent_device, zclip_prod, zclip_pat, epoch):
    product_encoder.train() 
    patent_encoder.train()
    prod_adversary.train()
    pat_adversary.train()
    running_loss = 0.0
    running_adversary_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_losses = []
    adv_reward = []
    dataset = dataloader.dataset
    num_negatives = 5*epoch
    for batch_idx, batch in enumerate(progress_bar):
        encoder_optimizer.zero_grad()
        adv_optimizer.zero_grad()
        product_ids = batch['product_id']
        patent_ids = batch['patent_id']
        product_inputs = batch['product_embedding'].to(product_device)
        product_attention_mask = batch['product_attention_mask'].to(product_device)
        patent_inputs = batch['patent_embedding'].to(patent_device)
        patent_attention_mask = batch['patent_attention_mask'].to(patent_device)
        # Get embeddings
        product_embeddings = product_encoder(product_inputs, product_attention_mask)
        patent_embeddings = patent_encoder(patent_inputs, patent_attention_mask)
        
        if num_negatives > 0:
            neg_products, prod_log_probs, prod_comp = prod_adversary(patent_ids, set(product_ids), patent_embeddings.detach(), num_negatives)
            neg_prod_init_embed = torch.stack([dataset.product_encodings[elem]['embedding'] for elem in neg_products], dim=0)
            neg_attention_mask = torch.stack([dataset.product_encodings[elem]['attention_mask'] for elem in neg_products], dim=0)
            neg_prod_embeds = product_encoder(neg_prod_init_embed, neg_attention_mask).to('cpu')
            
            neg_patents, pat_log_probs, pat_comp = pat_adversary(product_ids, set(patent_ids), product_embeddings.detach(), num_negatives)
            neg_pat_init_embed = torch.stack([dataset.patent_encodings[elem]['embedding'] for elem in neg_patents], dim=0)
            neg_pat_attention_mask = torch.stack([dataset.patent_encodings[elem]['attention_mask'] for elem in neg_patents], dim=0)
            neg_pat_embeds = patent_encoder(neg_pat_init_embed, neg_pat_attention_mask).to('cpu')
            
            with torch.no_grad():
                prod_comp = torch.cat(prod_comp, dim=0).to('cpu')
                pat_comp = torch.cat(pat_comp, dim=0).to('cpu')
                prod_sim = (neg_prod_embeds * prod_comp).sum(dim=1) / TEMPERATURE
                pat_sim = (neg_pat_embeds * pat_comp).sum(dim=1) / TEMPERATURE
                
                # Center rewards around mean for stability
                prod_reward = prod_sim - prod_sim.mean()
                pat_reward = pat_sim - pat_sim.mean()
                
                adv_reward.append((prod_sim.mean() + pat_sim.mean()).item()/2)
            
            prod_loss = -(prod_reward * torch.cat(prod_log_probs, dim=0).to('cpu')).mean()
            pat_loss = -(pat_reward * torch.cat(pat_log_probs, dim=0).to('cpu')).mean()
            adv_loss = (prod_loss + pat_loss) / 2
            adv_loss.backward()
            adv_optimizer.step()
            running_adversary_loss += adv_loss.item()
            
            all_product_embeddings = torch.cat([product_embeddings.to('cpu'), neg_prod_embeds.to('cpu')], dim=0)
            product_ids.extend(neg_products)
            all_patent_embeddings = torch.cat([patent_embeddings.to('cpu'), neg_pat_embeds.to('cpu')], dim=0)
            patent_ids.extend(neg_patents)
        else:
            all_product_embeddings = product_embeddings.to('cpu')
            all_patent_embeddings = patent_embeddings.to('cpu')
            adv_reward.append(0.0)
        
        # Calculate main loss
        loss = criterion(
            all_product_embeddings,
            all_patent_embeddings,
            product_ids,
            patent_ids,
        )
        
        # Backward pass for encoders
        loss.backward()
        zclip_prod.step(product_encoder)
        zclip_pat.step(patent_encoder)
        encoder_optimizer.step()
        encoder_scheduler.step()
        train_losses.append(loss.item())
        running_loss += loss.item()
        postfix = {"enc_loss": running_loss/(batch_idx+1)}
        postfix["adv_loss"] = running_adversary_loss/(batch_idx+1)
        progress_bar.set_postfix(postfix)
    return train_losses, adv_reward

def validate_epoch(product_encoder, patent_encoder, dataloader, criterion, 
                    product_device, patent_device):
    """Validate for one epoch"""
    product_encoder.eval()
    patent_encoder.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            product_ids = batch['product_id']
            patent_ids = batch['patent_id']
            product_inputs = batch['product_embedding'].to(product_device)
            product_attention_mask = batch['product_attention_mask'].to(product_device)
            patent_inputs = batch['patent_embedding'].to(patent_device)
            patent_attention_mask = batch['patent_attention_mask'].to(patent_device)
            
            # Get embeddings
            product_embeddings = product_encoder(product_inputs, product_attention_mask)
            patent_embeddings = patent_encoder(patent_inputs, patent_attention_mask)
            
            # Calculate loss
            loss = criterion(
                product_embeddings.to('cpu'),
                patent_embeddings.to('cpu'),
                product_ids,
                patent_ids,
            )
            running_loss += loss.item()
            num_batches += 1
    return running_loss / num_batches

path = "../Data/pairs.json"
if not os.path.exists(path):
    # Load the data
    df = pd.read_csv("../Data/honeywell_patents_products.tsv", sep="\t")
    df.drop(columns=['Table Number'], inplace=True)
    df['Associated Patent(s)'] = df.apply(lambda x: [p.strip() for p in x['Associated Patent(s)'].replace(':',';').replace(',', '').split(';')], axis=1)
    df = df.explode(column='Associated Patent(s)',ignore_index=True)
    df.rename(columns={'Associated Patent(s)': 'patent_id', 'Product Name': 'Product'}, inplace=True)
    df = df[df['patent_id'].str.match(r'^\d')]
    df['patent_id'] = df['patent_id'].astype(str)
    df.drop_duplicates(subset=['patent_id', 'Product'], inplace=True)
    
    # Load patent text
    patent_df = pd.read_json("../Data/summarized_patents.json", lines=True)
    patent_df['patent_id'] = patent_df['patent_id'].astype(str)
    patent_df = patent_df[patent_df['patent_id'].str.match(r'^\d')]
    patent_df['patent_text'] = patent_df['title'] + "\n" + patent_df['summary']
    patent_df.drop(columns=['title', 'kind_code', 'family_id', 'summary'], inplace=True)
    
    # load product text
    product_df = pd.read_json("../Data/products.json", orient='index')
    product_df.rename(columns={'Description': 'product_text'}, inplace=True)
    product_df.drop(columns=['Bulletpoints'], inplace=True)
    
    df = df.merge(patent_df, on='patent_id', how='left')
    df = df.merge(product_df, left_on='Product', right_index=True, how='left')
    df = df[df['product_text'] != ""]
    df = df[df['patent_text'] != ""]
    df.to_json(path, orient='records', lines=True)

df = pd.read_json(path, orient='records', lines=True, dtype=str)

# Split: 70% train, 15% validation, 15% test
print("Splitting train set...")
train_df, temp_df = split_data(df, 0.3, 2, verbose=True)
print("\nSplitting validation and test sets...")
test_df, val_df = split_data(temp_df, 0.5, 0, verbose=True)

#extract most frequent products in train set
most_frequent_product = train_df['Product'].value_counts().nlargest(1).index.tolist()[0]
#extract most frequent patents in train set
most_frequent_patent = train_df['patent_id'].value_counts().nlargest(1).index.tolist()[0]

# Setup devices
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    product_device = torch.device("cuda:0")
    patent_device = torch.device("cuda:1")
else:
    product_device = patent_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create datasets
train_dataset = PatentProductDataset(train_df, model_name, patent_device, product_device, MAX_LENGTH, is_training=True)
val_dataset = PatentProductDataset(val_df, model_name, patent_device, product_device, MAX_LENGTH, is_training=False)
test_dataset = PatentProductDataset(test_df, model_name, patent_device, product_device, MAX_LENGTH, is_training=False)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

# Initialize models
product_encoder = Encoder(train_dataset.embedding_dim ,product_device, EMBEDDING_DIM, NUM_PROD_HEADS).to(product_device)
patent_encoder = Encoder(train_dataset.embedding_dim ,patent_device, EMBEDDING_DIM, NUM_PAT_HEADS).to(patent_device)
prod_adversary = Adversary('product', train_df, patent_device, MAX_LENGTH, EMBEDDING_DIM, most_frequent_patent).to(patent_device)
pat_adversary = Adversary('patent', train_df, product_device, MAX_LENGTH, EMBEDDING_DIM, most_frequent_product).to(product_device)
criterion = SupervisedContrastiveLoss(df, TEMPERATURE).to('cpu')

adv_optimizer = optim.AdamW(
    list(prod_adversary.parameters()) + list(pat_adversary.parameters()),
    lr=ADV_LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

encoder_optimizer = optim.AdamW(
    list(product_encoder.parameters()) + list(patent_encoder.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
zclip_prod = ZClip(alpha=0.97, z_thresh=2.5)
zclip_pat = ZClip(alpha=0.97, z_thresh=2.5)

total_steps = len(train_dataloader) * NUM_EPOCHS
encoder_scheduler = get_cosine_schedule_with_warmup(
    encoder_optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

model_checkpoint = ModelCheckpoint()

# Training loop with validation
train_losses = []
val_losses = []
adv_rewards = []
print("Starting training with early stopping...")

for epoch in range(NUM_EPOCHS):
    # Training
    train_loss, adv_reward = train_epoch(product_encoder, patent_encoder, train_dataloader,
                                        prod_adversary, pat_adversary, adv_optimizer,
                                        encoder_optimizer, encoder_scheduler, criterion,
                                        product_device, patent_device,
                                        zclip_prod, zclip_pat, epoch)
    
    # Validation
    val_loss = validate_epoch(product_encoder, patent_encoder, val_dataloader, 
                            criterion, product_device, patent_device)
    
    val_loss_extend = [val_loss]*len(train_loss)
    train_losses.extend(train_loss)
    val_losses.extend(val_loss_extend)
    adv_rewards.extend(adv_reward)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {np.mean(train_loss):.4f}, Adv reward: {np.mean(adv_reward):.4f}, Val Loss: {val_loss:.4f}\n")
    
    if epoch == NUM_EPOCHS - 1 :
        model_checkpoint(val_loss, product_encoder, patent_encoder, end=True)
    else:
        model_checkpoint(val_loss, product_encoder, patent_encoder)

with torch.no_grad():
    test_product_embs, test_patent_embs = get_embeddings_frozen(product_encoder, patent_encoder, test_dataloader, product_device, patent_device)
    train_product_embs, train_patent_embs = get_embeddings_frozen(product_encoder, patent_encoder, train_dataloader, product_device, patent_device)
    val_product_embs, val_patent_embs = get_embeddings_frozen(product_encoder, patent_encoder, val_dataloader, product_device, patent_device)

plot_policy_heatmap(prod_adversary.policy_history, pat_adversary.policy_history)
test_metrics = evaluate_performance(test_product_embs, test_patent_embs, test_df, EMBEDDING_DIM)
val_metrics = evaluate_performance(val_product_embs, val_patent_embs, val_df, EMBEDDING_DIM)
train_metrics = evaluate_performance(train_product_embs, train_patent_embs, train_df, EMBEDDING_DIM)
plot_metrics(test_metrics, train_losses, val_losses, adv_rewards, 'test')
plot_metrics(val_metrics, train_losses, val_losses, adv_rewards, 'val')
plot_metrics(train_metrics, train_losses, val_losses, adv_rewards, 'train')

save_embeddings(train_product_embs, "train_product")
save_embeddings(train_patent_embs, "train_patent")
save_embeddings(test_product_embs, "test_product")
save_embeddings(test_patent_embs, "test_patent")
save_embeddings(val_product_embs, "val_product")
save_embeddings(val_patent_embs, "val_patent")

# Save model
torch.save({
    'product_encoder': product_encoder.state_dict(),
    'patent_encoder': patent_encoder.state_dict(),
    'prod_adversary': prod_adversary.state_dict(),
    'pat_adversary': pat_adversary.state_dict(),
}, "models.pt")