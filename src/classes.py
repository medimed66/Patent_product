import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import copy

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Better initialization for SiLU activation
        self._init_weights()
        
    def _init_weights(self):
        # He initialization for SiLU (similar to ReLU)
        nn.init.kaiming_uniform_(self.linear.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        
        # LayerNorm initialization
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        
    def forward(self, x):
        residual = x
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Learnable query vectors - better initialization
        self.queries = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        
        # Projection matrices for keys, values, and output
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = self.head_dim ** -0.5
        
        # Initialize all parameters
        self._init_weights()
        
    def _init_weights(self):
        # Initialize queries with small random values
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        
        # Ensure query diversity across heads using orthogonal initialization
        if self.num_heads > 1:
            # Create orthogonal vectors for each head
            with torch.no_grad():
                # Reshape queries for orthogonal initialization
                queries_2d = self.queries.squeeze(0)  # [num_heads, head_dim]
                
                # Apply orthogonal initialization if possible
                if self.num_heads <= self.head_dim:
                    # Full orthogonal initialization
                    nn.init.orthogonal_(queries_2d)
                else:
                    # Partial orthogonal initialization for more heads than dimensions
                    for i in range(0, self.num_heads, self.head_dim):
                        end_idx = min(i + self.head_dim, self.num_heads)
                        nn.init.orthogonal_(queries_2d[i:end_idx])
                
                # Scale down to maintain reasonable magnitude
                queries_2d *= 0.1
                self.queries.data = queries_2d.unsqueeze(0)
        
        # Initialize projection matrices with Xavier/Glorot for attention
        for module in [self.key_projection, self.value_projection, self.output_projection]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
        
        # LayerNorm initialization
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        
    def forward(self, token_embeddings, attention_mask):
        batch_size, seq_len, _ = token_embeddings.shape
        
        # Apply layer norm to input
        token_embeddings = self.norm(token_embeddings)
        
        # Project keys and values
        keys = self.key_projection(token_embeddings)  # [batch, seq_len, hidden_size]
        values = self.value_projection(token_embeddings)  # [batch, seq_len, hidden_size]
        
        # Reshape keys and values for multi-head attention
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch, num_heads, seq_len, head_dim]
        
        # Expand queries for batch
        queries = self.queries.expand(batch_size, -1, -1)  # [batch, num_heads, head_dim]
        queries = queries.unsqueeze(2)  # [batch, num_heads, 1, head_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            # Expand mask for multi-head attention
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        pooled = torch.matmul(attention_weights, values)  # [batch, num_heads, 1, head_dim]
        pooled = pooled.squeeze(2)  # [batch, num_heads, head_dim]
        
        # Concatenate heads and apply output projection
        pooled = pooled.reshape(batch_size, self.hidden_size)  # [batch, hidden_size]
        pooled = self.output_projection(pooled)
        
        # Return pooled output and attention weights (averaged across heads for interpretability)
        avg_attention_weights = attention_weights.mean(dim=1).squeeze(1)  # [batch, seq_len]
        
        return pooled, avg_attention_weights

class Encoder(nn.Module):
    def __init__(self, input_dim, device, embedding_dim, num_heads=32, dropout_rate=0.1):
        super(Encoder, self).__init__()
        
        # Base encoder
        self.device = device
        
        # Multi-head attention pooling
        self.attention_pooler = MultiHeadAttentionPooling(
            input_dim, num_heads, dropout_rate
        )
        
        # Enhanced projection layers with residuals
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks for deeper processing
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(embedding_dim, dropout_rate) for _ in range(6)
        ])
        
        # Final projection layer
        self.final_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate methods for each layer type"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use He initialization for layers followed by SiLU activation
                if hasattr(module, '_is_output_layer'):
                    # For final output layers, use Xavier
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                else:
                    # For intermediate layers with SiLU, use He initialization
                    nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Mark the final layer for different initialization
        final_linear_layers = [m for m in self.final_projection.modules() if isinstance(m, nn.Linear)]
        if final_linear_layers:
            final_linear_layers[-1]._is_output_layer = True
            # Re-initialize the final layer with Xavier
            nn.init.xavier_uniform_(final_linear_layers[-1].weight, gain=1.0)
    
    def forward(self, token_embeddings, attention_mask, return_attention=False):
        # Multi-head attention pooling
        pooled_embeddings, attention_weights = self.attention_pooler(
            token_embeddings, attention_mask
        )
        
        # Initial projection
        embeddings = self.input_projection(pooled_embeddings)
        
        # Apply residual blocks
        for residual_block in self.residual_blocks:
            embeddings = residual_block(embeddings)
        
        embeddings = self.final_projection(embeddings)
        
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        if return_attention:
            return embeddings, attention_weights
        return embeddings
    
    def get_attention_weights(self, input_ids, attention_mask):
        """Get attention weights for visualization"""
        with torch.no_grad():
            _, attention_weights = self.forward(
                input_ids, attention_mask, return_attention=True
            )
        return attention_weights

class PatentProductDataset(Dataset):
    def __init__(self, dataframe, model_name, patent_device, product_device, max_length, 
                is_training=True, batch_size=64):
        self.data = dataframe
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.is_training = is_training
        self.embedding_dim = encoder.config.hidden_size
        
        # Get unique texts
        product_texts = dataframe[['Product', 'product_text']].drop_duplicates(subset=['Product'])
        patent_texts = dataframe[['patent_id', 'patent_text']].drop_duplicates(subset=['patent_id'])
        
        # Batch encode products
        self.product_encodings = self._batch_encode_texts(
            encoder, tokenizer, product_texts, 'Product', 'product_text', 
            product_device, batch_size
        )
        
        # Batch encode patents
        self.patent_encodings = self._batch_encode_texts(
            encoder, tokenizer, patent_texts, 'patent_id', 'patent_text', 
            patent_device, batch_size
        )
        
        del encoder  # Free up memory
    
    def _batch_encode_texts(self, encoder, tokenizer, texts_df, id_col, text_col, 
                            device, batch_size):
        """Encode texts in batches to improve memory efficiency"""
        encodings = {}
        text_list = texts_df[text_col].tolist()
        id_list = texts_df[id_col].tolist()
        
        # Move encoder to target device
        encoder = encoder.to(device)
        encoder.eval()  # Set to evaluation mode
        
        with torch.no_grad():  # Disable gradients for encoding
            for i in range(0, len(text_list), batch_size):
                batch_texts = text_list[i:i + batch_size]
                batch_ids = id_list[i:i + batch_size]
                
                # Tokenize batch
                batch_tokens = tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(device)
                
                # Encode batch
                batch_embeddings = encoder(
                    batch_tokens['input_ids'],
                    batch_tokens['attention_mask']
                ).last_hidden_state
                
                # Store encodings
                for j, text_id in enumerate(batch_ids):
                    encodings[text_id] = {
                        'embedding': batch_embeddings[j].clone().detach(),
                        'attention_mask': batch_tokens['attention_mask'][j].clone().detach(),
                    }
                
                # Clear batch tensors to free memory
                del batch_tokens, batch_embeddings
                torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        return encodings
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        product_id = self.data.iloc[idx]['Product']
        patent_id = self.data.iloc[idx]['patent_id']
        
        item = {
            'product_id': product_id,
            'patent_id': patent_id,
            'product_embedding': self.product_encodings[product_id]['embedding'],
            'patent_embedding': self.patent_encodings[patent_id]['embedding'],
            'product_attention_mask': self.product_encodings[product_id]['attention_mask'],
            'patent_attention_mask': self.patent_encodings[patent_id]['attention_mask'],
        }
        return item

def collate_fn(batch):
    seen_product_ids = set()
    seen_patent_ids = set()

    unique_product_embedding = []
    unique_product_attention_mask = []
    unique_product_ids = []

    unique_patent_embedding = []
    unique_patent_attention_mask = []
    unique_patent_ids = []

    for item in batch:
        # Handle unique products
        if item['product_id'] not in seen_product_ids:
            seen_product_ids.add(item['product_id'])
            unique_product_embedding.append(item['product_embedding'])
            unique_product_attention_mask.append(item['product_attention_mask'])
            unique_product_ids.append(item['product_id'])

        # Handle unique patents
        if item['patent_id'] not in seen_patent_ids:
            seen_patent_ids.add(item['patent_id'])
            unique_patent_embedding.append(item['patent_embedding'])
            unique_patent_attention_mask.append(item['patent_attention_mask'])
            unique_patent_ids.append(item['patent_id'])

    return {
        'product_embedding': torch.stack(unique_product_embedding),
        'product_attention_mask': torch.stack(unique_product_attention_mask),
        'patent_embedding': torch.stack(unique_patent_embedding),
        'patent_attention_mask': torch.stack(unique_patent_attention_mask),
        'product_id': unique_product_ids,
        'patent_id': unique_patent_ids,
    }

class ModelCheckpoint:
    """
    Saves the model weights with the best validation loss and restores them at the end of training.
    """
    def __init__(self):
        self.best_loss = float('inf')
        self.best_prod_weights = None
        self.best_pat_weights = None
    
    def __call__(self, val_loss, prod_model, pat_model, end=False):
        if val_loss < self.best_loss + 0.01:
            self.best_loss = val_loss
            self.best_prod_weights = copy.deepcopy(prod_model.state_dict())
            self.best_pat_weights = copy.deepcopy(pat_model.state_dict())
        
        if end:
            if self.best_prod_weights is not None:
                prod_model.load_state_dict(self.best_prod_weights)
            if self.best_pat_weights is not None:
                pat_model.load_state_dict(self.best_pat_weights)

class Adversary(nn.Module):
    def __init__(self, adv_type, data, device, max_length, input_dim, input_to_log=None):
        super(Adversary, self).__init__()
        self.associated_elems = defaultdict(set)
        self.input_to_log = input_to_log
        self.policy_history = []
        self.device = device
        self.embedding_dim = input_dim
        
        if adv_type == 'product':
            elems = data['Product'].unique().tolist()
            for _, row in data.iterrows():
                self.associated_elems[row['patent_id']].add(row['Product'])
        elif adv_type == 'patent':
            elems = data['patent_id'].unique().tolist()
            for _, row in data.iterrows():
                self.associated_elems[row['Product']].add(row['patent_id'])
        
        self.elem_to_idx = {elem: idx for idx, elem in enumerate(elems)}
        self.idx_to_elem = {v: k for k, v in self.elem_to_idx.items()}
        
        self.max_length = max_length
        output_dim = len(elems)
        hidden_dim = round((input_dim + output_dim)/2)
        
        self.pipeline = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )
        
        # Initialize hidden layers with small random weights
        for layer in self.pipeline[:-1]:  # All except final layer
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)  # Small random weights
                nn.init.zeros_(layer.bias)  # Zero bias
        
        # Initialize FINAL layer to zeros (for uniform policy)
        final_layer = self.pipeline[-1]
        nn.init.zeros_(final_layer.weight)  # Zero weights
        nn.init.zeros_(final_layer.bias)    # Zero bias
    
    def forward(self, input_ids, batch_elem_ids, input_embeds, num_negatives):
        """Process batch of patents instead of single patent"""
        neg_elems = []
        log_probs = []
        expanded_input = []
        for input, embed in zip(input_ids, input_embeds):
            # Get related products for this patent
            related_elems = self.associated_elems[input] | set(neg_elems) | batch_elem_ids
            related_elem_indices = [self.elem_to_idx[elem] for elem in related_elems]
            logit = self.pipeline(embed.to(self.device))
            logit = torch.clamp(logit, min=-25, max=25)
            policy = F.softmax(logit, dim=0)
            mask = torch.ones_like(policy)
            mask[related_elem_indices] = 0
            masked_policy = policy * mask
            policy_sum = torch.sum(masked_policy)
            policy = masked_policy / policy_sum
            
            if input == self.input_to_log:
                self.policy_history.append(policy.clone().detach().cpu())
            available_negatives = torch.count_nonzero(policy).item()
            num_to_sample = min(num_negatives, available_negatives)
            if num_to_sample > 0:
                selected_idx = torch.multinomial(policy, num_to_sample, replacement=False)
                expanded_input.append(embed.repeat(num_to_sample, 1))
                log_prob = torch.log(policy[selected_idx])
                log_probs.append(log_prob)
                added_elems = [self.idx_to_elem[idx.item()] for idx in selected_idx]
                neg_elems.extend(added_elems)
        
        return neg_elems, log_probs, expanded_input

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, data, temp):
        super(SupervisedContrastiveLoss, self).__init__()
        pairs = set(zip(data['Product'], data['patent_id']))
        self.temp = temp
        self.positive_pairs = defaultdict(lambda: defaultdict(bool))
        
        # Populate with True for known positive pairs
        for product_id, patent_id in pairs:
            self.positive_pairs[product_id][patent_id] = True
    
    def forward(self, product_embeddings, patent_embeddings, product_ids, patent_ids):
        # Compute similarity matrix (dot product scaled by temperature)
        sims = torch.mm(product_embeddings, patent_embeddings.t()) / self.temp
        
        # Construct positive mask: [num_products x num_patents]
        positive_mask = torch.zeros_like(sims, dtype=torch.bool)
        for i, prod_id in enumerate(product_ids):
            associated_patents = self.positive_pairs[prod_id]
            for j, pat_id in enumerate(patent_ids):
                positive_mask[i, j] = associated_patents[pat_id]
        
        num_pos_per_prod = positive_mask.sum(dim=1)
        num_pos_per_pat = positive_mask.sum(dim=0)
        positive_i = num_pos_per_prod > 0
        positive_j = num_pos_per_pat > 0
        pos_i_sims = sims[positive_i, :]
        pos_j_sims = sims[:, positive_j]
        
        prod_denominator = torch.logsumexp(pos_i_sims, dim=1)
        pat_denominator  = torch.logsumexp(pos_j_sims, dim=0)
        
        prod_numerator = (pos_i_sims * positive_mask[positive_i, :]).sum(dim=1)/ num_pos_per_prod[positive_i]
        pat_numerator = (pos_j_sims * positive_mask[:, positive_j]).sum(dim=0)/ num_pos_per_pat[positive_j]
        
        prod_loss = (prod_denominator - prod_numerator).mean()
        pat_loss = (pat_denominator - pat_numerator).mean()
        loss =  (prod_loss + pat_loss)/2
        return loss