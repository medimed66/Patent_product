import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt
import networkx as nx

def split_data(edges, fraction=0.2, n=2, verbose=True):
    """
    Split a bipartite graph into train and test sets using a two-pass approach.
    Compares bridge deletion strategy vs. small components first strategy.
    
    Parameters:
    G: NetworkX bipartite graph
    fraction: desired fraction of nodes/edges for test set (default 0.2)
    n: number of largest components to process (default 2)
    verbose: whether to print detailed output (default True)
    Returns:
    train_graph, test_graph: tuple of NetworkX graphs
    """
    def calculate_distribution_score(component, ammounts, fraction):
        """Calculate how well the test set matches the target distribution."""
        added_nodes = component.nodes(data='bipartite')
        added_patents = sum(1 for _, node_type in added_nodes if node_type == 'patent')
        added_products = component.number_of_nodes() - added_patents
        added_edges = component.number_of_edges()
        
        target_test_patents = fraction * ammounts['tot_patent']
        target_test_products = fraction * ammounts['tot_product']
        target_test_edges = fraction * ammounts['tot_edges']
        
        # Calculate relative squared errors
        patent_error = ((added_patents + ammounts['test_patent'] - target_test_patents) / target_test_patents)**2
        product_error = ((added_products + ammounts['test_product'] - target_test_products) / target_test_products)**2
        edge_error = ((added_edges + ammounts['test_edges'] - target_test_edges) / target_test_edges)**2
        
        return patent_error + product_error + edge_error
    
    def calculate_distribution_score_small(component, ammounts, fraction, update=True):
        """Calculate how well the test set matches the target distribution."""
        added_nodes = component.nodes(data='bipartite')
        added_patents = sum(1 for _, node_type in added_nodes if node_type == 'patent')
        added_products = component.number_of_nodes() - added_patents
        added_edges = component.number_of_edges()
        
        if update:
            ammounts['tot_patent'] += added_patents
            ammounts['tot_product'] += added_products
            ammounts['tot_edges'] += added_edges
        
        target_test_patents = fraction * ammounts['tot_patent']
        target_test_products = fraction * ammounts['tot_product']
        target_test_edges = fraction * ammounts['tot_edges']
        
        # Calculate relative squared errors
        test_patent_error = ((added_patents + ammounts['test_patent'] - target_test_patents) / target_test_patents)**2
        test_product_error = ((added_products + ammounts['test_product'] - target_test_products) / target_test_products)**2
        test_edge_error = ((added_edges + ammounts['test_edges'] - target_test_edges) / target_test_edges)**2
        test_error = test_patent_error + test_product_error + test_edge_error
        
        train_patent_error = ((ammounts['test_patent'] - target_test_patents) / target_test_patents)**2
        train_product_error = ((ammounts['test_product'] - target_test_products) / target_test_products)**2
        train_edge_error = ((ammounts['test_edges'] - target_test_edges) / target_test_edges)**2
        train_error = train_patent_error + train_product_error + train_edge_error
        
        if test_error < train_error:
            ammounts['test_patent'] += added_patents
            ammounts['test_product'] += added_products
            ammounts['test_edges'] += added_edges
        
        return test_error, train_error, ammounts
    
    def split_strategy_bridge_first(G, fraction, n):
        """Original strategy: process largest components with bridge deletion first."""
        # Initialize train and test graphs
        train_graph = nx.Graph()
        test_graph = nx.Graph()
        ammounts = {'tot_patent': 0, 'tot_product': 0, 'tot_edges': 0, 'test_patent': 0, 'test_product': 0, 'test_edges': 0}
        
        # Get all connected components sorted by number of edges (descending)
        components = list(nx.connected_components(G))
        components_with_edges = [(comp, G.subgraph(comp).number_of_edges()) for comp in components]
        components_with_edges.sort(key=lambda x: x[1], reverse=True)
        
        # Process top n largest components
        for comp in components_with_edges[:n]:
            component_subgraph = G.subgraph(comp[0]).copy()
            component_patents = sum([1 for n in comp[0] if G.nodes[n].get('bipartite') == 'patent'])
            component_products = component_subgraph.number_of_nodes() - component_patents
            ammounts['tot_patent'] += component_patents
            ammounts['tot_product'] += component_products
            ammounts['tot_edges'] += component_subgraph.number_of_edges()-1
            
            # Find bridge edges
            bridges = list(nx.bridges(component_subgraph))
            
            best_bridge = None
            best_score = float('inf')
            best_smaller_subgraph = None
            best_bigger_subgraph = None
            
            # Evaluate each bridge
            for bridge in bridges:
                component_subgraph.remove_edge(*bridge)
                subcomponents = list(nx.connected_components(component_subgraph))
                
                # Identify smaller subcomponent
                sub1, sub2 = subcomponents
                smaller_sub, bigger_sub = (sub1, sub2) if len(sub1) < len(sub2) else (sub2, sub1)
                if len(smaller_sub) == 1:
                    component_subgraph.add_edge(*bridge)  # Restore edge
                    continue
                
                smaller_subgraph = component_subgraph.subgraph(smaller_sub)
                
                # Calculate score using the dynamic function
                score = calculate_distribution_score(smaller_subgraph, ammounts, fraction)
                
                if score < best_score:
                    best_score = score
                    best_bridge = bridge
                    best_smaller_subgraph = smaller_subgraph
                    best_bigger_subgraph = component_subgraph.subgraph(bigger_sub)
                component_subgraph.add_edge(*bridge)  # Restore edge
            
            # Apply best split if found
            if best_bridge is not None:
                test_graph.add_nodes_from(best_smaller_subgraph.nodes(data=True))
                test_graph.add_edges_from(best_smaller_subgraph.edges())
                train_graph.add_nodes_from(best_bigger_subgraph.nodes(data=True))
                train_graph.add_edges_from(best_bigger_subgraph.edges())
                ammounts['test_patent'] += len([n for n in best_smaller_subgraph.nodes() if G.nodes[n].get('bipartite') == 'patent'])
                ammounts['test_product'] += len([n for n in best_smaller_subgraph.nodes() if G.nodes[n].get('bipartite') == 'product'])
                ammounts['test_edges'] += best_smaller_subgraph.number_of_edges()
            else:
                # No suitable bridge found, add to train
                ammounts['tot_edges'] += 1
                train_graph.add_nodes_from(component_subgraph)
                train_graph.add_edges_from(component_subgraph.edges())
        
        # Process each remaining component
        for comp in components_with_edges[n:]:
            comp_subgraph = G.subgraph(comp[0])
            test_score, train_score, ammounts = calculate_distribution_score_small(comp_subgraph, ammounts, fraction)
            
            # Assign component to the set that minimizes the score
            if test_score < train_score:
                final_score = test_score
                test_graph.add_nodes_from(comp_subgraph.nodes(data=True))
                test_graph.add_edges_from(comp_subgraph.edges())
            else:
                final_score = train_score
                train_graph.add_nodes_from(comp_subgraph.nodes(data=True))
                train_graph.add_edges_from(comp_subgraph.edges())
        
        return train_graph, test_graph, final_score

    def split_strategy_small_first(G, fraction, n):
        """Original strategy: process largest components with bridge deletion first."""
        # Initialize train and test graphs
        train_graph = nx.Graph()
        test_graph = nx.Graph()
        tot_patents = len([n for n in G.nodes() if G.nodes[n].get('bipartite') == 'patent'])
        tot_products = len([n for n in G.nodes() if G.nodes[n].get('bipartite') == 'product'])
        tot_edges = G.number_of_edges()-n
        ammounts = {'tot_patent': tot_patents, 'tot_product': tot_products, 'tot_edges': tot_edges, 'test_patent': 0, 'test_product': 0, 'test_edges': 0}
        best_init_score = float('inf')
        # Get all connected components sorted by number of edges (descending)
        components = list(nx.connected_components(G))
        components_with_edges = [(comp, G.subgraph(comp).number_of_edges()) for comp in components]
        components_with_edges.sort(key=lambda x: x[1], reverse=False)
        
        for comp in reversed(components_with_edges[:len(components_with_edges)-n]):
            comp_subgraph = G.subgraph(comp[0])
            test_score, train_score, ammounts = calculate_distribution_score_small(comp_subgraph, ammounts, fraction, update=False)
            
            # Assign component to the set that minimizes the score
            if test_score < train_score:
                test_graph.add_nodes_from(comp_subgraph.nodes(data=True))
                test_graph.add_edges_from(comp_subgraph.edges())
                best_init_score = test_score
            else:
                train_graph.add_nodes_from(comp_subgraph.nodes(data=True))
                train_graph.add_edges_from(comp_subgraph.edges())
                best_init_score = train_score
            
        # Process top n largest components
        for comp in components_with_edges[-n:]:
            component_subgraph = G.subgraph(comp[0]).copy()
            
            # Find bridge edges
            bridges = list(nx.bridges(component_subgraph))
            best_bridge = None
            best_smaller_subgraph = None
            best_bigger_subgraph = None
            best_score = float('inf')
            
            # Evaluate each bridge
            for bridge in bridges:
                component_subgraph.remove_edge(*bridge)
                subcomponents = list(nx.connected_components(component_subgraph))
                
                # Identify smaller subcomponent
                sub1, sub2 = subcomponents
                smaller_sub, bigger_sub = (sub1, sub2) if len(sub1) < len(sub2) else (sub2, sub1)
                if len(smaller_sub) == 1:
                    component_subgraph.add_edge(*bridge)  # Restore edge
                    continue
                
                smaller_subgraph = component_subgraph.subgraph(smaller_sub)
                
                # Calculate score using the dynamic function
                score = calculate_distribution_score(smaller_subgraph, ammounts, fraction)
                
                if score < best_score:
                    best_score = score
                    best_bridge = bridge
                    best_smaller_subgraph = smaller_subgraph
                    best_bigger_subgraph = component_subgraph.subgraph(bigger_sub)
                component_subgraph.add_edge(*bridge)  # Restore edge
            
            # Apply best split if found
            if best_bridge is not None:
                test_graph.add_nodes_from(best_smaller_subgraph.nodes(data=True))
                test_graph.add_edges_from(best_smaller_subgraph.edges())
                train_graph.add_nodes_from(best_bigger_subgraph.nodes(data=True))
                train_graph.add_edges_from(best_bigger_subgraph.edges())
                ammounts['test_patent'] += len([n for n in best_smaller_subgraph.nodes() if G.nodes[n].get('bipartite') == 'patent'])
                ammounts['test_product'] += len([n for n in best_smaller_subgraph.nodes() if G.nodes[n].get('bipartite') == 'product'])
                ammounts['test_edges'] += best_smaller_subgraph.number_of_edges()
            else:
                # No suitable bridge found, add to train
                ammounts['tot_edges'] += 1
                train_graph.add_nodes_from(component_subgraph.nodes(data=True))
                train_graph.add_edges_from(component_subgraph.edges())
        
        if best_score == float('inf'):
            best_score = best_init_score
        
        return train_graph, test_graph, best_score
    
    def print_splitting_state(G, train_graph, test_graph, score, strategy_used):
        # Calculate and print final statistics
        orig_nodes = G.number_of_nodes()
        orig_edges = G.number_of_edges()
        orig_products = len([n for n in G.nodes() if G.nodes[n].get('bipartite') == 'product'])
        orig_patents = len([n for n in G.nodes() if G.nodes[n].get('bipartite') == 'patent'])
        
        train_nodes = train_graph.number_of_nodes()
        train_edges = train_graph.number_of_edges()
        train_products = len([n for n in train_graph.nodes() if G.nodes[n].get('bipartite') == 'product'])
        train_patents = len([n for n in train_graph.nodes() if G.nodes[n].get('bipartite') == 'patent'])
        
        test_nodes = test_graph.number_of_nodes()
        test_edges = test_graph.number_of_edges()
        test_products = len([n for n in test_graph.nodes() if G.nodes[n].get('bipartite') == 'product'])
        test_patents = len([n for n in test_graph.nodes() if G.nodes[n].get('bipartite') == 'patent'])
        
        # Print results
        print(f"\nFinal results using {strategy_used} strategy:")
        print("Original graph:")
        print(f"  Nodes: {orig_nodes}, Edges: {orig_edges}")
        print(f"  Products: {orig_products}")
        print(f"  Patents: {orig_patents}")
        
        print("\nTrain graph:")
        print(f"  Nodes: {train_nodes} ({train_nodes/orig_nodes*100:.1f}%)")
        print(f"  Edges: {train_edges} ({train_edges/orig_edges*100:.1f}%)")
        print(f"  Products: {train_products} ({train_products/orig_products*100:.1f}%)")
        print(f"  Patents: {train_patents} ({train_patents/orig_patents*100:.1f}%)")
        
        print("\nTest graph:")
        print(f"  Nodes: {test_nodes} ({test_nodes/orig_nodes*100:.1f}%)")
        print(f"  Edges: {test_edges} ({test_edges/orig_edges*100:.1f}%)")
        print(f"  Products: {test_products} ({test_products/orig_products*100:.1f}%)")
        print(f"  Patents: {test_patents} ({test_patents/orig_patents*100:.1f}%)")
        
        print(f"\nEdges deleted: {orig_edges - train_edges - test_edges}")
        print(f"Distribution quality score: {score:.4f}")
    
    #creating graph
    G = nx.Graph()
    G.add_nodes_from(edges['Product'], bipartite='product')
    G.add_nodes_from(edges['patent_id'], bipartite='patent')
    G.add_edges_from(zip(edges['Product'], edges['patent_id']))
    
    # Run both strategies
    print("\nRunning bridge-first strategy...")
    train1, test1, score1 = split_strategy_bridge_first(G, fraction, n)
    
    print("\nRunning small-components-first strategy...")
    train2, test2, score2 = split_strategy_small_first(G, fraction, n)
    
    # Choose the better strategy
    if score1 <= score2:
        print(f"\nBridge-first strategy selected (score: {score1:.4f} vs {score2:.4f})")
        train_graph, test_graph, score = train1, test1, score1
        strategy_used = "bridge-first"
    else:
        print(f"\nSmall-components-first strategy selected (score: {score2:.4f} vs {score1:.4f})")
        train_graph, test_graph, score = train2, test2, score2
        strategy_used = "small-first"
    
    if verbose:
        print_splitting_state(G, train_graph, test_graph, score, strategy_used)
    train_products = [n for n in train_graph.nodes() if train_graph.nodes[n].get('bipartite') == 'product']
    train_patents = [n for n in train_graph.nodes() if train_graph.nodes[n].get('bipartite') == 'patent']
    test_products = [n for n in test_graph.nodes() if test_graph.nodes[n].get('bipartite') == 'product']
    test_patents = [n for n in test_graph.nodes() if test_graph.nodes[n].get('bipartite') == 'patent']
    train_df = edges[edges['Product'].isin(train_products) & edges['patent_id'].isin(train_patents)]
    test_df = edges[edges['Product'].isin(test_products) & edges['patent_id'].isin(test_patents)]
    return train_df, test_df

# Evaluation function to get embeddings
def get_embeddings(product_encoder, patent_encoder, dataloader, product_device, patent_device):
    product_encoder.eval()
    patent_encoder.eval()
    product_embeddings = {}
    patent_embeddings = {}
    progress_bar = tqdm(dataloader, desc="Generating embeddings")

    for batch in progress_bar:
        # Move batch to device
        product_input_ids = batch['product_input_ids'].to(product_device)
        product_attention_mask = batch['product_attention_mask'].to(product_device)
        patent_input_ids = batch['patent_input_ids'].to(patent_device)
        patent_attention_mask = batch['patent_attention_mask'].to(patent_device)

        # Get embeddings
        product_embs = product_encoder(product_input_ids, product_attention_mask)
        patent_embs = patent_encoder(patent_input_ids, patent_attention_mask)

        # Store embeddings with their IDs
        for i, (prod_id, pat_id) in enumerate(zip(batch['product_id'], batch['patent_id'])):
            product_embeddings[prod_id] = product_embs[i].cpu().numpy()
            patent_embeddings[pat_id] = patent_embs[i].cpu().numpy()

    return product_embeddings, patent_embeddings

def get_embeddings_frozen(product_encoder, patent_encoder, dataloader, product_device, patent_device):
    product_encoder.eval()
    patent_encoder.eval()
    product_embeddings = {}
    patent_embeddings = {}
    progress_bar = tqdm(dataloader, desc="Generating embeddings")

    for batch in progress_bar:
        # Move batch to device
        product_input_ids = batch['product_embedding'].to(product_device)
        product_attention_mask = batch['product_attention_mask'].to(product_device)
        patent_input_ids = batch['patent_embedding'].to(patent_device)
        patent_attention_mask = batch['patent_attention_mask'].to(patent_device)

        # Get embeddings
        product_embs = product_encoder(product_input_ids, product_attention_mask)
        patent_embs = patent_encoder(patent_input_ids, patent_attention_mask)

        # Store embeddings with their IDs
        # Note: collate_fn returns only unique products and patents per batch
        for i, prod_id in enumerate(batch['product_id']):
            product_embeddings[prod_id] = product_embs[i].cpu().numpy()
        
        for i, pat_id in enumerate(batch['patent_id']):
            patent_embeddings[pat_id] = patent_embs[i].cpu().numpy()

    return product_embeddings, patent_embeddings

# Function to evaluate performance with metrics
def evaluate_performance(product_embeddings, patent_embeddings, test_df, embedding_dim):
    # Convert embeddings to matrices
    unique_products = list(product_embeddings.keys())
    unique_patents =  list(patent_embeddings.keys())

    product_matrix = np.zeros((len(unique_products), embedding_dim))
    patent_matrix = np.zeros((len(unique_patents), embedding_dim))

    product_to_idx = {pid: i for i, pid in enumerate(unique_products)}
    patent_to_idx = {pid: i for i, pid in enumerate(unique_patents)}

    for pid, emb in product_embeddings.items():
        product_matrix[product_to_idx[pid],:] = emb

    for pid, emb in patent_embeddings.items():
        patent_matrix[patent_to_idx[pid],:] = emb

    # Create ground truth pairs
    product_patent_pairs = set(zip(test_df['Product'], test_df['patent_id']))

    # Initialize metrics
    ranks = []
    precisions_at_k = {1: [], 2: [], 3: [], 4: [], 5: [], 7: [], 10: [], 15: [], 20: []}
    y_true = []
    y_scores = []
    map=0

    for prod_id in unique_products:
        prod_idx = product_to_idx[prod_id]
        prod_emb = product_matrix[prod_idx]

        similarities = np.dot(patent_matrix, prod_emb)

        ranked_patents = [(unique_patents[i], sim) for i, sim in enumerate(similarities)]
        ranked_patents.sort(key=lambda x: x[1], reverse=True)

        true_patents = [pat_id for pat_id in unique_patents if (prod_id, pat_id) in product_patent_pairs]
        if not true_patents:
            continue

        for i, (pat_id, _) in enumerate(ranked_patents):
            if pat_id in true_patents:
                ranks.append(i + 1)
                break
        
        hits = 0
        avg_prec = 0
        for i, (pat_id, _) in enumerate(ranked_patents):
            if pat_id in true_patents:
                hits += 1
                avg_prec += hits / (i + 1)
                if hits == len(true_patents):
                    break
        if len(true_patents) > 0:
            map += (avg_prec / len(true_patents))

        for k in precisions_at_k.keys():
            top_k_patents = [pat_id for pat_id, _ in ranked_patents[:k]]
            relevant = sum(1 for pat_id in top_k_patents if pat_id in true_patents)
            precisions_at_k[k].append(relevant / min(k, len(true_patents)))

        for pat_id, sim in ranked_patents:
            y_true.append(1 if pat_id in true_patents else 0)
            y_scores.append(sim)

    mrr = np.mean([1/r for r in ranks]) if ranks else 0
    map = map / len(unique_products) if len(unique_products) > 0 else 0
    avg_precision_at_k = {k: np.mean(v) if v else 0 for k, v in precisions_at_k.items()}

    # Handle ROC and PR curves only if valid data exists
    if sum(y_true) > 0 and sum(y_true) < len(y_true):  # Both 0s and 1s exist
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
    else:
        fpr, tpr, roc_auc = [], [], 0.0
        precision, recall, pr_auc = [], [], 0.0
        print("Warning: Not enough positive/negative samples for ROC/PR curves.")

    return {
        'MRR': mrr,
        'MAP': map,
        'Precision@k': avg_precision_at_k,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'ROC_curve': (fpr, tpr),
        'PR_curve': (precision, recall)
    }

def evaluate_performance(product_embeddings, patent_embeddings, test_df, embedding_dim):
    # Convert embeddings to matrices
    unique_products = list(product_embeddings.keys())
    unique_patents =  list(patent_embeddings.keys())

    product_matrix = np.zeros((len(unique_products), embedding_dim))
    patent_matrix = np.zeros((len(unique_patents), embedding_dim))

    product_to_idx = {pid: i for i, pid in enumerate(unique_products)}
    patent_to_idx = {pid: i for i, pid in enumerate(unique_patents)}

    for pid, emb in product_embeddings.items():
        product_matrix[product_to_idx[pid],:] = emb

    for pid, emb in patent_embeddings.items():
        patent_matrix[patent_to_idx[pid],:] = emb

    # Create ground truth pairs
    product_patent_pairs = set(zip(test_df['Product'], test_df['patent_id']))

    # Initialize metrics
    ranks = []
    precisions_at_k = {1: [], 2: [], 3: [], 4: [], 5: [], 7: [], 10: [], 15: [], 20: []}
    y_true = []
    y_scores = []
    total_map = 0

    # Store per-product MAP and details
    per_product_map = []

    for prod_id in unique_products:
        prod_idx = product_to_idx[prod_id]
        prod_emb = product_matrix[prod_idx]

        similarities = np.dot(patent_matrix, prod_emb)

        ranked_patents = [(unique_patents[i], sim) for i, sim in enumerate(similarities)]
        ranked_patents.sort(key=lambda x: x[1], reverse=True)

        true_patents = [pat_id for pat_id in unique_patents if (prod_id, pat_id) in product_patent_pairs]
        if not true_patents:
            continue

        # Find first rank for MRR
        for i, (pat_id, _) in enumerate(ranked_patents):
            if pat_id in true_patents:
                ranks.append(i + 1)
                break

        # Compute MAP for this product
        hits = 0
        avg_prec = 0
        hit_ranks = []
        for i, (pat_id, _) in enumerate(ranked_patents):
            if pat_id in true_patents:
                hits += 1
                avg_prec += hits / (i + 1)
                hit_ranks.append((pat_id, i + 1))
                if hits == len(true_patents):
                    break
        if len(true_patents) > 0:
            prod_map = avg_prec / len(true_patents)
            total_map += prod_map
            per_product_map.append((prod_id, prod_map, hit_ranks))

        for k in precisions_at_k.keys():
            top_k_patents = [pat_id for pat_id, _ in ranked_patents[:k]]
            relevant = sum(1 for pat_id in top_k_patents if pat_id in true_patents)
            precisions_at_k[k].append(relevant / min(k, len(true_patents)))

        for pat_id, sim in ranked_patents:
            y_true.append(1 if pat_id in true_patents else 0)
            y_scores.append(sim)

    mrr = np.mean([1/r for r in ranks]) if ranks else 0
    map = total_map / len(per_product_map) if per_product_map else 0
    avg_precision_at_k = {k: np.mean(v) if v else 0 for k, v in precisions_at_k.items()}

    # Handle ROC and PR curves only if valid data exists
    if sum(y_true) > 0 and sum(y_true) < len(y_true):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
    else:
        fpr, tpr, roc_auc = [], [], 0.0
        precision, recall, pr_auc = [], [], 0.0
        print("Warning: Not enough positive/negative samples for ROC/PR curves.")

    # Print bottom 10 products by MAP
    print("\nBottom 10 products by MAP:")
    per_product_map.sort(key=lambda x: x[1])  # Sort ascending by MAP
    for prod_id, prod_map, hit_ranks in per_product_map[:10]:
        print(f"Product ID: {prod_id}, MAP: {prod_map:.4f}")
        for pat_id, rank in hit_ranks:
            print(f"  ↳ Patent ID: {pat_id}, Rank: {rank}")
        if not hit_ranks:
            print("  ↳ No correct patents found in ranking")

    return {
        'MRR': mrr,
        'MAP': map,
        'Precision@k': avg_precision_at_k,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'ROC_curve': (fpr, tpr),
        'PR_curve': (precision, recall)
    }

# Plotting functions
def plot_metrics(metrics, train_losses, val_losses, adv_rewards, prefix):
    _, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot Training and Validation Loss
    epochs = range(1, len(train_losses) + 1)
    axs[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axs[0, 0].plot(epochs, val_losses, 'g-', label='Validation Loss')
    if adv_rewards:
        axs[0, 0].plot(epochs, adv_rewards, 'm-', label='Adversarial Reward')
    axs[0, 0].set_xlabel('Training Steps')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot Precision@k
    k_values = list(metrics['Precision@k'].keys())
    precisions = list(metrics['Precision@k'].values())
    axs[0, 1].bar(k_values, precisions)
    axs[0, 1].set_title(f'Precision@k, MRR, MAP')
    axs[0, 1].set_xlabel('k')
    axs[0, 1].set_ylabel('Precision')
    # Add dotted horizontal lines for MRR and MRE
    axs[0, 1].axhline(y=metrics["MRR"], color='red', linestyle=':', label=f'MRR: {metrics["MRR"]:.3f}')
    axs[0, 1].axhline(y=metrics["MAP"], color='blue', linestyle=':', label=f'MAP: {metrics["MAP"]:.3f}')
    axs[0, 1].legend()

    # Plot PR curve
    precision, recall = metrics['PR_curve']
    axs[1, 0].plot(recall, precision, lw=2)
    axs[1, 0].set_xlim([0.0, 1.0])
    axs[1, 0].set_ylim([0.0, 1.05])
    axs[1, 0].set_xlabel('Recall')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].set_title(f'Precision-Recall Curve (AUC = {metrics["PR_AUC"]:.3f})')
    
    # Plot ROC curve
    fpr, tpr = metrics['ROC_curve']
    axs[1, 1].plot(fpr, tpr, lw=2)
    axs[1, 1].plot([0, 1], [0, 1], 'k--', lw=2)
    axs[1, 1].set_xlim([0.0, 1.0])
    axs[1, 1].set_ylim([0.0, 1.05])
    axs[1, 1].set_xlabel('False Positive Rate')
    axs[1, 1].set_ylabel('True Positive Rate')
    axs[1, 1].set_title(f'ROC Curve (AUC = {metrics["ROC_AUC"]:.3f})')

    plt.tight_layout()
    plt.savefig('../Images/' + prefix + '_evaluation_metrics.png')

    # Display numeric metrics
    print('\n' + prefix + " evaluation metrics :")
    print(f"Mean Reciprocal Rank (MRR): {metrics['MRR']:.4f}")
    print(f"Mean Average Precision (MAP): {metrics['MAP']:.4f}")
    print("\nPrecision@k:")
    for k, p in metrics['Precision@k'].items():
        print(f"  P@{k}: {p:.4f}")
    print(f"\nROC AUC: {metrics['ROC_AUC']:.4f}")
    print(f"PR AUC: {metrics['PR_AUC']:.4f}")

def plot_policy_heatmap(prod_history, pat_history, prefix=False):
    _, axs = plt.subplots(1, 2, figsize=(25, 10))
    prod_policy_matrix = torch.stack(prod_history).numpy()
    pat_policy_matrix = torch.stack(pat_history).numpy()
    im1 = axs[0].imshow(prod_policy_matrix.T, aspect='auto', cmap='viridis')
    plt.colorbar(im1, ax=axs[0], label='Probability')
    axs[0].set_xlabel('Training Step')
    axs[0].set_ylabel('Product Index')
    axs[0].set_title('Product Policy Heatmap')
    
    im2 = axs[1].imshow(pat_policy_matrix.T, aspect='auto', cmap='viridis')
    plt.colorbar(im2, ax=axs[1], label='Probability')
    axs[1].set_xlabel('Training Step')
    axs[1].set_ylabel('Patent Index')
    axs[1].set_title('Patent Policy Heatmap')
    if prefix:
        plt.savefig("2RL" + 'policy_heatmap.png')
    else:
        plt.savefig('../Images/policy_heatmap.png')

# Save embeddings to CSV
def save_embeddings(embeddings_dict, prefix):
    """Helper to save embeddings with IDs as index."""
    df = pd.DataFrame.from_dict(embeddings_dict, orient='index')
    df.to_csv(f"../Data/{prefix}_embeddings.csv")
    return df