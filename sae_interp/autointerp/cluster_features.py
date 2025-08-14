#!/usr/bin/env python3
"""
Advanced clustering for SAE feature interpretations.

Handles blob-and-tendril structures using multiple clustering approaches:
1. HDBSCAN for density-based clustering
2. Graph-based community detection
3. Ensemble clustering combining multiple methods
"""

import json
import numpy as np
import argparse
from typing import List, Dict, Tuple, Optional
import hdbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import networkx as nx
from collections import Counter, defaultdict
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm


def load_data(embeddings_path: str, projections_path: str, interpretations_path: str):
    """Load embeddings, UMAP projections, and interpretations."""
    embeddings = np.load(embeddings_path)
    projections = np.load(projections_path)
    
    with open(interpretations_path, 'r') as f:
        data = json.load(f)
    
    feature_ids = [e['feature_id'] for e in data['explanations']]
    explanations = [e['explanation'] for e in data['explanations']]
    
    return embeddings, projections, feature_ids, explanations


def calculate_density_metrics(data: np.ndarray, k: int = 10) -> Dict:
    """Calculate local density metrics for each point."""
    nn = NearestNeighbors(n_neighbors=k+1)  # +1 because it includes self
    nn.fit(data)
    distances, indices = nn.kneighbors(data)
    
    # Remove self-distance (first column)
    distances = distances[:, 1:]
    
    # Calculate various density metrics
    metrics = {
        'mean_distance': distances.mean(axis=1),
        'median_distance': np.median(distances, axis=1),
        'density': 1.0 / (distances.mean(axis=1) + 1e-10),
        'relative_density': None
    }
    
    # Calculate relative density (compared to neighbors)
    relative_densities = []
    for i, neighbor_idx in enumerate(indices[:, 1:]):
        neighbor_densities = metrics['density'][neighbor_idx]
        relative_densities.append(metrics['density'][i] / (neighbor_densities.mean() + 1e-10))
    metrics['relative_density'] = np.array(relative_densities)
    
    return metrics


def hdbscan_clustering(data: np.ndarray, min_cluster_size: int = 10, 
                       min_samples: int = 5, cluster_selection_epsilon: float = 0.0) -> Dict:
    """
    Apply HDBSCAN clustering with soft clustering probabilities.
    
    Returns dict with labels, probabilities, and outlier scores.
    """
    print(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method='eom',  # Excess of Mass
        prediction_data=True,
        core_dist_n_jobs=-1
    )
    
    labels = clusterer.fit_predict(data)
    
    # Get soft clustering (probabilities)
    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
    
    # Get outlier scores
    outlier_scores = clusterer.outlier_scores_
    
    # Calculate cluster statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"  Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)")
    
    return {
        'labels': labels,
        'probabilities': soft_clusters,
        'outlier_scores': outlier_scores,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'clusterer': clusterer
    }


def graph_clustering(data: np.ndarray, k: int = 15, resolution: float = 1.0) -> Dict:
    """
    Graph-based clustering using k-NN graph and community detection.
    """
    print(f"Running graph-based clustering (k={k})...")
    
    # Build k-NN graph
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(data)
    distances, indices = nn.kneighbors(data)
    
    # Create weighted graph
    G = nx.Graph()
    for i in range(len(data)):
        for j, dist in zip(indices[i], distances[i]):
            if i != j:
                # Weight is inverse of distance (closer = higher weight)
                weight = 1.0 / (dist + 1e-10)
                G.add_edge(i, j, weight=weight)
    
    # Apply Louvain community detection
    import networkx.algorithms.community as nx_comm
    communities = nx_comm.louvain_communities(G, resolution=resolution, seed=42)
    
    # Convert to labels
    labels = np.zeros(len(data), dtype=int) - 1
    for comm_id, community in enumerate(communities):
        for node in community:
            labels[node] = comm_id
    
    n_clusters = len(communities)
    print(f"  Found {n_clusters} communities")
    
    # Calculate modularity (quality metric)
    modularity = nx_comm.modularity(G, communities)
    print(f"  Modularity: {modularity:.3f}")
    
    return {
        'labels': labels,
        'n_clusters': n_clusters,
        'modularity': modularity,
        'graph': G,
        'communities': communities
    }


def ensemble_clustering(clustering_results: List[Dict], n_samples: int) -> np.ndarray:
    """
    Combine multiple clustering results using consensus approach.
    """
    print("\nCreating ensemble clustering...")
    
    # Create co-association matrix
    coassoc = np.zeros((n_samples, n_samples))
    
    for result in clustering_results:
        labels = result['labels']
        # For each pair of points, add 1 if they're in the same cluster
        for i in range(n_samples):
            if labels[i] == -1:  # Skip noise points
                continue
            for j in range(i+1, n_samples):
                if labels[j] == -1:  # Skip noise points
                    continue
                if labels[i] == labels[j]:
                    coassoc[i, j] += 1
                    coassoc[j, i] += 1
    
    # Normalize by number of clustering results
    coassoc /= len(clustering_results)
    
    # Apply hierarchical clustering on co-association matrix
    # Convert similarity to distance
    distance_matrix = 1 - coassoc
    
    # Use average linkage hierarchical clustering
    agg_clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.5,  # Threshold for cutting dendrogram
        metric='precomputed',
        linkage='average'
    )
    
    ensemble_labels = agg_clusterer.fit_predict(distance_matrix)
    
    n_clusters = len(set(ensemble_labels))
    print(f"  Ensemble produced {n_clusters} clusters")
    
    return ensemble_labels, coassoc


def identify_cluster_representatives(embeddings: np.ndarray, labels: np.ndarray, 
                                    explanations: List[str], n_representatives: int = 5) -> Dict:
    """
    Find representative features for each cluster.
    """
    representatives = {}
    
    for cluster_id in set(labels):
        if cluster_id == -1:  # Skip noise
            continue
            
        # Get indices of features in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        # Calculate centroid of cluster
        cluster_embeddings = embeddings[cluster_indices]
        centroid = cluster_embeddings.mean(axis=0)
        
        # Find features closest to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_indices = cluster_indices[np.argsort(distances)[:n_representatives]]
        
        representatives[int(cluster_id)] = {
            'indices': closest_indices.tolist(),
            'explanations': [explanations[i] for i in closest_indices],
            'size': len(cluster_indices)
        }
    
    return representatives


def create_clustering_visualization(projections: np.ndarray, clustering_results: Dict,
                                   density_metrics: Dict, feature_ids: List[int],
                                   explanations: List[str], output_path: str):
    """
    Create comprehensive visualization of clustering results.
    """
    import plotly.subplots as sp
    
    # Create subplots
    fig = sp.make_subplots(
        rows=2, cols=3,
        subplot_titles=('HDBSCAN Clustering', 'Graph Communities', 'Ensemble Clustering',
                       'Local Density', 'Outlier Scores', 'Cluster Sizes'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # Prepare hover text
    hover_texts = [f"Feature {fid}: {exp[:50]}..." for fid, exp in zip(feature_ids, explanations)]
    
    # Helper function to generate random colors for clusters
    def generate_cluster_colors(labels):
        """Generate random colors for each unique cluster label."""
        import random
        random.seed(42)  # For reproducibility
        
        unique_labels = list(set(labels))
        colors = {}
        
        for label in unique_labels:
            if label == -1:  # Noise points in gray
                colors[label] = '#808080'
            else:
                # Generate random RGB color
                colors[label] = '#{:06x}'.format(random.randint(0, 0xFFFFFF))
        
        # Map colors to all points
        return [colors[label] for label in labels]
    
    # 1. HDBSCAN clustering
    hdbscan_labels = clustering_results['hdbscan']['labels']
    hdbscan_colors = generate_cluster_colors(hdbscan_labels)
    fig.add_trace(
        go.Scatter(
            x=projections[:, 0], y=projections[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=hdbscan_colors,
                showscale=False
            ),
            text=[f"{ht}<br>Cluster: {label}" for ht, label in zip(hover_texts, hdbscan_labels)],
            hovertemplate='%{text}<extra></extra>',
            name='HDBSCAN'
        ),
        row=1, col=1
    )
    
    # 2. Graph communities
    graph_labels = clustering_results['graph']['labels']
    graph_colors = generate_cluster_colors(graph_labels)
    fig.add_trace(
        go.Scatter(
            x=projections[:, 0], y=projections[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=graph_colors,
                showscale=False
            ),
            text=[f"{ht}<br>Community: {label}" for ht, label in zip(hover_texts, graph_labels)],
            hovertemplate='%{text}<extra></extra>',
            name='Graph'
        ),
        row=1, col=2
    )
    
    # 3. Ensemble clustering
    ensemble_labels = clustering_results['ensemble']['labels']
    ensemble_colors = generate_cluster_colors(ensemble_labels)
    fig.add_trace(
        go.Scatter(
            x=projections[:, 0], y=projections[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=ensemble_colors,
                showscale=False
            ),
            text=[f"{ht}<br>Ensemble: {label}" for ht, label in zip(hover_texts, ensemble_labels)],
            hovertemplate='%{text}<extra></extra>',
            name='Ensemble'
        ),
        row=1, col=3
    )
    
    # 4. Local density
    fig.add_trace(
        go.Scatter(
            x=projections[:, 0], y=projections[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=np.log1p(density_metrics['density']),
                colorscale='Hot',
                showscale=True,
                colorbar=dict(title="Log Density", x=0.3, y=0.35, len=0.3)
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            name='Density'
        ),
        row=2, col=1
    )
    
    # 5. Outlier scores
    outlier_scores = clustering_results['hdbscan']['outlier_scores']
    fig.add_trace(
        go.Scatter(
            x=projections[:, 0], y=projections[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=outlier_scores,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Outlier Score", x=0.65, y=0.35, len=0.3)
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            name='Outliers'
        ),
        row=2, col=2
    )
    
    # 6. Cluster size distribution
    cluster_sizes = Counter(ensemble_labels)
    fig.add_trace(
        go.Bar(
            x=list(cluster_sizes.keys()),
            y=list(cluster_sizes.values()),
            marker=dict(color=list(cluster_sizes.keys()), colorscale='Viridis'),
            text=[f"Cluster {k}: {v} features" for k, v in cluster_sizes.items()],
            hovertemplate='%{text}<extra></extra>',
            name='Sizes'
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        title="SAE Feature Clustering Analysis",
        height=800,
        width=1400,
        showlegend=False,
        hovermode='closest'
    )
    
    # Update axes labels
    for i in range(1, 3):
        for j in range(1, 4):
            if not (i == 2 and j == 3):  # Skip the bar chart
                fig.update_xaxes(title_text="UMAP 1", row=i, col=j)
                fig.update_yaxes(title_text="UMAP 2", row=i, col=j)
    
    fig.update_xaxes(title_text="Cluster ID", row=2, col=3)
    fig.update_yaxes(title_text="Number of Features", row=2, col=3)
    
    # Save
    fig.write_html(output_path)
    print(f"\nSaved visualization to {output_path}")


def save_clustering_results(clustering_results: Dict, representatives: Dict,
                           feature_ids: List[int], output_path: str):
    """Save clustering results to JSON."""
    # Convert numpy arrays to lists for JSON serialization
    results_for_json = {
        'hdbscan': {
            'labels': clustering_results['hdbscan']['labels'].tolist(),
            'n_clusters': clustering_results['hdbscan']['n_clusters'],
            'n_noise': clustering_results['hdbscan']['n_noise'],
            'outlier_scores': clustering_results['hdbscan']['outlier_scores'].tolist()
        },
        'graph': {
            'labels': clustering_results['graph']['labels'].tolist(),
            'n_clusters': clustering_results['graph']['n_clusters'],
            'modularity': clustering_results['graph']['modularity']
        },
        'ensemble': {
            'labels': clustering_results['ensemble']['labels'].tolist(),
            'n_clusters': len(set(clustering_results['ensemble']['labels']))
        },
        'representatives': representatives,
        'feature_ids': feature_ids
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    print(f"Saved clustering results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Advanced clustering for SAE features")
    parser.add_argument("--embeddings", type=str, default="feature_embeddings.npy",
                       help="Path to embeddings file")
    parser.add_argument("--projections", type=str, default="umap_projections_2d.npy",
                       help="Path to UMAP projections")
    parser.add_argument("--interpretations", type=str, default="all_interpretations_o3.json",
                       help="Path to interpretations JSON")
    parser.add_argument("--min-cluster-size", type=int, default=10,
                       help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min-samples", type=int, default=5,
                       help="Minimum samples for HDBSCAN")
    parser.add_argument("--graph-k", type=int, default=15,
                       help="Number of neighbors for graph construction")
    parser.add_argument("--output-prefix", type=str, default="clustering",
                       help="Prefix for output files")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    embeddings, projections, feature_ids, explanations = load_data(
        args.embeddings, args.projections, args.interpretations
    )
    
    print(f"Loaded {len(embeddings)} features")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"UMAP dimension: {projections.shape[1]}")
    
    # Calculate density metrics
    print("\nCalculating density metrics...")
    density_metrics = calculate_density_metrics(projections, k=10)
    
    # Apply multiple clustering methods
    clustering_results = {}
    
    # 1. HDBSCAN on UMAP projections
    clustering_results['hdbscan'] = hdbscan_clustering(
        projections, 
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples
    )
    
    # 2. HDBSCAN on original embeddings (with scaling)
    print("\nTrying HDBSCAN on original embeddings...")
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    clustering_results['hdbscan_embeddings'] = hdbscan_clustering(
        scaled_embeddings,
        min_cluster_size=args.min_cluster_size * 2,  # Larger for high-dim
        min_samples=args.min_samples
    )
    
    # 3. Graph-based clustering
    clustering_results['graph'] = graph_clustering(
        projections,
        k=args.graph_k
    )
    
    # 4. Ensemble clustering
    ensemble_inputs = [
        clustering_results['hdbscan'],
        clustering_results['hdbscan_embeddings'],
        clustering_results['graph']
    ]
    ensemble_labels, coassoc = ensemble_clustering(ensemble_inputs, len(embeddings))
    clustering_results['ensemble'] = {
        'labels': ensemble_labels,
        'coassoc': coassoc
    }
    
    # Find cluster representatives
    print("\nIdentifying cluster representatives...")
    representatives = identify_cluster_representatives(
        embeddings, ensemble_labels, explanations, n_representatives=5
    )
    
    # Print summary
    print("\n" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    
    for cluster_id, info in sorted(representatives.items()):
        print(f"\nCluster {cluster_id} ({info['size']} features):")
        for i, exp in enumerate(info['explanations'][:3]):
            print(f"  {i+1}. {exp}")
    
    # Create visualization
    print("\nCreating visualizations...")
    create_clustering_visualization(
        projections, clustering_results, density_metrics,
        feature_ids, explanations,
        f"{args.output_prefix}_visualization.html"
    )
    
    # Save results
    save_clustering_results(
        clustering_results, representatives, feature_ids,
        f"{args.output_prefix}_results.json"
    )
    
    print("\n" + "="*60)
    print("Clustering complete!")
    print(f"Visualization: {args.output_prefix}_visualization.html")
    print(f"Results: {args.output_prefix}_results.json")
    print("="*60)


if __name__ == "__main__":
    main()