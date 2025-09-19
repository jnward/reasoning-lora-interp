#!/usr/bin/env python3
"""
Generate LLM-based annotations for clusters using OpenAI API.

Takes the graph-based clustering results and generates concise descriptions
for each cluster based on the feature explanations within it.
"""

import json
import numpy as np
import argparse
from typing import List, Dict, Optional
from openai import OpenAI
import os
from tqdm import tqdm
import time


def load_clustering_results(results_path: str, interpretations_path: str):
    """Load clustering results and interpretations."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    with open(interpretations_path, 'r') as f:
        interp_data = json.load(f)
    
    # Extract graph clustering labels
    graph_labels = results['graph']['labels']
    feature_ids = results['feature_ids']
    
    # Create mapping from feature_id to explanation
    explanations_dict = {e['feature_id']: e['explanation'] for e in interp_data['explanations']}
    
    # Group features by cluster
    clusters = {}
    for idx, (fid, label) in enumerate(zip(feature_ids, graph_labels)):
        if label == -1:  # Skip noise
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            'feature_id': fid,
            'explanation': explanations_dict.get(fid, "")
        })
    
    return clusters


def generate_cluster_summary(cluster_features: List[Dict], cluster_id: int, 
                            client: OpenAI, model: str = "gpt-4o", verbose: bool = False) -> str:
    """
    Generate a concise summary for a cluster based on its feature explanations.
    """
    # Prepare the explanations
    explanations = [f['explanation'] for f in cluster_features if f['explanation']]
    
    if not explanations:
        return "Empty cluster"
    
    # Sample explanations if too many (to fit in context)
    if len(explanations) > 50:
        # Take a representative sample
        import random
        random.seed(42)
        sampled = random.sample(explanations, 50)
        explanations = sampled
    
    # Create prompt - more directive for gpt-5-mini
    prompt = f"""Analyze these {len(cluster_features)} related neural network features and output a single line summary.

Features:
{chr(10).join(f"- {exp}" for exp in explanations[:30])}  # Limit to 30 for context window

Output only a 10-20 word description of the common pattern. Do not include any other text, explanation, or formatting.
Example: "LaTeX mathematical notation and formatting commands"
Example: "Numeric tokens and arithmetic operators in equations"

Your summary:"""

    try:
        # Use different parameter names for gpt-5 models
        completion_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that analyzes patterns in neural network features."},
                {"role": "user", "content": prompt}
            ],
        }
        
        # gpt-5 models have different requirements
        if "gpt-5" in model:
            completion_params["max_completion_tokens"] = 50
            # gpt-5-mini only supports default temperature (1)
        else:
            completion_params["max_tokens"] = 50
            completion_params["temperature"] = 0.3  # Lower temperature for more consistent summaries
            
        response = client.chat.completions.create(**completion_params)
        
        summary = response.choices[0].message.content
        
        if verbose:
            print(f"\nCluster {cluster_id} raw response: '{summary}'")
        
        # Handle empty or None responses
        if not summary or summary.strip() == "":
            print(f"ERROR: Empty response for cluster {cluster_id}")
            return f"[No summary generated for cluster {cluster_id}]"
        
        # Clean up the summary
        summary = summary.strip()
        summary = summary.strip('"\'')
        if summary.endswith('.'):
            summary = summary[:-1]
            
        return summary
        
    except Exception as e:
        print(f"Error generating summary for cluster {cluster_id}: {e}")
        return f"Cluster with {len(cluster_features)} features"


def generate_all_summaries(clusters: Dict, client: OpenAI, model: str = "gpt-4o",
                          delay: float = 0.5, verbose: bool = False) -> Dict:
    """Generate summaries for all clusters."""
    summaries = {}
    
    print(f"Generating summaries for {len(clusters)} clusters using {model}...")
    
    for cluster_id in tqdm(sorted(clusters.keys()), desc="Processing clusters"):
        cluster_features = clusters[cluster_id]
        
        # Generate summary
        summary = generate_cluster_summary(cluster_features, cluster_id, client, model, verbose)
        
        summaries[cluster_id] = {
            'summary': summary,
            'size': len(cluster_features),
            'sample_features': [f['explanation'][:100] for f in cluster_features[:3]]
        }
        
        # Small delay to avoid rate limits
        time.sleep(delay)
    
    return summaries


def create_annotated_visualization(clusters: Dict, summaries: Dict, output_path: str):
    """Create a markdown file with annotated clusters."""
    
    lines = ["# Graph-Based Clustering with LLM Annotations\n"]
    lines.append(f"Generated summaries for {len(clusters)} clusters\n")
    lines.append("=" * 60 + "\n")
    
    # Sort by cluster size (descending)
    sorted_clusters = sorted(summaries.items(), key=lambda x: x[1]['size'], reverse=True)
    
    for cluster_id, info in sorted_clusters:
        lines.append(f"\n## Cluster {cluster_id}: {info['summary']}")
        lines.append(f"*Size: {info['size']} features*\n")
        
        # Show sample features
        lines.append("**Sample features:**")
        for i, sample in enumerate(info['sample_features'], 1):
            lines.append(f"- {sample}")
        
        # Show all features if cluster is small
        if info['size'] <= 10:
            lines.append("\n**All features:**")
            for feat in clusters[cluster_id]:
                lines.append(f"- Feature {feat['feature_id']}: {feat['explanation']}")
        
        lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved annotated clusters to {output_path}")


def save_json_summary(summaries: Dict, clusters: Dict, output_path: str):
    """Save summaries in JSON format for programmatic use."""
    
    json_data = {
        'metadata': {
            'num_clusters': len(summaries),
            'total_features': sum(s['size'] for s in summaries.values()),
            'method': 'graph_based_clustering'
        },
        'clusters': []
    }
    
    for cluster_id, info in summaries.items():
        cluster_data = {
            'id': cluster_id,
            'summary': info['summary'],
            'size': info['size'],
            'features': [
                {
                    'feature_id': f['feature_id'],
                    'explanation': f['explanation']
                }
                for f in clusters[cluster_id]
            ]
        }
        json_data['clusters'].append(cluster_data)
    
    # Sort by size
    json_data['clusters'].sort(key=lambda x: x['size'], reverse=True)
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved JSON summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LLM annotations for clusters")
    parser.add_argument("--results", type=str, default="clustering_results.json",
                       help="Path to clustering results JSON")
    parser.add_argument("--interpretations", type=str, default="all_interpretations_o3.json",
                       help="Path to interpretations JSON")
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="OpenAI model to use for summaries")
    parser.add_argument("--output-prefix", type=str, default="cluster_annotations",
                       help="Prefix for output files")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between API calls in seconds")
    parser.add_argument("--verbose", action="store_true",
                       help="Show verbose output including raw API responses")
    
    args = parser.parse_args()
    
    # Setup OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Load data
    print("Loading clustering results and interpretations...")
    clusters = load_clustering_results(args.results, args.interpretations)
    print(f"Found {len(clusters)} clusters from graph-based method")
    
    # Show cluster size distribution
    sizes = [len(features) for features in clusters.values()]
    print(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
    
    # Generate summaries
    summaries = generate_all_summaries(clusters, client, args.model, args.delay, args.verbose)
    
    # Create outputs
    print("\nCreating output files...")
    
    # Markdown visualization
    create_annotated_visualization(
        clusters, summaries, 
        f"{args.output_prefix}.md"
    )
    
    # JSON for programmatic use
    save_json_summary(
        summaries, clusters,
        f"{args.output_prefix}.json"
    )
    
    # Print top clusters
    print("\n" + "=" * 60)
    print("TOP 10 CLUSTERS BY SIZE:")
    print("=" * 60)
    
    sorted_summaries = sorted(summaries.items(), key=lambda x: x[1]['size'], reverse=True)[:10]
    
    for cluster_id, info in sorted_summaries:
        print(f"\nCluster {cluster_id} ({info['size']} features):")
        print(f"  {info['summary']}")
        print("  Sample features:")
        for sample in info['sample_features'][:2]:
            print(f"    - {sample}")
    
    print("\n" + "=" * 60)
    print("Annotation complete!")
    print(f"  Markdown: {args.output_prefix}.md")
    print(f"  JSON: {args.output_prefix}.json")
    print("=" * 60)


if __name__ == "__main__":
    main()