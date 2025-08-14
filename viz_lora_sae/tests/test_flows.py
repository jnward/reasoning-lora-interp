"""
Unit tests for flow computation module.
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
from pathlib import Path

from ..flows import (
    load_inputs_mode_a,
    load_inputs_mode_b,
    normalize_confidences,
    aggregate_middle_nodes,
    _trimmed_mean
)
from .synthetic_data import (
    generate_synthetic_activations,
    generate_preaggregated_data
)


class TestFlowComputation:
    """Test flow computation functions."""
    
    def test_trimmed_mean(self):
        """Test trimmed mean calculation."""
        # Test with simple array
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = _trimmed_mean(arr, trim_percent=0.1)
        # Should remove 1 and 10, mean of [2,3,4,5,6,7,8,9] = 5.5
        assert abs(result - 5.5) < 0.01
        
        # Test with empty array
        assert _trimmed_mean(np.array([]), trim_percent=0.1) == 0.0
        
        # Test with single element
        assert _trimmed_mean(np.array([5.0]), trim_percent=0.1) == 5.0
    
    def test_load_preaggregated(self):
        """Test loading pre-aggregated data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate test data
            csv_path = generate_preaggregated_data(
                num_features=10,
                num_sites=6,
                categories=['cat1', 'cat2'],
                output_path=f"{tmpdir}/features.csv"
            )
            
            # Load data
            df = load_inputs_mode_a(csv_path)
            
            # Check structure
            assert len(df) == 10
            assert set(df.columns) == {'feature_id', 'category', 'conf', 'mass', 'site_contrib'}
            
            # Check site_contrib is dict
            assert isinstance(df.iloc[0]['site_contrib'], dict)
            
            # Check mass is positive
            assert (df['mass'] > 0).all()
    
    def test_load_raw_activations(self):
        """Test loading and computing flows from raw activations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate synthetic data
            lora_dir, sae_path, labels_path = generate_synthetic_activations(
                num_rollouts=2,
                num_tokens=50,
                num_layers=2,
                num_features=6,
                adapter_types=['gate', 'up'],
                output_dir=tmpdir
            )
            
            # Load and compute flows
            df = load_inputs_mode_b(
                lora_dir,
                sae_path,
                labels_path,
                robust=False
            )
            
            # Check structure
            assert 'feature_id' in df.columns
            assert 'category' in df.columns
            assert 'mass' in df.columns
            assert 'site_contrib' in df.columns
            
            # Check mass conservation
            total_mass = df['mass'].sum()
            assert total_mass >= 0  # Should be non-negative
            
            # Check site contributions sum to mass
            for _, row in df.iterrows():
                site_sum = sum(row['site_contrib'].values())
                assert abs(site_sum - row['mass']) < 1e-6
    
    def test_normalize_confidences(self):
        """Test confidence normalization."""
        # Create test data with multiple labels per feature
        df = pd.DataFrame([
            {'feature_id': 'f1', 'conf': 0.6, 'category': 'cat1', 'mass': 1.0, 'site_contrib': {}},
            {'feature_id': 'f1', 'conf': 0.3, 'category': 'cat2', 'mass': 1.0, 'site_contrib': {}},
            {'feature_id': 'f2', 'conf': 1.0, 'category': 'cat1', 'mass': 2.0, 'site_contrib': {}},
        ])
        
        # Normalize
        df_norm = normalize_confidences(df)
        
        # Check f1 confidences sum to 1
        f1_confs = df_norm[df_norm['feature_id'] == 'f1']['conf'].values
        assert abs(f1_confs.sum() - 1.0) < 1e-6
        
        # Check f2 confidence unchanged (already 1.0)
        f2_conf = df_norm[df_norm['feature_id'] == 'f2']['conf'].values[0]
        assert abs(f2_conf - 1.0) < 1e-6
    
    def test_aggregate_middle_nodes(self):
        """Test node aggregation for Sankey diagram."""
        # Create test data
        df = pd.DataFrame([
            {'feature_id': f'f{i}', 'category': 'cat1' if i < 8 else 'cat2',
             'mass': 10 - i, 'conf': 1.0,
             'site_contrib': {'L0.gate': 5 - i/2, 'L1.up': 5 - i/2}}
            for i in range(10)
        ])
        
        categories = ['cat1', 'cat2']
        sites = ['L0.gate', 'L1.up']
        
        # Aggregate with top_k=3
        nodes, links, palette = aggregate_middle_nodes(df, categories, sites, top_k=3)
        
        # Check nodes structure
        assert 'labels' in nodes
        assert 'types' in nodes
        assert 'colors' in nodes
        
        # Check we have site nodes
        assert 'LoRA:L0.gate' in nodes['labels']
        assert 'LoRA:L1.up' in nodes['labels']
        
        # Check we have category nodes
        assert 'Cat:cat1' in nodes['labels']
        assert 'Cat:cat2' in nodes['labels']
        
        # Check we have feature nodes (top 3 for cat1)
        assert 'Ff0' in nodes['labels']
        assert 'Ff1' in nodes['labels']
        assert 'Ff2' in nodes['labels']
        
        # Check we have "Other" node for cat1 (features 3-7)
        assert 'Other (cat1)' in nodes['labels']
        
        # Check links structure
        assert 'source' in links
        assert 'target' in links
        assert 'value' in links
        assert 'color' in links
        
        # Check link counts are consistent
        assert len(links['source']) == len(links['target'])
        assert len(links['source']) == len(links['value'])
        assert len(links['source']) == len(links['color'])
        
        # Check palette
        assert 'cat1' in palette
        assert 'cat2' in palette
        assert palette['cat1'].startswith('#')  # Hex color


class TestMassConservation:
    """Test mass conservation properties."""
    
    def test_flow_conservation(self):
        """Test that flows are conserved through the pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate test data
            df = pd.DataFrame([
                {'feature_id': f'f{i}', 'category': 'cat1',
                 'mass': 1.0, 'conf': 1.0,
                 'site_contrib': {'L0.gate': 0.5, 'L1.up': 0.5}}
                for i in range(5)
            ])
            
            categories = ['cat1']
            sites = ['L0.gate', 'L1.up']
            
            # Aggregate
            nodes, links, palette = aggregate_middle_nodes(df, categories, sites, top_k=10)
            
            # Check total outgoing from sites equals total incoming to features
            site_outgoing = {}
            for i, source in enumerate(links['source']):
                if nodes['types'][source] == 'site':
                    label = nodes['labels'][source]
                    if label not in site_outgoing:
                        site_outgoing[label] = 0
                    site_outgoing[label] += links['value'][i]
            
            # Check total incoming to categories
            cat_incoming = {}
            for i, target in enumerate(links['target']):
                if nodes['types'][target] == 'category':
                    label = nodes['labels'][target]
                    if label not in cat_incoming:
                        cat_incoming[label] = 0
                    cat_incoming[label] += links['value'][i]
            
            # Total from sites should equal total mass
            total_site_out = sum(site_outgoing.values())
            total_mass = df['mass'].sum()
            assert abs(total_site_out - total_mass) < 1e-6
            
            # Total to categories should equal total mass (since conf=1.0)
            total_cat_in = sum(cat_incoming.values())
            assert abs(total_cat_in - total_mass) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])