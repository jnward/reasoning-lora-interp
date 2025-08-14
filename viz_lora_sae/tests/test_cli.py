"""
CLI integration tests.
"""

import pytest
import tempfile
import subprocess
import json
from pathlib import Path

from .synthetic_data import (
    generate_synthetic_activations,
    generate_preaggregated_data
)


class TestCLI:
    """Test CLI functionality."""
    
    def test_cli_preagg_mode(self):
        """Test CLI with pre-aggregated data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Generate test data
            features_path = generate_preaggregated_data(
                num_features=15,
                num_sites=8,
                categories=['math', 'logic', 'syntax'],
                output_path=str(tmpdir / "features.csv")
            )
            
            # Create categories file
            categories = [
                {'id': 'math', 'label': 'Mathematics'},
                {'id': 'logic', 'label': 'Logic'},
                {'id': 'syntax', 'label': 'Syntax'}
            ]
            cat_path = tmpdir / "categories.json"
            with open(cat_path, 'w') as f:
                json.dump(categories, f)
            
            # Run CLI
            cmd = [
                'python', '-m', 'viz_lora_sae',
                '--mode', 'preagg',
                '--features', features_path,
                '--categories-file', str(cat_path),
                '--top-k', '3',
                '--out-sankey', str(tmpdir / 'sankey.html'),
                '--out-bars', str(tmpdir / 'bars.html'),
                '--snapshot', str(tmpdir / 'snapshot.json')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check execution succeeded
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            
            # Check output files exist
            assert (tmpdir / 'sankey.html').exists()
            assert (tmpdir / 'bars.html').exists()
            assert (tmpdir / 'snapshot.json').exists()
            
            # Check snapshot content
            with open(tmpdir / 'snapshot.json', 'r') as f:
                snapshot = json.load(f)
            
            assert 'params' in snapshot
            assert 'nodes' in snapshot
            assert 'links' in snapshot
            assert 'summary' in snapshot
            
            assert snapshot['params']['mode'] == 'preagg'
            assert snapshot['params']['top_k'] == 3
    
    def test_cli_raw_mode(self):
        """Test CLI with raw activation data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Generate synthetic activations
            lora_dir, sae_path, labels_path = generate_synthetic_activations(
                num_rollouts=2,
                num_tokens=30,
                num_layers=2,
                num_features=8,
                adapter_types=['gate', 'up'],
                output_dir=str(tmpdir)
            )
            
            # Run CLI
            cmd = [
                'python', '-m', 'viz_lora_sae',
                '--mode', 'raw',
                '--lora-acts', lora_dir,
                '--sae-features', sae_path,
                '--labels', labels_path,
                '--categories-file', str(tmpdir / 'categories.json'),
                '--top-k', '2',
                '--out-sankey', str(tmpdir / 'sankey.html'),
                '--out-bars', str(tmpdir / 'bars.html'),
                '--snapshot', str(tmpdir / 'snapshot.json')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check execution succeeded
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            
            # Check output files exist
            assert (tmpdir / 'sankey.html').exists()
            assert (tmpdir / 'bars.html').exists()
            assert (tmpdir / 'snapshot.json').exists()
            
            # Verify HTML files are valid
            with open(tmpdir / 'sankey.html', 'r') as f:
                html_content = f.read()
                assert '<html>' in html_content or '<!DOCTYPE html>' in html_content
                assert 'plotly' in html_content.lower()
    
    def test_cli_robust_mean(self):
        """Test CLI with robust mean option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Generate synthetic activations
            lora_dir, sae_path, labels_path = generate_synthetic_activations(
                num_rollouts=1,
                num_tokens=20,
                num_layers=2,
                num_features=4,
                output_dir=str(tmpdir)
            )
            
            # Run CLI with robust mean
            cmd = [
                'python', '-m', 'viz_lora_sae',
                '--mode', 'raw',
                '--lora-acts', lora_dir,
                '--sae-features', sae_path,
                '--labels', labels_path,
                '--categories-file', str(tmpdir / 'categories.json'),
                '--robust-mean',  # Enable robust mean
                '--top-k', '2',
                '--out-sankey', str(tmpdir / 'sankey_robust.html'),
                '--out-bars', str(tmpdir / 'bars_robust.html'),
                '--snapshot', str(tmpdir / 'snapshot_robust.json')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            
            # Check snapshot indicates robust mean was used
            with open(tmpdir / 'snapshot_robust.json', 'r') as f:
                snapshot = json.load(f)
            
            assert snapshot['params']['robust_mean'] == True
    
    def test_cli_error_handling(self):
        """Test CLI error handling for missing arguments."""
        # Test missing required arguments
        cmd = ['python', '-m', 'viz_lora_sae', '--mode', 'preagg']
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0
        assert 'required' in result.stderr.lower()
        
        # Test invalid mode
        cmd = ['python', '-m', 'viz_lora_sae', '--mode', 'invalid']
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0
        
        # Test missing categories
        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = generate_preaggregated_data(
                output_path=f"{tmpdir}/features.csv"
            )
            
            cmd = [
                'python', '-m', 'viz_lora_sae',
                '--mode', 'preagg',
                '--features', features_path
                # Missing --categories or --categories-file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])