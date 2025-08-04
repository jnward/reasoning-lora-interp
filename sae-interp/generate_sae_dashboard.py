#!/usr/bin/env python3
"""
Generate HTML dashboard for SAE feature interpretation.
"""

import json
import argparse
from typing import Dict, List, Any


def generate_html_dashboard(data: Dict[str, Any], output_path: str):
    """Generate the HTML dashboard with embedded data"""
    
    metadata = data['metadata']
    features = data['features']
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAE Feature Interpretation Dashboard</title>
    <style>
        /* Reset and base styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Header */
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .metadata {
            color: #666;
            font-size: 14px;
        }
        
        /* Feature selector */
        .feature-selector {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .feature-label {
            font-weight: bold;
        }
        
        .nav-button {
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
        }
        
        .nav-button:hover {
            background: #2980b9;
        }
        
        .nav-button:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }
        
        .feature-input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            width: 100px;
        }
        
        .feature-info {
            color: #666;
            font-size: 14px;
            margin-left: auto;
        }
        
        .feature-dropdown {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
            background: white;
        }
        
        /* Toggle switch */
        .toggle-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 26px;
            background-color: #ccc;
            border-radius: 13px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .toggle-switch.active {
            background-color: #3498db;
        }
        
        .toggle-slider {
            position: absolute;
            top: 3px;
            left: 3px;
            width: 20px;
            height: 20px;
            background-color: white;
            border-radius: 50%;
            transition: transform 0.3s;
        }
        
        .toggle-switch.active .toggle-slider {
            transform: translateX(24px);
        }
        
        .toggle-label {
            font-size: 14px;
            color: #666;
        }
        
        /* Examples container */
        .examples-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .examples-header {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .max-activation {
            font-size: 14px;
            color: #666;
            font-weight: normal;
        }
        
        /* Example item */
        .example-item {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
            background: #fafafa;
        }
        
        .example-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-size: 14px;
            color: #666;
        }
        
        .activation-value {
            font-weight: bold;
            color: #e74c3c;
        }
        
        /* Token display */
        .token-display {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 14px;
            line-height: 1.8;
            padding: 10px;
            background: white;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .token {
            display: inline;
            padding: 2px 0;
            border-radius: 3px;
            transition: all 0.2s;
            position: relative;
        }
        
        .token.target {
            font-weight: bold;
            border: 2px solid #e74c3c;
            padding: 2px 4px;
            margin: 0 2px;
        }
        
        .token:hover {
            background: #f0f0f0;
        }
        
        /* Activation color scale */
        .activation-0 { background-color: rgba(255, 255, 255, 0); }
        .activation-1 { background-color: rgba(255, 230, 230, 0.3); }
        .activation-2 { background-color: rgba(255, 200, 200, 0.4); }
        .activation-3 { background-color: rgba(255, 170, 170, 0.5); }
        .activation-4 { background-color: rgba(255, 140, 140, 0.6); }
        .activation-5 { background-color: rgba(255, 110, 110, 0.7); }
        .activation-6 { background-color: rgba(255, 80, 80, 0.8); }
        .activation-7 { background-color: rgba(255, 50, 50, 0.9); }
        .activation-8 { background-color: rgba(255, 20, 20, 1.0); }
        
        /* Tooltip */
        .tooltip {
            position: absolute;
            background: #333;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            z-index: 1000;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }
        
        .tooltip.show {
            opacity: 1;
        }
        
        /* No data message */
        .no-data {
            text-align: center;
            color: #999;
            padding: 40px;
            font-style: italic;
        }
        
        /* Loading */
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        /* Keyboard shortcuts help */
        .shortcuts-help {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px 15px;
            border-radius: 4px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">SAE Feature Interpretation Dashboard</div>
            <div class="metadata">
                ${metadata.n_features} features | ${metadata.top_k} examples per feature | 
                ${metadata.n_rollouts_processed} rollouts processed
            </div>
        </div>
        
        <div class="feature-selector">
            <span class="feature-label">Feature:</span>
            <button class="nav-button" id="prevBtn" onclick="navigateFeature(-1)">←</button>
            <input type="number" id="featureInput" class="feature-input" 
                   min="0" max="${metadata.n_features - 1}" value="0"
                   onchange="loadFeature(this.value)">
            <button class="nav-button" id="nextBtn" onclick="navigateFeature(1)">→</button>
            <select class="feature-dropdown" id="activeFeatureSelect" onchange="loadFeature(this.value)">
                <option value="">Select active feature...</option>
            </select>
            <div class="toggle-container">
                <span class="toggle-label">Only full examples:</span>
                <div class="toggle-switch" id="toggleFullExamples" onclick="toggleFullExamplesFilter()">
                    <div class="toggle-slider"></div>
                </div>
            </div>
            <span class="feature-info" id="featureInfo"></span>
        </div>
        
        <div class="examples-container" id="examplesContainer">
            <div class="loading">Loading...</div>
        </div>
        
        <div class="shortcuts-help">
            Use ← → arrow keys to navigate features
        </div>
    </div>
    
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        // Embed data
        const featuresData = ${json.dumps(features)};
        const metadata = ${json.dumps(metadata)};
        
        let currentFeature = 0;
        let fullExamplesOnly = false;
        
        // Load feature on page load
        window.addEventListener('load', () => {
            // Populate dropdown with active features
            populateActiveFeatures();
            
            // Load first active feature if exists
            const activeFeatures = getActiveFeatures();
            if (activeFeatures.length > 0) {
                loadFeature(activeFeatures[0]);
            } else {
                loadFeature(0);
            }
            
            // Keyboard navigation
            document.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowLeft') {
                    navigateFeature(-1);
                } else if (e.key === 'ArrowRight') {
                    navigateFeature(1);
                }
            });
        });
        
        function getActiveFeatures() {
            // Get list of features that have examples
            return Object.keys(featuresData)
                .filter(f => featuresData[f] && featuresData[f].examples && featuresData[f].examples.length > 0)
                .map(f => parseInt(f))
                .sort((a, b) => a - b);
        }
        
        function toggleFullExamplesFilter() {
            fullExamplesOnly = !fullExamplesOnly;
            const toggle = document.getElementById('toggleFullExamples');
            toggle.classList.toggle('active', fullExamplesOnly);
            
            // Repopulate dropdown
            populateActiveFeatures();
            
            // If current feature doesn't meet criteria, navigate to next valid one
            if (fullExamplesOnly) {
                const validFeatures = getFilteredFeatures();
                if (!validFeatures.includes(currentFeature) && validFeatures.length > 0) {
                    loadFeature(validFeatures[0]);
                }
            }
        }
        
        function getFilteredFeatures() {
            const activeFeatures = getActiveFeatures();
            
            if (!fullExamplesOnly) {
                return activeFeatures;
            }
            
            // Filter to only features with max examples
            return activeFeatures.filter(featureId => {
                const stats = featuresData[featureId].stats;
                return stats.n_examples === metadata.top_k;
            });
        }
        
        function populateActiveFeatures() {
            const select = document.getElementById('activeFeatureSelect');
            const activeFeatures = fullExamplesOnly ? getFilteredFeatures() : getActiveFeatures();
            
            // Clear existing options except the first
            select.innerHTML = '<option value="">Select active feature...</option>';
            
            // Add active features
            activeFeatures.forEach(featureId => {
                const stats = featuresData[featureId].stats;
                const option = document.createElement('option');
                option.value = featureId;
                option.textContent = `Feature ${featureId} (${stats.n_examples} examples, max=${stats.max_activation.toFixed(2)})`;
                select.appendChild(option);
            });
            
            // Add summary to dropdown
            const summaryOption = document.createElement('option');
            summaryOption.disabled = true;
            const filterText = fullExamplesOnly ? 'features with full examples' : 'active features total';
            summaryOption.textContent = `── ${activeFeatures.length} ${filterText} ──`;
            select.insertBefore(summaryOption, select.children[1]);
        }
        
        function navigateFeature(delta) {
            if (fullExamplesOnly) {
                // Navigate through filtered features only
                const validFeatures = getFilteredFeatures();
                const currentIndex = validFeatures.indexOf(currentFeature);
                
                if (currentIndex === -1) {
                    // Current feature not in filtered list, go to first/last
                    if (delta > 0 && validFeatures.length > 0) {
                        loadFeature(validFeatures[0]);
                    } else if (delta < 0 && validFeatures.length > 0) {
                        loadFeature(validFeatures[validFeatures.length - 1]);
                    }
                    return;
                }
                
                const newIndex = currentIndex + delta;
                if (newIndex >= 0 && newIndex < validFeatures.length) {
                    loadFeature(validFeatures[newIndex]);
                }
            } else {
                // Normal navigation
                const newFeature = currentFeature + delta;
                if (newFeature >= 0 && newFeature < metadata.n_features) {
                    loadFeature(newFeature);
                }
            }
        }
        
        function loadFeature(featureIdx) {
            featureIdx = parseInt(featureIdx);
            if (isNaN(featureIdx) || featureIdx < 0 || featureIdx >= metadata.n_features) {
                return;
            }
            
            currentFeature = featureIdx;
            document.getElementById('featureInput').value = featureIdx;
            
            // Update navigation buttons
            if (fullExamplesOnly) {
                const validFeatures = getFilteredFeatures();
                const currentIndex = validFeatures.indexOf(featureIdx);
                document.getElementById('prevBtn').disabled = currentIndex <= 0;
                document.getElementById('nextBtn').disabled = currentIndex === -1 || currentIndex >= validFeatures.length - 1;
            } else {
                document.getElementById('prevBtn').disabled = featureIdx === 0;
                document.getElementById('nextBtn').disabled = featureIdx === metadata.n_features - 1;
            }
            
            // Update feature info
            const featureData = featuresData[featureIdx];
            if (!featureData || !featureData.examples || featureData.examples.length === 0) {
                document.getElementById('featureInfo').textContent = 'No activations found';
                document.getElementById('examplesContainer').innerHTML = 
                    '<div class="no-data">No examples found for this feature</div>';
                return;
            }
            
            const stats = featureData.stats;
            document.getElementById('featureInfo').textContent = 
                `${stats.n_examples} examples found`;
            
            // Render examples
            renderExamples(featureData);
        }
        
        function renderExamples(featureData) {
            const container = document.getElementById('examplesContainer');
            const maxActivation = featureData.stats.max_activation;
            
            let html = `
                <div class="examples-header">
                    <span>Top Activating Examples</span>
                    <span class="max-activation">Max activation: ${maxActivation.toFixed(4)}</span>
                </div>
            `;
            
            featureData.examples.forEach((example, idx) => {
                html += renderExample(example, idx + 1, maxActivation);
            });
            
            container.innerHTML = html;
            
            // Add hover listeners
            setupTokenHovers();
        }
        
        function renderExample(example, rank, maxActivation) {
            const tokens = example.tokens;
            const activations = example.activations;
            const targetPos = example.target_position;
            
            let tokensHtml = '';
            tokens.forEach((token, idx) => {
                const activation = activations[idx];
                const isTarget = idx === targetPos;
                const classes = ['token'];
                
                if (isTarget) {
                    classes.push('target');
                }
                
                // Color based on activation strength
                if (activation > 0) {
                    const intensity = Math.min(8, Math.floor((activation / maxActivation) * 8));
                    classes.push(`activation-${intensity}`);
                }
                
                tokensHtml += `<span class="${classes.join(' ')}" 
                    data-activation="${activation.toFixed(4)}">${escapeHtml(token)}</span>`;
            });
            
            return `
                <div class="example-item">
                    <div class="example-header">
                        <span>Rank #${rank} | Rollout ${example.rollout_idx}, Token ${example.token_idx}</span>
                        <span class="activation-value">Activation: ${example.activation_value.toFixed(4)}</span>
                    </div>
                    <div class="token-display">${tokensHtml}</div>
                </div>
            `;
        }
        
        function setupTokenHovers() {
            const tooltip = document.getElementById('tooltip');
            const tokens = document.querySelectorAll('.token');
            
            tokens.forEach(token => {
                token.addEventListener('mouseenter', (e) => {
                    const activation = e.target.dataset.activation;
                    tooltip.textContent = `Activation: ${activation}`;
                    tooltip.classList.add('show');
                });
                
                token.addEventListener('mousemove', (e) => {
                    tooltip.style.left = e.pageX + 10 + 'px';
                    tooltip.style.top = e.pageY - 30 + 'px';
                });
                
                token.addEventListener('mouseleave', () => {
                    tooltip.classList.remove('show');
                });
            });
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>'''
    
    # Replace template variables
    html_content = html_content.replace('${metadata.n_features}', str(metadata['n_features']))
    html_content = html_content.replace('${metadata.top_k}', str(metadata['top_k']))
    html_content = html_content.replace('${metadata.n_rollouts_processed}', str(metadata['n_rollouts_processed']))
    html_content = html_content.replace('${json.dumps(features)}', json.dumps(features))
    html_content = html_content.replace('${json.dumps(metadata)}', json.dumps(metadata))
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Dashboard generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate SAE interpretation dashboard')
    parser.add_argument('--input', type=str, default='sae_features_data.json',
                       help='Input JSON file with feature data')
    parser.add_argument('--output', type=str, default='sae_dashboard.html',
                       help='Output HTML file')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Generate dashboard
    generate_html_dashboard(data, args.output)
    
    # Print summary
    n_features = data['metadata']['n_features']
    n_features_with_examples = sum(1 for f in data['features'].values() 
                                  if f.get('examples'))
    print(f"\nSummary:")
    print(f"- Total features: {n_features}")
    print(f"- Features with examples: {n_features_with_examples}")
    print(f"- Dashboard saved to: {args.output}")


if __name__ == '__main__':
    main()