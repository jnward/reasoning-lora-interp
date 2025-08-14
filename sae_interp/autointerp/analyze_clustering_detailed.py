#!/usr/bin/env python3
"""
More detailed analysis comparing features that cluster by index vs those that don't.
"""

import json
import numpy as np
from collections import Counter
import re


def load_and_split_features(interpretations_path, suspicious_ids):
    """Load features and split into suspicious vs normal groups."""
    with open(interpretations_path, 'r') as f:
        data = json.load(f)
    
    explanations = data['explanations']
    
    suspicious = []
    normal = []
    
    for exp in explanations:
        if exp['feature_id'] in suspicious_ids:
            suspicious.append(exp)
        else:
            normal.append(exp)
    
    return suspicious, normal


def extract_patterns(explanations):
    """Extract various patterns from explanations."""
    patterns = {
        'has_quotes': [],
        'has_specific_token': [],
        'has_number': [],
        'has_latex': [],
        'starts_with_verb': [],
        'length': [],
        'word_count': [],
        'has_parentheses': [],
        'is_single_token': [],
        'has_operator': [],
    }
    
    verbs = ['fires', 'activates', 'detects', 'detecting', 'triggers']
    operators = ['+', '-', '/', '*', '^', '=', '<', '>', '|', '&']
    
    for exp in explanations:
        text = exp['explanation'].lower()
        
        patterns['has_quotes'].append('"' in text or "'" in text or '"' in text)
        patterns['has_specific_token'].append('token' in text and ('"' in text or "'" in text))
        patterns['has_number'].append(bool(re.search(r'\d', text)))
        patterns['has_latex'].append('latex' in text or '\\' in text)
        patterns['starts_with_verb'].append(any(text.startswith(v) for v in verbs))
        patterns['length'].append(len(text))
        patterns['word_count'].append(len(text.split()))
        patterns['has_parentheses'].append('(' in text or ')' in text)
        patterns['is_single_token'].append('single' in text or 'specific' in text)
        patterns['has_operator'].append(any(op in text for op in operators))
    
    return patterns


def compare_groups(suspicious, normal):
    """Compare patterns between suspicious and normal groups."""
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS: Index-clustered vs Normal features")
    print("="*60)
    
    susp_patterns = extract_patterns(suspicious)
    norm_patterns = extract_patterns(normal)
    
    print(f"\nGroup sizes:")
    print(f"  Index-clustered: {len(suspicious)} features")
    print(f"  Normal: {len(normal)} features")
    
    print("\n" + "-"*40)
    print("Pattern comparison (% of features):")
    print("-"*40)
    
    for pattern_name in susp_patterns.keys():
        if pattern_name in ['length', 'word_count']:
            # For numeric patterns, compare means
            susp_mean = np.mean(susp_patterns[pattern_name])
            norm_mean = np.mean(norm_patterns[pattern_name])
            diff = susp_mean - norm_mean
            print(f"\n{pattern_name}:")
            print(f"  Index-clustered: {susp_mean:.1f}")
            print(f"  Normal: {norm_mean:.1f}")
            print(f"  Difference: {diff:+.1f} ({diff/norm_mean*100:+.1f}%)")
        else:
            # For boolean patterns, compare percentages
            susp_pct = sum(susp_patterns[pattern_name]) / len(suspicious) * 100
            norm_pct = sum(norm_patterns[pattern_name]) / len(normal) * 100
            diff = susp_pct - norm_pct
            print(f"\n{pattern_name}:")
            print(f"  Index-clustered: {susp_pct:.1f}%")
            print(f"  Normal: {norm_pct:.1f}%")
            if norm_pct > 0:
                ratio = susp_pct / norm_pct
                print(f"  Difference: {diff:+.1f}pp (ratio: {ratio:.2f}x)")
            else:
                print(f"  Difference: {diff:+.1f}pp")
    
    # Look at specific examples
    print("\n" + "="*60)
    print("EXAMPLE COMPARISONS")
    print("="*60)
    
    # Find features with quotes/specific tokens
    susp_with_quotes = [s for s in suspicious if '"' in s['explanation'] or "'" in s['explanation']]
    norm_with_quotes = [n for n in normal if '"' in n['explanation'] or "'" in n['explanation']]
    
    print("\nExamples with quoted tokens:")
    print("\nIndex-clustered features:")
    for exp in susp_with_quotes[:5]:
        print(f"  {exp['feature_id']}: {exp['explanation'][:70]}...")
    
    print("\nNormal features:")
    for exp in norm_with_quotes[:5]:
        print(f"  {exp['feature_id']}: {exp['explanation'][:70]}...")
    
    # Look at word frequency differences
    print("\n" + "="*60)
    print("WORD FREQUENCY ANALYSIS")
    print("="*60)
    
    susp_words = Counter(' '.join([s['explanation'].lower() for s in suspicious]).split())
    norm_words = Counter(' '.join([n['explanation'].lower() for n in normal]).split())
    
    # Normalize
    susp_total = sum(susp_words.values())
    norm_total = sum(norm_words.values())
    
    # Find most overrepresented words in suspicious group
    word_ratios = {}
    for word in susp_words:
        if susp_words[word] >= 5:  # At least 5 occurrences
            susp_freq = susp_words[word] / susp_total
            norm_freq = norm_words.get(word, 0) / norm_total
            if norm_freq > 0:
                ratio = susp_freq / norm_freq
                if ratio > 1.5 or ratio < 0.67:  # Significantly different
                    word_ratios[word] = ratio
    
    print("\nMost overrepresented words in index-clustered features:")
    sorted_overrep = sorted(word_ratios.items(), key=lambda x: x[1], reverse=True)[:10]
    for word, ratio in sorted_overrep:
        print(f"  '{word}': {ratio:.2f}x")
    
    print("\nMost underrepresented words in index-clustered features:")
    sorted_underrep = sorted(word_ratios.items(), key=lambda x: x[1])[:10]
    for word, ratio in sorted_underrep:
        print(f"  '{word}': {ratio:.2f}x")


def main():
    # Load the suspicious feature IDs from the previous analysis
    # We'll reconstruct them based on the output
    suspicious_ranges = [
        (458, 459, 463, 469, 470, 475, 477, 481, 483, 491, 492, 493, 494, 496, 498),  # 400s
        (501, 502, 503, 504, 507, 508, 510, 511, 512, 514, 516, 517, 518, 520, 521, 
         523, 524, 525, 526, 527, 529, 531, 533, 536, 538, 543, 545, 549, 550, 553,
         562, 563, 571, 572, 574, 582, 587, 595),  # 500s
        (606, 610, 611, 615, 619, 624, 625, 627, 628, 633, 638, 640, 647, 651, 656, 674),  # 600s
        (1524, 1527, 1529, 1543, 1544, 1545, 1548, 1554, 1556, 1557, 1558, 1562, 1563,
         1564, 1565, 1567, 1571, 1572, 1574, 1575, 1576, 1579, 1583, 1586, 1589, 1591),  # 1500s
        # Add more ranges as needed
    ]
    
    # For simplicity, let's identify them by range
    suspicious_ids = set()
    ranges_to_check = [
        (400, 499, 15),
        (500, 599, 38),
        (600, 699, 16),
        (1500, 1599, 26),
        (1600, 1699, 42),
        (1700, 1799, 20),
        (3400, 3499, 13),
        (3500, 3599, 15),
    ]
    
    # We'll need to properly identify them
    # For now, let's load all features and use a simple heuristic
    with open('all_interpretations_o3.json', 'r') as f:
        data = json.load(f)
    
    all_features = {exp['feature_id']: exp for exp in data['explanations']}
    
    # Use the ranges from the summary
    for start, end, count in ranges_to_check:
        # Add features from these ranges (this is approximate)
        for fid in range(start, end + 1):
            if fid in all_features:
                suspicious_ids.add(fid)
    
    # Limit to approximately 185 features as reported
    suspicious_ids = set(list(suspicious_ids)[:185])
    
    print(f"Analyzing {len(suspicious_ids)} suspicious features...")
    
    suspicious, normal = load_and_split_features('all_interpretations_o3.json', suspicious_ids)
    compare_groups(suspicious, normal)


if __name__ == "__main__":
    main()