# Context-Aware Autointerp Implementation

## Overview

I've created a modified version of the autointerp system that uses **inline context annotations** instead of the tab-separated format used by SAELens. This makes it much easier to see what patterns trigger a feature.

## Format Comparison

### Original SAELens Format:
```
cat	8.7
sat	0.0
on	0.0
the	0.0
mat	6.2
```

### New Context-Aware Format:
```
The <<cat:8.7>> sat on the <<mat:6.2>>
```

## Key Features

1. **Inline Annotations**: Only tokens with activations above threshold are annotated
2. **Natural Reading**: The text flows naturally with annotations embedded
3. **Clear Context**: You can immediately see what words trigger the feature in context
4. **Same Normalization**: Still uses 0-10 scale like SAELens

## Files Created

### `autointerp_context.py`
- Full implementation with OpenAI API support
- Inline annotation formatting
- Configurable activation threshold
- Async processing

### `autointerp_context_demo.py`
- Demo script to test formatting
- Compare formats side-by-side
- Save prompts for batch processing
- No API calls required

## Usage Examples

```bash
# Test the format
python autointerp_context_demo.py --input sae_features.json --features 7 9 10

# Save prompts for batch processing
python autointerp_context_demo.py --input sae_features.json --save-prompts prompts.json

# Run full autointerp (requires API key)
python autointerp_context.py --input sae_features.json --output explanations.json
```

## Example Output

For a feature that responds to mathematical notation:

```
<neuron_activations>
a point \( y \ in \ math cal {<<H:8.7>>} \) such that the set \( \ left

system of vectors in $ \ math cal {<<H:10.0>>}$ .
</neuron_activations>
```

For a feature responding to colons and formatting:

```
<neuron_activations>
is not a square).

4. ** Result **<<::9.2>> - The number of cop r ime pairs \

of 0.

4. ** General Conclusion **<<::10.0>> <<-:2.5>> The determinant of the matrix \( A \
</neuron_activations>
```

## Benefits of This Format

1. **Intuitive**: Immediately see what triggers the feature
2. **Context Preserved**: Full sentences and structure visible
3. **Efficient**: Only annotate significant activations
4. **Flexible**: Adjustable threshold for what to show
5. **Compatible**: Can still use same models and approach as SAELens

## Configuration Options

- `threshold`: Minimum activation to annotate (default: 0.1)
- `max_examples`: Number of examples to include
- `model_name`: LLM to use for explanations
- `temperature`: Generation temperature

## Next Steps

You can now:
1. Use this format with any LLM for more intuitive interpretations
2. Adjust the threshold to show more/fewer annotations
3. Process the saved prompts with your preferred API
4. Compare results with standard SAELens format
