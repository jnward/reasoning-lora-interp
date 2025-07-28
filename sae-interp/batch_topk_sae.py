import torch
import torch.nn as nn
from einops import rearrange


class BatchTopKSAE(nn.Module):
    def __init__(self, d_model, dict_size, k):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.k = k
        
        # Encoder: d_model -> dict_size
        self.W_enc = nn.Parameter(torch.empty(d_model, dict_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        
        # Decoder: dict_size -> d_model (separate trainable parameter)
        self.W_dec = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize encoder weights
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")
        
        # Initialize decoder as transpose of encoder
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.T.clone()

    def get_latent_activations(self, x):
        """Compute initial feature activations."""
        # x: [batch, d_model]
        # returns: [batch, dict_size]
        return torch.einsum("bd,df->bf", x, self.W_enc) + self.b_enc
    
    def apply_batchtopk(self, latent_activations):
        """Apply batch top-k sparsity constraint using masking."""
        batch_size = latent_activations.shape[0]
        n_features = latent_activations.shape[1]
        
        # Calculate decoder norms for value scoring
        decoder_norms = torch.norm(self.W_dec, dim=1)
        
        # Calculate value scores
        value_scores = torch.einsum("bf,f->bf", latent_activations, decoder_norms)
        
        # Flatten scores to find top-k across batch
        flat_scores = rearrange(value_scores, "b f -> (b f)")
        
        # Find top k*batch_size activations
        # Make sure we don't try to select more than available
        total_elements = batch_size * n_features
        k_total = min(self.k * batch_size, total_elements)
        topk_indices = torch.topk(flat_scores, k=k_total, dim=0).indices
        
        # Create sparse mask
        mask = torch.zeros_like(flat_scores)
        mask[topk_indices] = 1.0
        mask = rearrange(mask, "(b f) -> b f", b=batch_size, f=n_features)
        
        # Apply mask to get sparse activations
        sparse_activations = latent_activations * mask
        
        return sparse_activations
    
    def decode(self, sparse_latent_activations):
        """Decode sparse activations back to input space."""
        # sparse_latent_activations: [batch, dict_size]
        # returns: [batch, d_model]
        return torch.einsum("bf,fd->bd", sparse_latent_activations, self.W_dec) + self.b_dec
    
    def encode(self, x):
        """
        Encode input to sparse latent representation.
        
        Args:
            x: Input activations of shape [batch, d_model]
            
        Returns:
            Sparse latent activations of shape [batch, dict_size]
        """
        latent_activations = self.get_latent_activations(x)
        sparse_latent_activations = self.apply_batchtopk(latent_activations)
        return sparse_latent_activations
    
    def forward(self, x):
        """
        Forward pass through the SAE.
        
        Args:
            x: Input activations of shape [batch, d_model]
            
        Returns:
            dict with:
                - reconstruction: Reconstructed activations [batch, d_model]
                - sparse_latent_activations: Sparse features [batch, dict_size]
                - latent_activations: Pre-sparsity features [batch, dict_size]
        """
        # Encode
        latent_activations = self.get_latent_activations(x)
        
        # Apply sparsity
        sparse_latent_activations = self.apply_batchtopk(latent_activations)
        
        # Decode
        reconstruction = self.decode(sparse_latent_activations)
        
        return {
            "reconstruction": reconstruction,
            "sparse_latent_activations": sparse_latent_activations,
            "latent_activations": latent_activations,
        }