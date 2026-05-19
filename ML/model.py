import torch
import torch.nn as nn

class ParticleJetClassifier(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, num_heads=4, num_layers=5, hidden_dim=128):
        super(ParticleJetClassifier, self).__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        self.particle_fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 3)
        )

        self.jet_fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_old(self, x, padding_mask):
        x = self.embedding(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        particle_out = self.particle_fc(x)

        #jet_representation = torch.mean(x, dim=1)
        mask = (~padding_mask).unsqueeze(-1)   # [B, N, 1]
        x_masked = x * mask
        valid_counts = mask.sum(dim=1)          # [B, 1]
        jet_repr = torch.where(
            valid_counts > 0,
            x_masked.sum(dim=1) / valid_counts,
            torch.zeros_like(x_masked.sum(dim=1))
        )

        jet_out = self.jet_fc(jet_repr)

        return particle_out, jet_out

    def forward(self, x, padding_mask):
        # x: [B, N, F]
        # padding_mask: [B, N]  (True where padding)

        B, N, _ = x.shape

        x = self.embedding(x)

        # Build attention mask that blocks attention TO padded tokens
        # attn_mask shape must be [B, N, N]

        # padding_mask: True where padding
        key_padding = padding_mask.unsqueeze(1).expand(B, N, N)  # [B, N, N]

        # Transformer expects float mask with -inf for blocked positions
        attn_mask = key_padding.float()
        attn_mask = attn_mask.masked_fill(attn_mask == 1, float('-inf'))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        # Reshape for multihead attention: [B * num_heads, N, N]
        num_heads = self.transformer.layers[0].self_attn.num_heads
        attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)

        x = self.transformer(x, mask=attn_mask)

        particle_out = self.particle_fc(x)

        # Masked mean pooling (safe now — shape preserved)
        valid_mask = (~padding_mask).float()
        x_weighted = x * valid_mask.unsqueeze(-1)
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)

        jet_repr = x_weighted.sum(dim=1) / valid_counts

        jet_out = self.jet_fc(jet_repr)

        return particle_out, jet_out
