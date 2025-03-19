import torch
import torch.nn as nn

from model.transformer_pytorch import TransformerEncoderLayer


class PhenoFormer(nn.Module):
    def __init__(
        self,
        target_list,
        d_in=7,
        d_out=1,
        d_model=64,
        nhead=8,
        dim_feedforward=128,
        n_layers=1,
        positional_encoding=True,
        elevation=False,
        latlon=False,
        T_pos_enc=1000,
        phases_as_input=None,
        **kwargs
    ):
        """Attention-based architecture for phenology modelling from
        climate time series.

        Args:
            target_list (list[str]): list of the names of the phenophases
            to be predicted, format: ["{species_name}:{phenophase_name}"]
            d_in (int, optional): Number of channels of the input time series 
            of climate variables. Defaults to 7.
            d_out (int, optional): Output dimension. Defaults to 1.
            d_model (int, optional): Dimension of the inner representations of the model.
            Defaults to 64.
            nhead (int, optional): Number of heads in the attention layer. Defaults to 8.
            dim_feedforward (int, optional): Number of neurons in the feedforward layer.
            Defaults to 128.
            n_layers (int, optional): Number of stacked attention layers. Defaults to 1.
            positional_encoding (bool, optional): If true, add positional encoding 
            to the input time series. Defaults to True.
            elevation (bool, optional): If true the elevation of the observaton site 
            is concatenated to the input data. Defaults to False.
            latlon (bool, optional): If true the geo-location of the observaton site 
            is concatenated to the input data. Defaults to False.
            T_pos_enc (int, optional): Maximal period used in the positional encoding. 
            Defaults to 1000.
            phases_as_input (list[str]): List of phenophase dates that 
            are given as input. Defaults to None.
        """
        super(PhenoFormer, self).__init__()

        self.positional_encoding = None
        if positional_encoding:
            self.positional_encoding = PositionalEncoder(d=d_model, T=T_pos_enc)
        self.target_list = target_list
        self.n_task = len(target_list) # number of phenophases to predict 
        self.d_out = d_out
        self.elevation = elevation
        self.latlon = latlon
        self.add_static = elevation + latlon + (phases_as_input is not None)
        self.phases_as_input = phases_as_input

        # Dimension of the static features that are potentially concatenated to the input
        self.d_static = 1 * elevation + 2 * latlon
        if phases_as_input is not None:
            self.d_static += len(phases_as_input)

        # Shared Linear Encoder
        self.shared_linear_encoder = nn.Sequential(nn.Linear(d_in + self.d_static, d_model))

        # Learnt tokens for each phenophase to predict
        self.learnt_tokens = nn.ParameterDict(
            {
                t: nn.Parameter(torch.rand((1, d_model))).requires_grad_()
                for t in target_list
            }
        )

        # Transformer layer for temporal encoding 
        if n_layers == 1:
            self.transformer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            )
        else:
            self.transformer = nn.Sequential(
                *[
                    TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        batch_first=True,
                    )
                    for _ in range(n_layers)
                ]
            )

        # Linear decoders (one per phenophase)
        self.linear_decoder = nn.Linear(
            in_features=d_model, out_features=d_out * self.n_task, bias=True
        )

    def forward(self, batch, return_attention=False):
        x = batch["climate"]
        b, t, c = x.shape

        # Concatenation of static features (if any)
        if self.d_static > 0:
            static_data = []
            if self.elevation:
                static_data.append(batch["elevation_normalised"].unsqueeze(1))
            if self.latlon:
                static_data.append(batch["latlon_normalised"])
            if self.phases_as_input is not None:
                for p in self.phases_as_input:
                    static_data.append(batch["input_phases"][p].unsqueeze(1))
            static_data = torch.cat(static_data, 1)
            x = torch.cat([x, static_data.unsqueeze(1).repeat(1, t, 1)], dim=2)
            b, t, c = x.shape

        # Shared linear encoder applied in parallel to all time steps 
        out = self.shared_linear_encoder(x.view(b * t, c)).view(b, t, -1)

        # Add positional encoding to encode the time information
        if self.positional_encoding is not None:
            positions = batch["doys"]
            out = out + self.positional_encoding(positions)

        # Prepend learnt tokens
        learnt_tokens = torch.cat([self.learnt_tokens[t] for t in self.target_list], 0)
        learnt_tokens = learnt_tokens.unsqueeze(0).repeat((b, 1, 1))
        out = torch.cat([learnt_tokens, out], dim=1)

        # Apply transformer layer
        if return_attention:
            if isinstance(self.encoder, nn.Sequential):
                attentions = []
                for layer in self.transformer:
                    out, attention = layer(out, return_attention=True)
                    attentions.append(attention)
                attention = torch.stack(
                    attentions, 0
                )  # n_layer x B x n_head x target_sequence x source_sequence
            else:
                out, attention = self.encoder(out, return_attention=True)
        else:
            out = self.transformer(out)

        # Decoding
        # Retrieve the embedding of each learnt token (each phenophase)
        task_embeddings = out[:, : self.n_task, :]  # batch x n_task x d_model
        # Linear decoder for each phenophase d_model -> 1 (or d_out)
        preds = self.linear_decoder(task_embeddings)
        # Split the output into the different phenophases
        predictions = {
            self.target_list[i]: chunk[:, i, :].squeeze(1)
            for i, chunk in enumerate(preds.chunk(self.n_task, dim=2))
        }
        if return_attention:
            return predictions, attention
        return predictions


class PositionalEncoder(nn.Module):
    """Positional encoding for the transformer model."""
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table



if __name__=="__main__":
    model = PhenoFormer(
        target_list=["European_beech:leaf_unfolding",
                        "European_larch:needle_emergence",
                        "Common_spruce:needle_emergence",
                        "Hazel:leaf_unfolding"]
                        )
    
    T = 365
    d = 7 
    batch_size = 32

    dummy_climate = torch.rand(batch_size, T, d)
    doys = torch.arange(T).unsqueeze(0).repeat(batch_size, 1)

    out = model({"climate": dummy_climate, "doys": doys})
    print(out)