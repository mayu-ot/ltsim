import os
import numpy as np
import fsspec
import torch
import torch.nn as nn

from torch.utils.data import default_collate


class TransformerWithToken(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        token_mask = torch.zeros(1, 1, dtype=torch.bool)
        self.register_buffer("token_mask", token_mask)

        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, src_key_padding_mask):
        # x: [N, B, E]
        # padding_mask: [B, N]
        #   `False` for valid values
        #   `True` for padded values

        B = x.size(1)

        token = self.token.expand(-1, B, -1)
        x = torch.cat([token, x], dim=0)

        token_mask = self.token_mask.expand(B, -1)
        padding_mask = torch.cat([token_mask, src_key_padding_mask], dim=1)

        x = self.core(x, src_key_padding_mask=padding_mask)

        return x


class FIDNet(nn.Module):
    def __init__(self, num_label):
        super().__init__()

        self.emb_label = nn.Embedding(num_label, 32)
        self.fc_bbox = nn.Linear(4, 32)
        self.transformer = TransformerWithToken(
            d_model=64, nhead=4, dim_feedforward=32, num_layers=4
        )
        self.fc_out = nn.Linear(64, 1)

    def extract_features(self, bbox, label, padding_mask):
        l = self.emb_label(label)
        b = self.fc_bbox(bbox)
        x = torch.cat([l, b], dim=-1).permute(1, 0, 2)
        x = self.transformer(x, padding_mask)
        return x[0]

    def forward(self, bbox, label, padding_mask):
        x = self.extract_features(bbox, label, padding_mask)
        x = self.fc_out(x)
        return x.squeeze(-1)


class FIDNetV2(nn.Module):
    def __init__(self, num_label, max_bbox=50):
        super().__init__()

        self.emb_label = nn.Embedding(num_label, 128)
        self.fc_bbox = nn.Linear(4, 128)
        self.encoder = TransformerWithToken(
            d_model=256, nhead=4, dim_feedforward=128, num_layers=8
        )

        self.fc_out = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.token = nn.Parameter(torch.rand(max_bbox, 1, 256))
        te = nn.TransformerEncoderLayer(d_model=256, dim_feedforward=128, nhead=4)
        self.decoder = nn.TransformerEncoder(te, num_layers=8)
        self.fc_out_cls = nn.Linear(256, num_label)
        self.fc_out_bbox = nn.Linear(256, 4)

    def extract_features(self, bbox, label, padding_mask):
        l = self.emb_label(label)
        b = self.fc_bbox(bbox)
        x = torch.cat([l, b], dim=-1).permute(1, 0, 2)
        x = self.encoder(x, padding_mask)
        return x[0]

    def forward(self, bbox, label, padding_mask):
        B, N, _ = bbox.size()
        x = self.extract_features(bbox, label, padding_mask)

        logit = self.fc_out(x).squeeze(-1)

        t = self.token[:N].expand(-1, B, -1)
        x = torch.cat([x.unsqueeze(0), t], dim=0)

        token_mask = self.encoder.token_mask.expand(B, -1)
        _padding_mask = torch.cat([token_mask, padding_mask], dim=1)

        x = self.decoder(x, src_key_padding_mask=_padding_mask)
        # x = x[1:].permute(1, 0, 2)[~padding_mask]
        x = x[1:].permute(1, 0, 2)

        logit_cls = self.fc_out_cls(x)
        bbox = torch.sigmoid(self.fc_out_bbox(x))

        return logit, logit_cls, bbox


class FIDNetV3(nn.Module):
    def __init__(self, num_label, d_model=256, nhead=4, num_layers=4, max_bbox=50):
        super().__init__()

        # encoder
        self.emb_label = nn.Embedding(num_label, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)

        self.enc_transformer = TransformerWithToken(
            d_model=d_model,
            dim_feedforward=d_model // 2,
            nhead=nhead,
            num_layers=num_layers,
        )

        self.fc_out_disc = nn.Linear(d_model, 1)

        # decoder
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)

        te = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model // 2
        )
        self.dec_transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out_cls = nn.Linear(d_model, num_label)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def extract_features(self, bbox, label, padding_mask):
        b = self.fc_bbox(bbox)
        l = self.emb_label(label)
        x = self.enc_fc_in(torch.cat([b, l], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, padding_mask)
        return x[0]

    def forward(self, bbox, label, padding_mask):
        B, N, _ = bbox.size()
        x = self.extract_features(bbox, label, padding_mask)

        logit_disc = self.fc_out_disc(x).squeeze(-1)

        x = x.unsqueeze(0).expand(N, -1, -1)
        t = self.pos_token[:N].expand(-1, B, -1)
        x = torch.cat([x, t], dim=-1)
        x = torch.relu(self.dec_fc_in(x))

        x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
        # x = x.permute(1, 0, 2)[~padding_mask]
        x = x.permute(1, 0, 2)

        # logit_cls: [B, N, L]    bbox_pred: [B, N, 4]
        logit_cls = self.fc_out_cls(x)
        bbox_pred = torch.sigmoid(self.fc_out_bbox(x))

        return logit_disc, logit_cls, bbox_pred


def load_fidnet_v3(
    fid_weight_file: str,
    num_classes: int,
    max_seq_length: int,
) -> nn.Module:
    fid_model = FIDNetV3(num_label=num_classes, max_bbox=max_seq_length)
    with fsspec.open(fid_weight_file, "rb") as file_obj:
        x = torch.load(file_obj, map_location=torch.device("cpu"))
    fid_model.load_state_dict(x["state_dict"])
    fid_model.eval()
    return fid_model


def collate(inputs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_seq_length = max(len(input_["category"]) for input_ in inputs)
    outputs = []
    for input_ in inputs:
        n = len(input_["category"])

        mask = torch.full((max_seq_length,), False, dtype=torch.bool)
        mask[:n] = True

        bbox = torch.full((max_seq_length, 4), 0.0)
        bbox[:n] = torch.tensor(input_["bbox"])

        category = torch.full((max_seq_length,), 0, dtype=torch.long)
        category[:n] = torch.tensor(input_["category"])

        outputs.append(
            {
                "bbox": bbox,
                "label": category,
                "padding_mask": ~mask,
            }
        )
    return default_collate(outputs)


def get_fid_features(
    layouts: list, fid_weight_file: str, num_classes: int, max_seq_length: int
):
    """Get features from a layout with a specified fid model.

    Args:
        layouts (list): List of layouts. Each layout is a dict with keys "bbox" and "category".
        layouts = [
            {
                "bbox": [[center_x, center_y, width, height], ...], # floats normalized to [0, 1]
                "category": [category_id, ...], # category_id is an integer
            },...
        ]
        fid_weight_file (str): Path to the fid model weight file.

    Returns:
        features: numpy array of shape (len(layouts), feature-dim).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 256
    fid_model = load_fidnet_v3(fid_weight_file, num_classes, max_seq_length).to(device)
    features = np.zeros((len(layouts), 256))

    for i_start in range(0, len(layouts), batch_size):
        i_end = min(i_start + batch_size, len(layouts))
        inputs = collate(layouts[i_start:i_end])
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            feat = fid_model.extract_features(**inputs)
            features[i_start:i_end] = feat.cpu().numpy()

    return features


def load_dataset(name: str) -> list:
    with open(name, "r") as f:
        result = json.load(f)

    if isinstance(result, dict):
        if "annotations" in result:
            layouts = list(result["annotations"].values())
        elif "results" in result:
            layouts = list(result["results"].values())
        else:
            raise RuntimeError
    else:
        layouts = result
    return layouts


if __name__ == "__main__":
    # note: requirements: torch, fsspec, (pytorch_fid)
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type", type=str, default="rico25", choices=["rico25", "publaynet"]
    )
    parser.add_argument("--input_dataset_json", type=str, default="")
    parser.add_argument("--output_feat_file", type=str, default="")
    parser.add_argument("--fid_weight_file", type=str)
    args = parser.parse_args()

    if args.dataset_type == "rico25":
        num_classes, max_seq_length = 25, 25
    elif args.dataset_type == "publaynet":
        num_classes, max_seq_length = 5, 25
    else:
        raise NotImplementedError

    layouts = load_dataset(args.input_dataset_json)

    feats = get_fid_features(
        layouts=layouts,
        fid_weight_file=args.fid_weight_file,
        num_classes=num_classes,
        max_seq_length=max_seq_length,
    )

    output_feat_file = Path(args.output_feat_file)
    if output_feat_file.exists():
        raise RuntimeError(f"{output_feat_file} exists")

    if not output_feat_file.parent.exists():
        output_feat_file.parent.mkdir(parents=True)

    np.save(args.output_feat_file, feats)
