
import os
import torch
import numpy as np
import sys
import fm

model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()

# Read FASTA
fasta_path = sys.argv[1]
output_path = sys.argv[2]

sequences = []
ids = []
with open(fasta_path, 'r') as f:
    current_seq = []
    current_id = None
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if current_id is not None:
                sequences.append(''.join(current_seq))
                ids.append(current_id)
            current_id = line[1:]
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        sequences.append(''.join(current_seq))
        ids.append(current_id)

print(f"Processing {len(sequences)} sequences...")

# Generate embeddings in batches
all_embeddings = []
all_ids = []
batch_size = 8

def sanitize_and_truncate(seq: str, max_len: int = 1022) -> str:
    seq = seq.upper().replace('T', 'U')
    cleaned = ''.join(ch if ch in {'A', 'C', 'G', 'U'} else 'A' for ch in seq)
    if len(cleaned) == 0:
        cleaned = 'A'
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned

sequences = [sanitize_and_truncate(s) for s in sequences]

requested_device = os.environ.get("RNAFM_DEVICE", "auto").lower()
if requested_device == "cpu":
    device = "cpu"
elif requested_device == "cuda":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = model.to(device=device)
print(f"RNA-FM device: {device}")

for i in range(0, len(sequences), batch_size):
    batch_seqs = sequences[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]
    data = list(zip(batch_ids, batch_seqs))
    labels, strs, toks = batch_converter(data)

    def embed_with_current_device(token_tensor):
        token_tensor = token_tensor.to(device)
        with torch.no_grad():
            outputs = model(token_tensor, repr_layers=[12])
            token_embeddings = outputs["representations"][12]

            pooled_embeddings = []
            for b_idx, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                seq_emb = token_embeddings[b_idx, 1:seq_len+1].mean(dim=0)
                pooled_embeddings.append(seq_emb)

            return torch.stack(pooled_embeddings, dim=0)

    try:
        batch_embeddings = embed_with_current_device(toks)
    except RuntimeError as e:
        msg = str(e).lower()
        if ("cuda" in msg or "device-side assert" in msg) and device.startswith("cuda"):
            print("CUDA failure in RNA-FM batch; retrying on CPU for stability")
            torch.cuda.empty_cache()
            device = "cpu"
            model_cpu = model.to(device="cpu")
            model_cpu.eval()
            model = model_cpu
            batch_embeddings = embed_with_current_device(toks)
        else:
            raise

    all_embeddings.append(batch_embeddings.cpu().numpy())
    all_ids.extend(batch_ids)
    
    print(f"Processed {min(i+batch_size, len(sequences))}/{len(sequences)} sequences")

# Concatenate all
embeddings = np.vstack(all_embeddings).astype(np.float32)
print(f"Final shape: {embeddings.shape}")

# Save as .npz
np.savez(output_path, ids=np.array(all_ids, dtype=object), embeddings=embeddings)
print(f"Saved to {output_path}")
