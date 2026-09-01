#!/usr/bin/env python
"""Generate ProteinMPNN structure embeddings from PDB structures.

Each protein chain is encoded independently. By default, the output concatenates
representations from six full-backbone and three C-alpha-only ProteinMPNN
encoders, producing a 1,152-dimensional embedding per residue. Embeddings from
multichain structures are concatenated in PDB chain order. Each record is stored
as an extensionless pickle containing ``mpnn_emb``, ``seq``, and ``length``.
"""

import argparse, os, glob, copy, csv, pickle
import numpy as np
import torch, biotite.structure
from biotite.structure.io import pdb
from biotite.structure  import filter_amino_acids, get_chains
from biotite.structure.residues import get_residues
from biotite.sequence   import ProteinSequence
from ProteinMPNN.protein_mpnn_utils import ProteinMPNN, tied_featurize

# Command-line interface
cli = argparse.ArgumentParser()
cli.add_argument("--pdb_folder",  required=True)
cli.add_argument("--out_csv",     required=True)
cli.add_argument("--out_pyd_dir", required=True)
cli.add_argument(
    "--skip_ca_only",
    action="store_true",
    help="Omit the three Cα-only encoders, reducing the embedding width from "
         "1,152 to 768. These embeddings are incompatible with the released "
         "model, which expects an input width of 1,152.",
)
args = cli.parse_args()
os.makedirs(args.out_pyd_dir, exist_ok=True)

# PDB parsing utilities
ATOM_NAMES = ["N","CA","C","O"]

def _load_pdb(path):
    arr  = pdb.get_structure(pdb.PDBFile.read(path), model=1)
    mask = ((arr.atom_name[:,None] == ATOM_NAMES).any(1) &
             filter_amino_acids(arr))
    return arr[mask]

def _coords4(arr):
    def per_res(s, axis=None):
        sel = np.stack([s.atom_name == n for n in ATOM_NAMES],1)
        xyz = s[sel.argmax(0)].coord
        xyz[sel.sum(0)==0] = np.nan
        return xyz
    return biotite.structure.apply_residue_wise(arr, arr, per_res)

# ProteinMPNN uses this 21-token residue alphabet; B and Z are unsupported.
_MPNN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")

def _seq(arr):
    def _res_to_letter(r):
        try:
            letter = ProteinSequence.convert_letter_3to1(r)
        except KeyError:
            return "X"  # Map non-standard residue names to the unknown token.
        if letter not in _MPNN_ALPHABET:
            return "X"  # Map unsupported ambiguous residues to the unknown token.
        return letter
    return "".join(_res_to_letter(r) for r in get_residues(arr)[1])

# ProteinMPNN encoder initialization
def _load_model(chk, *, ca):
    d = torch.load(chk, map_location="cpu")
    m = ProteinMPNN(ca_only=ca, num_letters=21,
                    node_features=128, edge_features=128,
                    hidden_dim=128, num_encoder_layers=3,
                    num_decoder_layers=3, augment_eps=0.,
                    k_neighbors=d["num_edges"]).cuda().eval()
    m.load_state_dict(d["model_state_dict"])
    return m

print("Loading ProteinMPNN encoders.")
root="ProteinMPNN"
model_1 = _load_model(f"{root}/vanilla_model_weights/v_48_002.pt",ca=False)
model_2 = _load_model(f"{root}/vanilla_model_weights/v_48_010.pt",ca=False)
model_3 = _load_model(f"{root}/vanilla_model_weights/v_48_020.pt",ca=False)
model_4 = _load_model(f"{root}/vanilla_model_weights/v_48_030.pt",ca=False)
model_5 = _load_model(f"{root}/soluble_model_weights/v_48_010.pt",ca=False)
model_6 = _load_model(f"{root}/soluble_model_weights/v_48_020.pt",ca=False)
if args.skip_ca_only:
    model_7=model_8=model_9=None
else:
    model_7=_load_model(f"{root}/ca_model_weights/v_48_002.pt",ca=True)
    model_8=_load_model(f"{root}/ca_model_weights/v_48_010.pt",ca=True)
    model_9=_load_model(f"{root}/ca_model_weights/v_48_020.pt",ca=True)
print("ProteinMPNN encoders initialized.\n")

# Structure embedding
def _embed_single(coords, seq, sample_name):
    """Return concatenated ProteinMPNN embeddings for one protein chain.

    The output dimension is 1,152 by default and 768 when Cα-only encoders
    are excluded.
    """
    samp = {
        "name": sample_name,
        "coords": coords,
        "seq": seq,
        "num_of_chains": 1,
        "visible_list": [],
        "masked_list": ["A"],
        "coords_chain_A": {
            "N_chain_A":  coords[:,0],
            "CA_chain_A": coords[:,1],
            "C_chain_A":  coords[:,2],
            "O_chain_A":  coords[:,3],
        },
        "seq_chain_A": seq,
    }
    chain_dict = {sample_name: (["A"], [])}

    with torch.no_grad():
        # Full-backbone encoder features.
        (_ ,X,S,mask,_,
         cM,cEnc,_,
         _ ,_ ,_ ,cMp,
         _ ,rIdx,_ ,_ ,_ ,_ ,_ ,_ ,_
        ) = tied_featurize([copy.deepcopy(samp)],"cuda",chain_dict)

        # Cα-only encoder features.
        (_ ,Xc,Sc,maskc,_,
         cM_c,cEnc_c,_,
         _ ,_ ,_ ,cMp_c,
         _ ,rIdx_c,_ ,_ ,_ ,_ ,_ ,_ ,_
        ) = tied_featurize([copy.deepcopy(samp)],"cuda",chain_dict,
                           ca_only=True)

        z_f = torch.randn_like(cM)
        z_c = torch.randn_like(cM_c)

        outs=[]
        for mdl,Xi,Si,maski,cM_,cMp_,rI,cE,z in [
            (model_1,X,S,mask,cM,cMp,rIdx,cEnc,z_f),
            (model_2,X,S,mask,cM,cMp,rIdx,cEnc,z_f),
            (model_3,X,S,mask,cM,cMp,rIdx,cEnc,z_f),
            (model_4,X,S,mask,cM,cMp,rIdx,cEnc,z_f),
            (model_5,X,S,mask,cM,cMp,rIdx,cEnc,z_f),
            (model_6,X,S,mask,cM,cMp,rIdx,cEnc,z_f),
            *( [] if args.skip_ca_only else [
               (model_7,Xc,Sc,maskc,cM_c,cMp_c,rIdx_c,cEnc_c,z_c),
               (model_8,Xc,Sc,maskc,cM_c,cMp_c,rIdx_c,cEnc_c,z_c),
               (model_9,Xc,Sc,maskc,cM_c,cMp_c,rIdx_c,cEnc_c,z_c)])
        ]:
            _, h = mdl(Xi,Si,maski,cM_*cMp_,rI,cE,z)
            outs.append(h)

        return torch.cat(outs,-1).squeeze(0).cpu()

# Dataset generation
# Write one extensionless embedding file per structure identifier.
written, skipped = [], []
with open(args.out_csv,"w",newline="",encoding="utf-8-sig") as csvfh:
    writer=csv.writer(csvfh)
    writer.writerow(["structure_id","seq","chain_ids"])

    for pdb_path in sorted(glob.glob(os.path.join(args.pdb_folder,"*.pdb"))):
        pdb_id = os.path.basename(pdb_path)[:-4]
        print(f"Processing structure {pdb_id}")

        try:
            struct=_load_pdb(pdb_path)
        except Exception as e:
            print(f"Skipping structure because PDB parsing failed: {e}.")
            skipped.append(pdb_id); continue

        chains=get_chains(struct)
        try:
            chains = list(chains) if chains is not None else []
        except Exception:
            chains = []
        if len(chains) == 0:
            print("Skipping structure because no protein chains were found.")
            skipped.append(pdb_id); continue

        emb_list=[]; seq_list=[]
        for ch in chains:
            frag = struct[struct.chain_id==ch]
            coords=_coords4(frag);  seq=_seq(frag)
            try:
                emb=_embed_single(coords,seq,f"{pdb_id}_{ch}")
                emb_list.append(emb);  seq_list.append(seq)
            except Exception as e:
                print(f"Skipping structure because embedding failed for chain {ch}: {e}.")
                skipped.append(pdb_id); emb_list=[]; break

        if not emb_list: continue

        emb_all=torch.cat(emb_list,0)
        seq_all=''.join(seq_list)

        # Store each embedding under the structure identifier used by the training loader.
        out_path = os.path.join(args.out_pyd_dir, pdb_id)
        with open(out_path, "wb") as f:
            pickle.dump({"mpnn_emb":emb_all,"seq":seq_all,
                         "length":emb_all.shape[0]},f)

        writer.writerow([pdb_id,seq_all,",".join(str(c) for c in chains)])
        written.append(pdb_id)

# Processing summary
total = len(written) + len(skipped)
print("\nProcessing summary")
print(f"Total structures: {total}")
print(f"Embeddings written: {len(written)}")
print(f"Structures skipped: {len(skipped)}")
if total:
    print(f"Completion rate: {100.0*len(written)/total:.1f}%")
if skipped:
    print("Skipped structure identifiers:")
    for x in skipped:
        print("   -", x)
print("Processing complete.")
