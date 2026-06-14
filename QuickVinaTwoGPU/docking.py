import hashlib
import json
import os
import subprocess
from pathlib import Path

import psutil
import re

from meeko import MoleculePreparation
from rdkit import Chem
from rdkit.Chem import AllChem

HASH_PREFIX_LEN = 16
MAPPING_FILENAME = "ligand_map.json"


def _load_mapping(map_path: Path):
    if map_path.exists():
        try:
            with open(map_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}


def _save_mapping(map_path: Path, mapping: dict):
    map_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = map_path.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    os.replace(tmp_path, map_path)


def _generate_ligand_name(smiles: str, existing_map: dict):
    seed = ""
    counter = 0
    while True:
        digest = hashlib.sha1((smiles + seed).encode('utf-8')).hexdigest()
        ligand_name = f"mol_{digest[:HASH_PREFIX_LEN]}"
        entry = existing_map.get(ligand_name)
        if entry is None or entry.get('smiles') == smiles:
            return ligand_name
        counter += 1
        seed = f"_{counter}"


def _get_mapping_path(lig_dir: str) -> Path:
    return Path(lig_dir) / MAPPING_FILENAME


def dock(receptor_input,
         smiles,
         center_x=14.444,
         center_y=5.250,
         center_z=-18.278,
         size_x=20,
         size_y=20,
         size_z=20,
         lig_dir=None,
         out_dir=None,
         log_dir=None,
        conf_dir=None,
        vina_cwd=None,
         seed=None):
    timeout_duration = 10000

    # mkdir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(lig_dir, exist_ok=True)

    canonical_smiles = smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
       canonical_smiles = Chem.MolToSmiles(mol)

    map_path = _get_mapping_path(lig_dir)
    mapping = _load_mapping(map_path)
    ligand_name = _generate_ligand_name(canonical_smiles, mapping)

    ligand = Path(lig_dir) / f"{ligand_name}.pdbqt"
    output = Path(out_dir) / f"{ligand_name}.pdbqt"
    config = Path(conf_dir) / f"{ligand_name}.txt"
    log = Path(log_dir) / f"{ligand_name}.txt"

    if not ligand.exists():
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            pdbqt_string = preparator.write_pdbqt_string()
            with open(ligand, 'w') as f:
                f.write(pdbqt_string)
        except:
            print("Couldn't write as PDBQT string")
            return 1000000.0

    # Dock
    if os.path.isfile(receptor_input):
        # Create configuration files
        if not config.exists():
            conf = 'receptor = ./proteins/LPA1-7yu4.pdbqt\n' + \
                   'ligand = ' + str(ligand) + '\n' + \
                   'out = ' + str(output) + '\n' + \
                   'center_x = ' + str(center_x) + '\n' + \
                   'center_y = ' + str(center_y) + '\n' + \
                   'center_z = ' + str(center_z) + '\n' + \
                   'size_x = ' + str(size_x) + '\n' + \
                   'size_y = ' + str(size_y) + '\n' + \
                   'size_z = ' + str(size_z) + '\n' + \
                   'thread=5000'

            if seed is not None:
                conf += 'seed = ' + str(seed) + '\n'

            with open(config, 'w') as f:
                f.write(conf)

        if not output.exists():
            # Run the docking simulation
            with subprocess.Popen("./QuickVina2-GPU" +
                                  ' --config ' + str(config) +
                                  ' --log ' + str(log),
                                  # ' > /dev/null 2>&1',
                                  stdout=subprocess.PIPE,
                                  cwd=vina_cwd,
                                  shell=True, start_new_session=True) as proc:
                try:
                    out = proc.communicate(timeout=timeout_duration)
                    out_str = str(out[0]).strip().replace(r"\n", "\n")
                    if out_str is not None and "CL_OUT_OF_HOST_MEMORY" in out_str:
                        print("Docking: OUT OF GPU MEMORY ERROR")
                    if proc.returncode != 0:
                        print(f"Docking error: code {proc.returncode}")
                    #proc.wait(timeout=timeout_duration)
                except subprocess.TimeoutExpired:
                    p = psutil.Process(proc.pid)
                    p.terminate()
        else:
            print("ALREADY EXISTS!")

        # Parse the docking score
        if output.exists():
            score = 1000000.0
            with open(output, 'r') as f:
                for line in f.readlines():
                    if "REMARK VINA RESULT" in line:
                        new_score = re.findall(r'([-+]?[0-9]*\.?[0-9]+)', line)[0]
                        score = min(score, float(new_score))
                result = score

        else:
            result = 1000000.0
    else:
        raise Exception(f'Protein file: {receptor_input!r} not found!')

    mapping[ligand_name] = {
        "smiles": canonical_smiles,
        "files": {
            "ligand": str(ligand),
            "output": str(output),
            "config": str(config),
            "log": str(log)
        }
    }
    _save_mapping(map_path, mapping)

    return result

def calculateDockingScore(smi, protein_file, lig_dir, out_dir, log_dir, conf_dir, vina_cwd):
    # creating appropriate file names for ligands

    if Chem.MolFromSmiles(smi) is not None:
        canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        return dock(
            protein_file,
            canonical_smiles,
            lig_dir=lig_dir,
            out_dir=out_dir,
            log_dir=log_dir,
            conf_dir=conf_dir,
            vina_cwd=vina_cwd,
        )
    else:
        return 1000000.0

