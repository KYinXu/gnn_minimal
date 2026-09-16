#!/usr/bin/env python3
"""
Geometric Feature Extraction from ESMFold PDB Files

Extracts coordinate-invariant features from peptide structures:
- pLDDT confidence scores
- Size/shape compactness metrics
- Secondary structure fractions
- Solvent accessible surface area (SASA)
- Sequence-based charge/hydrophobicity

All features are invariant to rotation/translation of coordinates.
Designed for peptides of 8-60 residues.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# BioPython imports
from Bio.PDB import PDBParser, DSSP, is_aa
from Bio.PDB.Polypeptide import is_aa as is_standard_aa
from Bio.PDB.SASA import ShrakeRupley

# Suppress BioPython warnings for cleaner output
warnings.filterwarnings('ignore', category=Warning, module='Bio')


# =============================================================================
# AMINO ACID PROPERTIES
# =============================================================================

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Charge at pH 7 (simplified)
CHARGE_PH7 = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,  # His ~10% protonated at pH 7
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

# Hydrophobic residues for SASA calculation
HYDROPHOBIC_RESIDUES = {'A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'}


# =============================================================================
# PDB PARSING UTILITIES
# =============================================================================

def parse_pdb_structure(pdb_path: str) -> Tuple[Any, List[Dict]]:
    """
    Parse PDB file and extract residue-level information.
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        structure: BioPython Structure object
        residues_info: List of dicts with residue data
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('peptide', pdb_path)
    
    residues_info = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip heteroatoms and water
                if residue.id[0] != ' ':
                    continue
                    
                res_name = residue.get_resname()
                
                # Get CA atom if present
                ca_coord = None
                if 'CA' in residue:
                    ca_coord = residue['CA'].get_coord()
                
                # Get N atom for phi/psi calculation
                n_coord = None
                if 'N' in residue:
                    n_coord = residue['N'].get_coord()
                
                # Get C atom
                c_coord = None
                if 'C' in residue:
                    c_coord = residue['C'].get_coord()
                
                # Get pLDDT from B-factor (CA atom)
                plddt = None
                if 'CA' in residue:
                    plddt = residue['CA'].get_bfactor()
                
                # Convert 3-letter to 1-letter code
                aa_map = {
                    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                }
                aa_1letter = aa_map.get(res_name, 'X')
                
                residues_info.append({
                    'residue': residue,
                    'resname': res_name,
                    'aa': aa_1letter,
                    'resid': residue.id[1],
                    'ca_coord': ca_coord,
                    'n_coord': n_coord,
                    'c_coord': c_coord,
                    'plddt': plddt
                })
    
    return structure, residues_info


# =============================================================================
# CONFIDENCE FEATURES (pLDDT)
# =============================================================================

def extract_plddt_features(residues_info: List[Dict]) -> Dict[str, float]:
    """
    Extract pLDDT confidence statistics.
    
    ESMFold stores pLDDT (0-1 scale) in the B-factor column.
    
    Returns:
        plddt_mean, plddt_std, plddt_min, plddt_max
    """
    plddt_values = [r['plddt'] for r in residues_info if r['plddt'] is not None]
    
    if not plddt_values:
        return {
            'plddt_mean': 0.0,
            'plddt_std': 0.0,
            'plddt_min': 0.0,
            'plddt_max': 0.0
        }
    
    plddt = np.array(plddt_values)
    
    return {
        'plddt_mean': float(np.mean(plddt)),
        'plddt_std': float(np.std(plddt)),
        'plddt_min': float(np.min(plddt)),
        'plddt_max': float(np.max(plddt))
    }


# =============================================================================
# SIZE/SHAPE COMPACTNESS FEATURES
# =============================================================================

def extract_compactness_features(residues_info: List[Dict]) -> Dict[str, float]:
    """
    Extract size and shape compactness metrics from CA atoms.
    
    Returns:
        radius_gyration: Rg - RMS distance from centroid
        end_to_end_distance: Distance between first and last CA
        max_pairwise_distance: Diameter - max CA-CA distance
        centroid_distance_mean: Mean distance of CAs to centroid
        centroid_distance_std: Std of distances to centroid
    """
    # Get CA coordinates
    ca_coords = np.array([r['ca_coord'] for r in residues_info 
                          if r['ca_coord'] is not None])
    
    n_residues = len(ca_coords)
    
    if n_residues < 2:
        return {
            'radius_gyration': 0.0,
            'end_to_end_distance': 0.0,
            'max_pairwise_distance': 0.0,
            'centroid_distance_mean': 0.0,
            'centroid_distance_std': 0.0
        }
    
    # Centroid
    centroid = np.mean(ca_coords, axis=0)
    
    # Distances to centroid
    centroid_distances = np.linalg.norm(ca_coords - centroid, axis=1)
    
    # Radius of gyration: sqrt(mean(r^2))
    rg = np.sqrt(np.mean(centroid_distances ** 2))
    
    # End-to-end distance
    end_to_end = np.linalg.norm(ca_coords[-1] - ca_coords[0])
    
    # Max pairwise distance (diameter)
    max_dist = 0.0
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if dist > max_dist:
                max_dist = dist
    
    return {
        'radius_gyration': float(rg),
        'end_to_end_distance': float(end_to_end),
        'max_pairwise_distance': float(max_dist),
        'centroid_distance_mean': float(np.mean(centroid_distances)),
        'centroid_distance_std': float(np.std(centroid_distances))
    }


# =============================================================================
# SECONDARY STRUCTURE FEATURES
# =============================================================================

def compute_dihedral(p1, p2, p3, p4) -> float:
    """
    Compute dihedral angle between four points in radians.
    Uses IUPAC convention for protein backbone dihedrals.
    
    For phi: p1=C(i-1), p2=N(i), p3=CA(i), p4=C(i)
    For psi: p1=N(i), p2=CA(i), p3=C(i), p4=N(i+1)
    """
    # Ensure numpy arrays
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    p4 = np.asarray(p4)
    
    # Bond vectors
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    # Normal vectors to planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0
    
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    
    # Unit vector along b2
    b2_unit = b2 / np.linalg.norm(b2)
    
    # m1 is perpendicular to n1 and b2
    m1 = np.cross(n1, b2_unit)
    
    # Dihedral angle using atan2 for correct quadrant
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    # Note: The sign convention - negate to match IUPAC
    return -np.arctan2(y, x)


def extract_secondary_structure_dssp(structure, pdb_path: str) -> Dict[str, float]:
    """
    Extract secondary structure fractions using DSSP.
    
    DSSP codes:
        H = α-helix, G = 3-10 helix, I = π-helix
        E = β-strand, B = β-bridge
        T = turn, S = bend, - = coil
    
    Returns:
        fraction_helix: (H + G + I) / total
        fraction_sheet: (E + B) / total
        fraction_coil: (T + S + -) / total
    """
    try:
        model = structure[0]
        dssp = DSSP(model, pdb_path, dssp='mkdssp')
        
        ss_counts = {'H': 0, 'G': 0, 'I': 0, 'E': 0, 'B': 0, 'T': 0, 'S': 0, '-': 0}
        
        for key in dssp:
            ss = dssp[key][2]  # Secondary structure code
            if ss in ss_counts:
                ss_counts[ss] += 1
            else:
                ss_counts['-'] += 1
        
        total = sum(ss_counts.values())
        
        if total == 0:
            return {'fraction_helix': 0.0, 'fraction_sheet': 0.0, 'fraction_coil': 0.0}
        
        helix = ss_counts['H'] + ss_counts['G'] + ss_counts['I']
        sheet = ss_counts['E'] + ss_counts['B']
        coil = ss_counts['T'] + ss_counts['S'] + ss_counts['-']
        
        return {
            'fraction_helix': float(helix / total),
            'fraction_sheet': float(sheet / total),
            'fraction_coil': float(coil / total)
        }
        
    except Exception:
        # DSSP not available, return None to trigger fallback
        return None


def extract_secondary_structure_phipsi(residues_info: List[Dict]) -> Dict[str, float]:
    """
    Fallback: Estimate secondary structure from backbone dihedrals (phi/psi).
    
    Typical Ramachandran regions:
        α-helix: phi ≈ -60° (±30°), psi ≈ -45° (±30°)
        3-10 helix: phi ≈ -50°, psi ≈ -25°
        β-sheet: phi ≈ -120° (±40°), psi ≈ +130° (±40°) [antiparallel]
                 phi ≈ -120°, psi ≈ +115° [parallel]
        
    This is approximate but doesn't require DSSP.
    """
    n_residues = len(residues_info)
    
    if n_residues < 4:
        return {
            'fraction_helix': 0.0,
            'fraction_sheet': 0.0,
            'fraction_coil': 1.0,
            'phi_psi_computed': 0
        }
    
    helix_count = 0
    sheet_count = 0
    total_count = 0
    
    # We need consecutive residues to compute phi and psi
    for i in range(1, n_residues - 1):
        prev_res = residues_info[i - 1]
        curr_res = residues_info[i]
        next_res = residues_info[i + 1]
        
        # Check if we have all required atoms
        if (prev_res['c_coord'] is None or curr_res['n_coord'] is None or
            curr_res['ca_coord'] is None or curr_res['c_coord'] is None or
            next_res['n_coord'] is None):
            continue
        
        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        phi = compute_dihedral(
            prev_res['c_coord'],
            curr_res['n_coord'],
            curr_res['ca_coord'],
            curr_res['c_coord']
        )
        
        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        psi = compute_dihedral(
            curr_res['n_coord'],
            curr_res['ca_coord'],
            curr_res['c_coord'],
            next_res['n_coord']
        )
        
        # Convert to degrees
        phi_deg = np.degrees(phi)
        psi_deg = np.degrees(psi)
        
        total_count += 1
        
        # α-helix region: phi in [-100, -30], psi in [-80, 0]
        # This covers α-helix, 3-10 helix, and π-helix regions
        if -100 <= phi_deg <= -30 and -80 <= psi_deg <= 0:
            helix_count += 1
        # β-sheet region: phi in [-180, -90] or [-180, -60], psi in [90, 180] or [-180, -120]
        # Antiparallel and parallel sheets
        elif (-180 <= phi_deg <= -60) and (90 <= psi_deg <= 180):
            sheet_count += 1
        elif (-180 <= phi_deg <= -60) and (-180 <= psi_deg <= -120):
            sheet_count += 1
        # Also check for extended conformation with positive psi
        elif (-160 <= phi_deg <= -60) and (100 <= psi_deg <= 180):
            sheet_count += 1
    
    if total_count == 0:
        return {
            'fraction_helix': 0.0,
            'fraction_sheet': 0.0,
            'fraction_coil': 1.0,
            'phi_psi_computed': 0
        }
    
    frac_helix = helix_count / total_count
    frac_sheet = sheet_count / total_count
    frac_coil = 1.0 - frac_helix - frac_sheet
    
    return {
        'fraction_helix': float(frac_helix),
        'fraction_sheet': float(frac_sheet),
        'fraction_coil': float(max(0.0, frac_coil)),
        'phi_psi_computed': total_count
    }


def extract_secondary_structure(structure, pdb_path: str, residues_info: List[Dict]) -> Dict[str, float]:
    """
    Extract secondary structure with DSSP, falling back to phi/psi if unavailable.
    
    Returns:
        fraction_helix: Fraction of residues in helix
        fraction_sheet: Fraction of residues in sheet
        fraction_coil: Fraction of residues in coil
        ss_method: 'dssp' or 'phi_psi'
        ss_residues_computed: Number of residues with SS assignment
    """
    # Try DSSP first
    dssp_result = extract_secondary_structure_dssp(structure, pdb_path)
    
    if dssp_result is not None:
        dssp_result['ss_method'] = 'dssp'
        dssp_result['ss_residues_computed'] = len(residues_info)
        return dssp_result
    
    # Fallback to phi/psi estimation
    phipsi_result = extract_secondary_structure_phipsi(residues_info)
    phipsi_result['ss_method'] = 'phi_psi'
    phipsi_result['ss_residues_computed'] = phipsi_result.pop('phi_psi_computed', 0)
    return phipsi_result


# =============================================================================
# SOLVENT ACCESSIBLE SURFACE AREA (SASA)
# =============================================================================

def extract_sasa_features(structure) -> Dict[str, float]:
    """
    Extract SASA features using Shrake-Rupley algorithm.
    
    Returns:
        total_sasa: Total solvent accessible surface area (Å²)
        hydrophobic_sasa: SASA of hydrophobic residues
        fraction_hydrophobic_sasa: hydrophobic_sasa / total_sasa
    """
    try:
        # Compute SASA using BioPython's Shrake-Rupley
        sr = ShrakeRupley()
        sr.compute(structure, level="R")  # Residue level
        
        total_sasa = 0.0
        hydrophobic_sasa = 0.0
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip heteroatoms
                    if residue.id[0] != ' ':
                        continue
                    
                    res_sasa = residue.sasa if hasattr(residue, 'sasa') else 0.0
                    total_sasa += res_sasa
                    
                    # Check if hydrophobic
                    resname = residue.get_resname()
                    aa_map = {
                        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                    }
                    aa = aa_map.get(resname, 'X')
                    
                    if aa in HYDROPHOBIC_RESIDUES:
                        hydrophobic_sasa += res_sasa
        
        fraction_hydrophobic = hydrophobic_sasa / total_sasa if total_sasa > 0 else 0.0
        
        return {
            'total_sasa': float(total_sasa),
            'hydrophobic_sasa': float(hydrophobic_sasa),
            'fraction_hydrophobic_sasa': float(fraction_hydrophobic)
        }
        
    except Exception as e:
        # SASA computation failed
        return {
            'total_sasa': 0.0,
            'hydrophobic_sasa': 0.0,
            'fraction_hydrophobic_sasa': 0.0
        }


# =============================================================================
# SEQUENCE-BASED FEATURES (Charge/Hydrophobicity)
# =============================================================================

def extract_sequence_features(residues_info: List[Dict], sequence: Optional[str] = None) -> Dict[str, float]:
    """
    Extract sequence-based physicochemical features.
    
    Returns:
        length: Number of residues
        net_charge: Sum of charges at pH 7
        mean_hydrophobicity: Mean Kyte-Doolittle score
        hydrophobic_moment: Amphipathicity measure (assumes helical periodicity)
    """
    # Get amino acids from structure or provided sequence
    if sequence:
        aa_list = list(sequence.upper())
    else:
        aa_list = [r['aa'] for r in residues_info if r['aa'] != 'X']
    
    n = len(aa_list)
    
    if n == 0:
        return {
            'length': 0,
            'net_charge': 0.0,
            'mean_hydrophobicity': 0.0,
            'hydrophobic_moment': 0.0
        }
    
    # Net charge at pH 7
    # +1 for N-terminus, -1 for C-terminus
    net_charge = 1.0 - 1.0  # N and C termini cancel
    for aa in aa_list:
        net_charge += CHARGE_PH7.get(aa, 0)
    
    # Mean hydrophobicity
    hydro_values = [HYDROPHOBICITY.get(aa, 0) for aa in aa_list]
    mean_hydro = np.mean(hydro_values)
    
    # Hydrophobic moment (assuming α-helix, 100° per residue)
    # μH = sqrt((Σ Hi*sin(θi))² + (Σ Hi*cos(θi))²) / n
    angle_per_residue = np.radians(100)  # α-helix periodicity
    
    sin_sum = 0.0
    cos_sum = 0.0
    for i, aa in enumerate(aa_list):
        h = HYDROPHOBICITY.get(aa, 0)
        angle = i * angle_per_residue
        sin_sum += h * np.sin(angle)
        cos_sum += h * np.cos(angle)
    
    hydrophobic_moment = np.sqrt(sin_sum**2 + cos_sum**2) / n
    
    return {
        'length': n,
        'net_charge': float(net_charge),
        'mean_hydrophobicity': float(mean_hydro),
        'hydrophobic_moment': float(hydrophobic_moment)
    }


# =============================================================================
# BACKBONE CURVATURE FEATURES (BONUS)
# =============================================================================

def extract_curvature_features(residues_info: List[Dict]) -> Dict[str, float]:
    """
    Extract discrete backbone curvature features from CA trace.
    
    Uses Menger curvature: κ = 4*Area / (|p1-p2|*|p2-p3|*|p3-p1|)
    
    Returns:
        curvature_mean, curvature_std, curvature_max
        torsion_mean, torsion_std
    """
    ca_coords = np.array([r['ca_coord'] for r in residues_info 
                          if r['ca_coord'] is not None])
    
    n = len(ca_coords)
    
    if n < 4:
        return {
            'curvature_mean': 0.0,
            'curvature_std': 0.0,
            'curvature_max': 0.0,
            'torsion_mean': 0.0,
            'torsion_std': 0.0
        }
    
    # Compute curvatures (need 3 consecutive points)
    curvatures = []
    for i in range(1, n - 1):
        p1, p2, p3 = ca_coords[i-1], ca_coords[i], ca_coords[i+1]
        
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        
        # Area via cross product
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        
        if a * b * c > 1e-10:
            kappa = 4 * area / (a * b * c)
        else:
            kappa = 0.0
        
        curvatures.append(kappa)
    
    # Compute torsions (need 4 consecutive points)
    torsions = []
    for i in range(1, n - 2):
        p1, p2, p3, p4 = ca_coords[i-1], ca_coords[i], ca_coords[i+1], ca_coords[i+2]
        tau = compute_dihedral(p1, p2, p3, p4)
        torsions.append(abs(tau))
    
    curvatures = np.array(curvatures)
    torsions = np.array(torsions) if torsions else np.array([0.0])
    
    return {
        'curvature_mean': float(np.mean(curvatures)),
        'curvature_std': float(np.std(curvatures)),
        'curvature_max': float(np.max(curvatures)),
        'torsion_mean': float(np.mean(torsions)),
        'torsion_std': float(np.std(torsions))
    }


# =============================================================================
# MAIN FEATURE EXTRACTION FUNCTION
# =============================================================================

def extract_all_features(
    pdb_path: str,
    peptide_id: str = None,
    sequence: str = None,
    svm_sigma: float = None,
    svm_prob: float = None,
    qsar_descriptors: List[float] = None
) -> Dict[str, float]:
    """
    Extract all geometric features from a PDB file.
    
    Args:
        pdb_path: Path to ESMFold PDB file
        peptide_id: Optional identifier
        sequence: Optional amino acid sequence
        svm_sigma: Optional SVM sigma (distance to margin)
        svm_prob: Optional SVM P(+) probability
        qsar_descriptors: Optional list of 12 QSAR descriptors
        
    Returns:
        Dictionary of all features
    """
    # Parse structure
    structure, residues_info = parse_pdb_structure(pdb_path)
    
    if len(residues_info) == 0:
        raise ValueError(f"No residues found in {pdb_path}")
    
    # Extract all feature groups
    features = {}
    
    # 1. pLDDT confidence (4 features)
    features.update(extract_plddt_features(residues_info))
    
    # 2. Compactness (5 features)
    features.update(extract_compactness_features(residues_info))
    
    # 3. Secondary structure (3 features + method flag)
    ss_features = extract_secondary_structure(structure, pdb_path, residues_info)
    features.update(ss_features)
    
    # 4. SASA (3 features)
    features.update(extract_sasa_features(structure))
    
    # 5. Sequence features (4 features)
    features.update(extract_sequence_features(residues_info, sequence))
    
    # 6. Curvature (5 features)
    features.update(extract_curvature_features(residues_info))
    
    # Optional: SVM outputs (2 features)
    if svm_sigma is not None:
        features['svm_sigma'] = float(svm_sigma)
    if svm_prob is not None:
        features['svm_prob_positive'] = float(svm_prob)
    
    # Optional: QSAR descriptors (12 features)
    if qsar_descriptors is not None and len(qsar_descriptors) == 12:
        qsar_names = [
            'qsar_netCharge', 'qsar_FC', 'qsar_LW', 'qsar_DP', 'qsar_NK', 'qsar_AE',
            'qsar_pcMK', 'qsar_SolventAccessibility', 'qsar_tau2', 'qsar_tau4',
            'qsar_QSO50', 'qsar_QSO29'
        ]
        for name, val in zip(qsar_names, qsar_descriptors):
            features[name] = float(val)
    
    # Add peptide ID if provided
    if peptide_id is not None:
        features['peptide_id'] = peptide_id
    
    # Add sequence if provided
    if sequence is not None:
        features['sequence'] = sequence
    
    return features


def get_feature_names(include_optional: bool = True) -> List[str]:
    """
    Get ordered list of feature names.
    
    Args:
        include_optional: Include SVM and QSAR feature names
        
    Returns:
        List of feature names in order
    """
    names = [
        # pLDDT (4)
        'plddt_mean', 'plddt_std', 'plddt_min', 'plddt_max',
        
        # Compactness (5)
        'radius_gyration', 'end_to_end_distance', 'max_pairwise_distance',
        'centroid_distance_mean', 'centroid_distance_std',
        
        # Secondary structure (4)
        'fraction_helix', 'fraction_sheet', 'fraction_coil', 'ss_method',
        
        # SASA (3)
        'total_sasa', 'hydrophobic_sasa', 'fraction_hydrophobic_sasa',
        
        # Sequence (4)
        'length', 'net_charge', 'mean_hydrophobicity', 'hydrophobic_moment',
        
        # Curvature (5)
        'curvature_mean', 'curvature_std', 'curvature_max',
        'torsion_mean', 'torsion_std',
    ]
    
    if include_optional:
        names.extend([
            # SVM (2)
            'svm_sigma', 'svm_prob_positive',
            
            # QSAR (12)
            'qsar_netCharge', 'qsar_FC', 'qsar_LW', 'qsar_DP', 'qsar_NK', 'qsar_AE',
            'qsar_pcMK', 'qsar_SolventAccessibility', 'qsar_tau2', 'qsar_tau4',
            'qsar_QSO50', 'qsar_QSO29'
        ])
    
    return names
