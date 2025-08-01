try:
    from collections import defaultdict
    from GMfold import gmfold, gm_dot_bracket
    import itertools
    import math
    import numpy as np
    import pandas as pd
except ImportError as e:
    missing_package = str(e).split("'")[1]
    raise ImportError(
        f"Missing required package: '{missing_package}'.\n"
        f"Please install it."
    )


def get_part_before_colon(input_string):

    if ':' in input_string:
        return input_string.split(':')[0]
    else:
        return input_string

# --- Step 1: Extract energy values grouped by face type ---
def get_face_energy_bins(df, n_bins=3):
    face_energy_dict = defaultdict(list)
    for i in range(len(df)):
        for j in range(len(df['faces'][i])):
            face = get_part_before_colon(df['faces'][i][j])
            energy = df['energy_faces'][i][j]
            if not isinstance(energy, str):  # ensure numeric
                face_energy_dict[face].append(energy)

    # Create quantile bins per face type
    face_to_bins = {}
    for face, energies in face_energy_dict.items():
        series = pd.Series(energies)
        try:
            _, bin_edges = pd.qcut(series, q=n_bins, retbins=True, duplicates='drop')
            face_to_bins[face] = bin_edges
        except ValueError:
            # Fallback: simple 2-bin rule
            face_to_bins[face] = [-np.inf, 0, np.inf]
    
    return face_to_bins

def categorize_energy_by_face(face, energy, face_to_bins):
    if face not in face_to_bins or not np.isfinite(energy):
        return "bin_unknown"
    bins = face_to_bins[face]
    bin_idx = pd.cut([energy], bins=bins, labels=False, include_lowest=True)[0]
    return f"{face}_bin{bin_idx}"

def Bof_in_df(df):
    face_to_bins = get_face_energy_bins(df, n_bins=5)

    words = []
    for i in range(len(df)):
        for j in range(len(df['faces'][i])):
            face = get_part_before_colon(df['faces'][i][j])
            energy = df['energy_faces'][i][j]
            label = categorize_energy_by_face(face, energy, face_to_bins)
            word_pair = (face, label)
            if word_pair not in words:
                words.append(word_pair)

    vector_list = []
    for i in range(len(df)):
        binary_array = np.zeros(len(words), dtype=int)
        for j in range(len(df['faces'][i])):
            face = get_part_before_colon(df['faces'][i][j])
            energy = df['energy_faces'][i][j]
            label = categorize_energy_by_face(face, energy, face_to_bins)
            binary_array[words.index((face, label))] += 1
        vector_list.append(binary_array)

    df['binary_array'] = vector_list

    X_raw = np.vstack(vector_list)
    return X_raw, words, face_to_bins

    #Function to compute Motzkin path descriptors
def compute_descriptor_motzkin(input_string):
    descriptor = []
    sum = 0
    for count, char in enumerate(input_string):
        if char == '(':
            sum +=1
            descriptor.append(sum)
        elif char == ')':
            sum -=1
            descriptor.append(sum)
        elif char == '.':
            descriptor.append(sum)
    return descriptor

# Assigns each face with the starting and ending indices
def face_assignment(motzkin_path, structs):
    m_path = compute_descriptor_motzkin(motzkin_path)

    # Contains the starting and ending indices of every face
    indices = []
    descriptions = []
    for s in structs:
        i, j = s.ij[0]
        idx = [k for k in range(i, j) if m_path[k] == m_path[i]]
        indices.append(idx)
        if j == len(m_path)-1:
            indices[0].append(j)

        faces = s.desc.split("\n")
        for face in faces:
            descriptions.append((face[0:face.index(":")], s.e))

    return indices, descriptions

# indices contains the starting and ending indices of every face
# descriptions contains the face/energy pair for every face

def seq_face_generator(seq, face_to_index, face_to_bins):
    structs = gmfold(seq)
    m_path = gm_dot_bracket(seq, structs)
    indices, descriptions = face_assignment(m_path, structs)

    seq_face = [0] * len(seq)  # default index if no face applies

    for face_desc, energy in descriptions:
        # Get bin label
        if face_desc in face_to_bins and np.isfinite(energy):
            bins = face_to_bins[face_desc]
            bin_idx = np.digitize([energy], bins[1:-1], right=True)[0]
            word = (face_desc, f"{face_desc}_bin{bin_idx}")
        else:
            word = (face_desc, "bin_unknown")

        # Get corresponding index
        idx = face_to_index.get(word, 0)

        # Assign to positions
        for position in indices[descriptions.index((face_desc, energy))]:
            seq_face[position] = idx

    return seq_face


def get_proportions(kmer, seq, seq_face, words):
  face_proportions = [0.0] * len(words)
  kmer_indices = [[]]
  for i in range(len(seq)-3):
    if kmer == seq[i:i+4]:
      kmer_indices.append(list(range(i, i+4)))

  for indxs in kmer_indices:
    for indx in indxs:
      face = seq_face[indx]
      face_proportions[face] += 0.25
  return face_proportions


def apta_proportions(k, seq, seq_face, words):
    all_vectors = []

    # Define alphabet and structure types
    nucleotides = 'ACGT'
    all_kmers = [''.join(p) for p in itertools.product(nucleotides, repeat=k)]

    for kmer_str in all_kmers:
        vec = get_proportions(kmer_str, seq, seq_face, words)
        all_vectors.append(vec)

    apta_vec = np.concatenate(all_vectors)
    return apta_vec


def get_proportion_matrix(df):
    Xraw, words, face_to_bins = Bof_in_df(df)
    face_to_index = {face: idx for idx, face in enumerate(words)}

    nucleotides = 'ACGT'
    kmer_list = [''.join(p) for p in itertools.product(nucleotides, repeat=4)]
    feature_names = [f"{f}|{k}" for f, k in itertools.product(words, kmer_list)]

    matrix = np.zeros((len(df), len(feature_names)))

    for i, seq in enumerate(df['Sequence']):
        seq_face = seq_face_generator(seq, face_to_index, face_to_bins)
        proportion = apta_proportions(4, seq, seq_face, words)
        matrix[i, :] = proportion  # assign the vector to the row

    return matrix
