import numpy as np
from GMfold import gmfold, gm_dot_bracket
import itertools
import pandas as pd

def get_part_before_colon(input_string):

    if ':' in input_string:
        return input_string.split(':')[0]
    else:
        return input_string
# Function to generate BoF descriptors

def BoF_in_df(df):
    words = []
    for i in range(len(df)):
        for j in range(len(df['faces'][i])):
            word_pair = (get_part_before_colon(df['faces'][i][j]), df['energy_faces'][i][j])
            if word_pair not in words:
                words.append(word_pair)
    vecotr_list = []
    for i in range(len(df)):
        binary_array = np.zeros(len(words), dtype=int)
        for j in range(len(df['faces'][i])):
            binary_array[words.index((get_part_before_colon(df['faces'][i][j]), df['energy_faces'][i][j]))] += 1
        vecotr_list.append(binary_array)
    # Apply function to each row
    df['binary_array'] = vecotr_list
    X_raw = np.zeros((len(df), len(words)), dtype=int)
    for count, desc in enumerate(df['binary_array']):
        X_raw[count, :] = desc

    return X_raw, words

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

def seq_face_generator(seq, face_to_index):

    structs = gmfold(seq)
    m_path = gm_dot_bracket(seq, structs)
    indices, descriptions = face_assignment(m_path, structs)

    # Create the list with the corresponding faces
    seq_face = [i for i in range(len(seq))]

    # Loop over every unique face in the sequence
    for i in range(len(descriptions)):
        # Loop over every set of indices of the faces
        for idx in indices[i]:
            # Assign this index to the corresponding dimension in feature_names
            seq_face[idx] = face_to_index.get(descriptions[i], 122)  # should i explain myself
    return seq_face


def get_proportions(kmer, seq, seq_face, bof_words_len):
  face_proportions = [0.0] * bof_words_len
  kmer_indices = [[]]
  for i in range(len(seq)-3):
    if kmer == seq[i:i+4]:
      kmer_indices.append(list(range(i, i+4)))

  for indxs in kmer_indices:
    for indx in indxs:
      face = seq_face[indx]
      face_proportions[face] += 0.25
  return face_proportions


def apta_proportions(seq, seq_face, bof_words_len):
    all_vectors = []

    # Define alphabet and structure types
    nucleotides = 'ACGT'
    all_kmers = [''.join(p) for p in itertools.product(nucleotides, repeat=4)]

    for kmer_str in all_kmers:
        vec = get_proportions(kmer_str, seq, seq_face, bof_words_len)
        all_vectors.append(vec)

    apta_vec = np.concatenate(all_vectors)
    return apta_vec


def get_proportion_matrix(df):
    Xraw, words = BoF_in_df(df)
    bof_words_len = len(words)

    face_to_index = {face: idx for idx, face in enumerate(words)}

    nucleotides = 'ACGT'
    kmer_list = [''.join(p) for p in itertools.product(nucleotides, repeat=4)]

    feature_names = [f"{f}|{k}" for f, k in itertools.product(words, kmer_list)]

    matrix = np.zeros((len(df), len(feature_names)))

    for i, seq in enumerate(df['Sequence']):
        seq_face = seq_face_generator(seq, face_to_index)
        proportion = apta_proportions(seq, seq_face, bof_words_len)
        matrix[i, :] = proportion  # assign the vector to the row

    return matrix

