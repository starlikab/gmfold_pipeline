try:
    import matplotlib.pyplot as plt
    import math
    from forgi.graph.bulge_graph import BulgeGraph
    import forgi.visual.mplotlib as fvm
    import RNA
    from GMfold import gmfold, gm_dot_bracket
except ImportError as e:
    missing_package = str(e).split("'")[1]
    raise ImportError(
        f"Missing required package: '{missing_package}'.\n"
        f"Please install it."
    )


def visualize_sequences(df, indices):
    num_seqs = len(indices)
    cols = 3
    rows = math.ceil(num_seqs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Flatten axes for easy iteration (handles both 1D and 2D cases)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        seq = df['Sequence'][idx]

        # Fold the sequence
        structs = gmfold(seq, l_fix=4)
        dot_bracket = gm_dot_bracket(seq, structs)

        # Create bulge graph
        bg = BulgeGraph.from_dotbracket(dot_bracket, seq)

        # Plot RNA structure
        fvm.plot_rna(bg, ax=ax, text_kwargs={"fontweight": "black"},
                     lighten=0.7, backbone_kwargs={"linewidth": 2})

        # Title with index and count
        ax.set_title(f"Sequence #{idx}\nCount: {df['Count'][idx]}", fontsize=12)
        ax.axis('off')

    # Hide unused subplots if total slots > number of sequences
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def highlight_shared_characters(df, indices):
    # Get selected sequences
    seqs = [df['Sequence'][i] for i in indices]

    # Find the minimum sequence length to safely compare
    seq_len = min(len(s) for s in seqs)

    # Identify shared characters at each position (up to shortest length)
    shared_chars = []
    for i in range(seq_len):
        chars_at_i = [seq[i] for seq in seqs]
        if all(c == chars_at_i[0] for c in chars_at_i):
            shared_chars.append(chars_at_i[0])
        else:
            shared_chars.append(None)

    # Display aligned sequences with color highlights
    for idx, seq in zip(indices, seqs):
        line = ''
        for i, c in enumerate(seq):
            if i < seq_len and shared_chars[i] == c:
                line += f"\033[92m{c}\033[0m"  # green
            else:
                line += c
        print(f"Aptamer {idx:0{4}d}, Len {len(df['Sequence'][idx])}: {line}")
