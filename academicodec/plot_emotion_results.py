from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

REF_FILE_NAME = "emotion_labels_ref.json"
GEN_FILE_NAME = "emotion_labels_gen.json"

def plot_emotion_results(results_dir: str, 
                         show_arrows: bool = True, 
                         aro_filter_thr: float = 0.0, 
                         val_filter_thr: float = 0.0):

    ref_path = os.path.join(results_dir, REF_FILE_NAME)
    gen_path = os.path.join(results_dir, GEN_FILE_NAME)

    with open(ref_path, "r") as f:
        ref_data = json.load(f)
    with open(gen_path, "r") as f:
        gen_data = json.load(f)
        
    # Prepare data for plotting
    files = list(set(ref_data.keys()) & set(gen_data.keys()))
    ref_arousal = [ref_data[f]["arousal"] for f in files]
    ref_valence = [ref_data[f]["valence"] for f in files]
    gen_arousal = [gen_data[f]["arousal"] for f in files]
    gen_valence = [gen_data[f]["valence"] for f in files]
    ref_probs = [ref_data[f]['emotions']['Neutral'] for f in files]
    gen_probs = [gen_data[f]['emotions']['Neutral'] for f in files]

    # Filter out neutral samples if thresholds are set
    if aro_filter_thr > 0.0 or val_filter_thr > 0.0:
        aro_mask = [abs(aro - 0.5) > aro_filter_thr for aro in ref_arousal]
        val_mask = [abs(val - 0.5) > val_filter_thr for val in ref_valence]
        combined_mask = [aro and val for aro, val in zip(aro_mask, val_mask)]
        ref_arousal = [ref_arousal[i] for i in range(len(ref_arousal)) if combined_mask[i]]
        ref_valence = [ref_valence[i] for i in range(len(ref_valence)) if combined_mask[i]]
        gen_arousal = [gen_arousal[i] for i in range(len(gen_arousal)) if combined_mask[i]]
        gen_valence = [gen_valence[i] for i in range(len(gen_valence)) if combined_mask[i]]
        files = [files[i] for i in range(len(files)) if combined_mask[i]]
        print(f"Filtered to {len(files)} samples after applying thresholds.")

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=ref_valence, y=ref_arousal, color='blue', label='Reference')
    sns.scatterplot(x=gen_valence, y=gen_arousal, color='red', label='Generated')

    # Draw arrows between ref and gen points for each file
    if show_arrows:
        for i, f in enumerate(files):
            plt.arrow(ref_valence[i], ref_arousal[i],
                      gen_valence[i] - ref_valence[i],
                      gen_arousal[i] - ref_arousal[i],
                      color='gray', alpha=0.5, length_includes_head=True, head_width=0.01)
        
    plt.axhline(0.5, color='red', linestyle='--', linewidth=1)
    plt.axvline(0.5, color='red', linestyle='--', linewidth=1)

    plt.xlabel("Valence", fontsize=18)
    plt.ylabel("Arousal", fontsize=18)
    plt.legend(fontsize=16)
    plt.title("Emotion Results: Reference vs Generated", fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "emotion_results_plot.png"))
    
    # Compute metrics
    ref_arousal_deltas = [abs(aro - 0.5) for aro in ref_arousal]
    ref_valence_deltas = [abs(val - 0.5) for val in ref_valence]
    gen_arousal_deltas = [abs(aro - 0.5) for aro in gen_arousal]
    gen_valence_deltas = [abs(val - 0.5) for val in gen_valence]

    arousal_shifts = [ref - gen for gen, ref in zip(gen_arousal_deltas, ref_arousal_deltas)]
    valence_shifts = [ref - gen for gen, ref in zip(gen_valence_deltas, ref_valence_deltas)]
    total_shift = [a + v for a, v in zip(arousal_shifts, valence_shifts)]

    prob_shifts = [gen - ref for gen, ref in zip(gen_probs, ref_probs)]

    # Save shifts to JSON
    shifts_dict = {}
    for i, f in enumerate(files):
        shifts_dict[f] = {
            "arousal_shift": arousal_shifts[i],
            "valence_shift": valence_shifts[i],
            "total_shift": total_shift[i],
            "neutral_prob_shift": prob_shifts[i]
        }
    with open(os.path.join(results_dir, "emotion_shifts.json"), "w") as f:
        json.dump(shifts_dict, f, indent=2)

    # Plot histograms of shifts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(arousal_shifts, bins=30, ax=axes[0], color='blue')
    axes[0].set_title("Arousal Shifts", fontsize=16)
    axes[0].set_xlabel("Shift", fontsize=14)
    axes[0].set_ylabel("Count", fontsize=14)

    sns.histplot(valence_shifts, bins=30, ax=axes[1], color='green')
    axes[1].set_title("Valence Shifts", fontsize=16)
    axes[1].set_xlabel("Shift", fontsize=14)
    axes[1].set_ylabel("Count", fontsize=14)

    sns.histplot(prob_shifts, bins=30, ax=axes[2], color='purple')
    axes[2].set_title("Neutral Prob. Shifts", fontsize=16)
    axes[2].set_xlabel("Shift", fontsize=14)
    axes[2].set_ylabel("Count", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "emotion_shifts_histograms.png"))
    plt.close(fig)
        
if __name__ == "__main__":
    
    import argparse 
    
    parser = argparse.ArgumentParser(description="Plot emotion results from reference and generated data.")
    parser.add_argument("results_dir", type=str, help="Directory containing the emotion results files.")
    parser.add_argument("--show-arrows", action="store_true", help="Whether to show arrows between reference and generated points.")

    args = parser.parse_args()
    plot_emotion_results(args.results_dir, show_arrows=args.show_arrows, aro_filter_thr=0.1, val_filter_thr=0.1)

