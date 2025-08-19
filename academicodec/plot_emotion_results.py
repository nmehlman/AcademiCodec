import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

REF_FILE_NAME = "emotion_labels_ref.json"
GEN_FILE_NAME = "emotion_labels_gen.json"

def plot_emotion_results(results_dir: str):
    
    ref_path = os.path.join(results_dir, REF_FILE_NAME)
    gen_path = os.path.join(results_dir, GEN_FILE_NAME)

    with open(ref_path, "r") as f:
        ref_data = json.load(f)
    with open(gen_path, "r") as f:
        gen_data = json.load(f)
        
    # Prepare data for plotting
    files = set(ref_data.keys()) & set(gen_data.keys())
    ref_arousal = [ref_data[f]["arousal"] for f in files]
    ref_valence = [ref_data[f]["valence"] for f in files]
    gen_arousal = [gen_data[f]["arousal"] for f in files]
    gen_valence = [gen_data[f]["valence"] for f in files]

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=ref_valence, y=ref_arousal, color='blue', label='Reference')
    sns.scatterplot(x=gen_valence, y=gen_arousal, color='red', label='Generated')

    # Draw arrows between ref and gen points for each file
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
    
if __name__ == "__main__":
    
    import argparse 
    
    parser = argparse.ArgumentParser(description="Plot emotion results from reference and generated data.")
    parser.add_argument("results_dir", type=str, help="Directory containing the emotion results files.")
    
    args = parser.parse_args()
    plot_emotion_results(args.results_dir)

