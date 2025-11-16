# --- EXPERIMENT 5: t-SNE FEATURE SPACE VISUALIZATION ---

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

print("\n\n--- Starting Experiment 5: t-SNE Feature Space Visualization ---")

# 1. Concatenate the features from the three base models
print("Concatenating test features...")
all_test_features = np.concatenate(
    [X_bilstm_test_feat, X_cnn_time_test_feat, X_cnn_freq_test_feat], 
    axis=1
)
print(f"Total feature shape: {all_test_features.shape}")
# Note: y_true (segment-level labels) is already defined from the previous step

# 2. Run t-SNE to reduce features to 2 dimensions
# This can take a minute or two, so we print a message
print(f"Running t-SNE (n_components=2, random_state={SEED}). This may take a moment...")
tsne = TSNE(
    n_components=2, 
    random_state=SEED, 
    perplexity=30,  # A good default value
    n_iter=1000     # Number of iterations
)
features_2d = tsne.fit_transform(all_test_features)
print("t-SNE complete.")

# 3. Create a DataFrame for easy plotting with Seaborn
# Map numeric labels back to names for a clearer legend
# From your process_data: label_map = {'AD': 1, 'Healthy': 0}
target_names = {0: 'Healthy', 1: 'AD'}
y_labels = [target_names[label] for label in y_true]

df_tsne = pd.DataFrame({
    't-SNE Component 1': features_2d[:, 0],
    't-SNE Component 2': features_2d[:, 1],
    'Label': y_labels
})

# 4. Create the scatter plot
print("Generating scatter plot...")
plt.figure(figsize=(14, 10))
sns.set(style="whitegrid", font_scale=1.1)

# Use seaborn for a clean plot with an automatic legend
sns.scatterplot(
    x='t-SNE Component 1',
    y='t-SNE Component 2',
    hue='Label',  # Color points by their true label
    palette={'Healthy': 'mediumblue', 'AD': 'crimson'}, # Use consistent colors
    data=df_tsne,
    alpha=0.7,    # Set transparency to see overlapping points
    s=50          # Set marker size
)

plt.title('t-SNE Visualization of Extracted Test Features (Segment-Level)', fontsize=18, fontweight='bold')
plt.xlabel('t-SNE Component 1', fontsize=14)
plt.ylabel('t-SNE Component 2', fontsize=14)
plt.legend(title='True Label', fontsize=12, title_fontsize=14)
plt.tight_layout()

# 5. Save the plot
plt.savefig("tsne_feature_visualization.png", dpi=300)
print("Plot saved as 'tsne_feature_visualization.png'")
print("--- Experiment 5 Complete ---")
