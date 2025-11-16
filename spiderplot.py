# --- EXPERIMENT: EVALUATION METHOD SPIDER PLOT ---

import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings

print("\n\n--- Starting Experiment: Evaluation Method Spider Plot ---")

# Suppress warnings for zero-division in metrics (e.g., if a class is never predicted)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Get Metrics for Segment-Level ---
# These variables (y_true, y_pred_classes, y_pred) should already be defined
# from the segment-level evaluation in your script.
print("Calculating Segment-Level metrics...")
try:
    seg_accuracy = accuracy_score(y_true, y_pred_classes)
    seg_f1 = f1_score(y_true, y_pred_classes, pos_label=1, zero_division=0)
    seg_precision = precision_score(y_true, y_pred_classes, pos_label=1, zero_division=0)
    seg_recall = recall_score(y_true, y_pred_classes, pos_label=1, zero_division=0)
    seg_auc = roc_auc_score(y_true, y_pred[:, 1]) # Use probabilities for AUC
    
    segment_metrics = [seg_accuracy, seg_f1, seg_precision, seg_recall, seg_auc]
    print(f"Segment-Level Metrics (AD, pos_label=1): F1={seg_f1:.4f}, AUC={seg_auc:.4f}")

except NameError:
    print("ERROR: Could not find variables 'y_true', 'y_pred_classes', or 'y_pred'.")
    print("Please ensure the segment-level evaluation runs before this experiment.")
    segment_metrics = [0, 0, 0, 0, 0] # Zeros to prevent crash
except Exception as e:
    print(f"An error occurred during segment-level metric calculation: {e}")
    segment_metrics = [0, 0, 0, 0, 0]

# --- 2. Get Metrics for Patient-Level (Soft Vote) ---
print("Calculating Patient-Level (Soft Vote) metrics...")
try:
    yt_soft, yp_soft, prob_soft = patient_level_ensemble_2(
        fusion_model, bilstm_model, cnn_time_model, cnn_freq_model, test_meta, voting='soft'
    )
    
    soft_accuracy = accuracy_score(yt_soft, yp_soft)
    soft_f1 = f1_score(yt_soft, yp_soft, pos_label=1, zero_division=0)
    soft_precision = precision_score(yt_soft, yp_soft, pos_label=1, zero_division=0)
    soft_recall = recall_score(yt_soft, yp_soft, pos_label=1, zero_division=0)
    soft_auc = roc_auc_score(yt_soft, prob_soft)
    
    soft_vote_metrics = [soft_accuracy, soft_f1, soft_precision, soft_recall, soft_auc]
    print(f"Patient-Level Soft Vote Metrics (AD, pos_label=1): F1={soft_f1:.4f}, AUC={soft_auc:.4f}")
except Exception as e:
    print(f"An error occurred during soft vote metric calculation: {e}")
    soft_vote_metrics = [0, 0, 0, 0, 0]

# --- 3. Get Metrics for Patient-Level (Hard Vote) ---
print("Calculating Patient-Level (Hard Vote) metrics...")
try:
    yt_hard, yp_hard, prob_hard = patient_level_ensemble_2(
        fusion_model, bilstm_model, cnn_time_model, cnn_freq_model, test_meta, voting='hard'
    )
    
    hard_accuracy = accuracy_score(yt_hard, yp_hard)
    hard_f1 = f1_score(yt_hard, yp_hard, pos_label=1, zero_division=0)
    hard_precision = precision_score(yt_hard, yp_hard, pos_label=1, zero_division=0)
    hard_recall = recall_score(yt_hard, yp_hard, pos_label=1, zero_division=0)
    hard_auc = roc_auc_score(yt_hard, prob_hard) # Note: Using mean probabilities for hard vote AUC
    
    hard_vote_metrics = [hard_accuracy, hard_f1, hard_precision, hard_recall, hard_auc]
    print(f"Patient-Level Hard Vote Metrics (AD, pos_label=1): F1={hard_f1:.4f}, AUC={hard_auc:.4f}")
except Exception as e:
    print(f"An error occurred during hard vote metric calculation: {e}")
    hard_vote_metrics = [0, 0, 0, 0, 0]


# --- 4. Plot the Spider Chart ---
print("Generating spider plot...")

categories = ['Accuracy', 'F1-Score (AD)', 'Precision (AD)', 'Recall (AD)', 'ROC AUC']

fig = go.Figure()

# --- MODIFIED FOR CLARITY ---
# Add Segment-Level Trace (Thick line, no fill)
fig.add_trace(go.Scatterpolar(
    r=segment_metrics,
    theta=categories,
    # fill='toself', # <-- REMOVED
    name='Segment-Level',
    line=dict(color='gray', width=3) # <-- MADE LINE THICKER
))

# Add Hard Vote Trace (Thick line, no fill)
fig.add_trace(go.Scatterpolar(
    r=hard_vote_metrics,
    theta=categories,
    # fill='toself', # <-- REMOVED
    name='Patient-Level (Hard Vote)',
    line=dict(color='mediumblue', width=3, dash='dash') # <-- ADDED 'dash'
))

# Add Soft Vote Trace (Thick line, no fill)
fig.add_trace(go.Scatterpolar(
    r=soft_vote_metrics,
    theta=categories,
    # fill='toself', # <-- REMOVED
    name='Patient-Level (Soft Vote)',
    line=dict(color='crimson', width=3) # <-- MADE LINE THICKER
))
# --- END MODIFICATION ---

# --- 5. Format the plot ---
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0.7, 1.0]  # <-- ZOOMED IN on the 70-100% range
        )
    ),
    showlegend=True,
    title={
        'text': "Comparison of Evaluation Methods",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    font=dict(size=14)
)

# --- 6. Save Files (with error handling) ---
# Save as an interactive HTML file
fig.write_html("evaluation_method_spider_plot.html")

try:
    # Save as a static image for the paper
    fig.write_image("evaluation_method_spider_plot.png", scale=2)
    print("Spider plot saved as 'evaluation_method_spider_plot.html' and '.png'")
except ValueError as e:
    print("\n--- KALEIDO ERROR ---")
    print(f"Successfully saved 'evaluation_method_spider_plot.html'.")
    print(f"COULD NOT SAVE .PNG. Error: {e}")
    print("Please install kaleido to save static images: pip install --upgrade kaleido")

print("--- Spider Plot Experiment Complete ---")
# Restore warnings
warnings.filterwarnings('default', category=UserWarning)
