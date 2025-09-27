# SlotFormer Feature Space Analysis

This directory contains a modular experiment designed to analyze the feature space of a trained SlotFormer model. The primary goal is to prove and visualize the model's **hierarchical learning capabilities**.

---

### 1. Experiment Goal

This analysis seeks to answer the fundamental question: **"Does the SlotFormer learn to represent scenes in a hierarchical, progressive manner?"**

If the model is successful, we expect to see that using more of its learned "slots" to reconstruct an image results in a progressively better and more detailed reconstruction. The first few slots should capture the main, low-frequency components of a scene, while subsequent slots should add finer details.

### 2. Methodology

The analysis pipeline, executed by `run_feature_analysis.py`, follows these steps:

1.  **Load Model**: A pre-trained `SlotFormer` checkpoint is loaded.
2.  **Build Diverse Dataset**: The experiment uses a combination of:
    *   **Real Images**: The validation set from the Imagenette dataset.
    *   **Synthetic Images**: A programmatically generated set of diverse images, including simple geometric shapes (circles, squares), gradients, and patterns (checkerboards, stripes). This allows for controlled probing of the model's representations.
3.  **Fit PCA**: To visualize the model's high-dimensional (768-D) feature space, Principal Component Analysis (PCA) is fitted on a large sample of feature patches. This learns the three most important dimensions, which are then mapped to RGB channels for visualization.
4.  **Iterative Reconstruction**: For each image, the script performs a single forward pass to generate the full set of 128 slots. It then reconstructs the image's feature space multiple times, using an increasing number of these slots (e.g., 1, 2, 4, 8, ..., 128).
5.  **Quantify Quality**: At each step, two metrics are calculated:
    *   **Reconstruction Loss (MSE)**: The raw error between the reconstructed features and the ground truth.
    *   **Cosine Similarity**: The similarity in orientation between the feature vectors.

### 3. How to Run the Analysis

To run the experiment, execute the provided shell script from the project's root directory:

```bash
bash feature_space_analysis/run_analysis.sh
```

This script will automatically find the latest model checkpoint (`model-*.ckpt`) in the root directory and launch the analysis on a SLURM cluster (or locally if you run the `python` command directly). All results will be saved in the `feature_space_analysis/results/` directory.

### 4. Expected Output

The script will generate three main types of output in the `results/` directory:

1.  **Grid Visualizations (`img_{id}_{label}_grid.png`)**: A detailed grid for each analyzed image, showing:
    *   **Row 1**: The original input image.
    *   **Row 2**: The ground truth features, visualized via PCA.
    *   **Row 3**: The model's reconstruction at different slot counts. **This should get progressively sharper and more detailed from left to right.**
    *   **Row 4**: An error map showing the difference between the ground truth and the reconstruction. **This should fade from bright to dark from left to right.**

2.  **Summary Plots (`summary_plots.png`)**: A collection of plots summarizing the results across all images, including average loss curves, similarity curves, and monotonic improvement ratios.

3.  **JSON Summary (`analysis_summary.json`)**: A data file containing all the raw metrics, average scores, and configuration details for the experiment, allowing for further programmatic analysis.

### 5. How to Interpret the Results

*   **A High Monotonic Ratio (close to 1.0)** is the key indicator of success. It provides quantitative proof that reconstruction quality consistently improves as more slots are used.
*   **A Consistently Decreasing Loss Curve** and **Consistently Increasing Similarity Curve** on the summary plots also demonstrate successful hierarchical learning.
*   **Visual inspection of the grid images** provides intuitive, qualitative proof of the model's progressive refinement capabilities.
