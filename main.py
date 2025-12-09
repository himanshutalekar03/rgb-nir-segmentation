"""
RGB–NIR Multispectral Segmentation with K Selection (Elbow Method)
------------------------------------------------------------------
Pipeline:

1. For each RGB+NIR image pair:
   - Build spectral features (R, G, B, NIR, ratios, differences).
   - Run K-Means for K in [K_MIN..K_MAX].
   - Select best K with the elbow method (distance-to-line).

2. Aggregate Ks over all images:
   - Count how many images prefer each K.
   - Plot a bar chart.
   - Choose GLOBAL_K as the most frequent K.

3. Using GLOBAL_K:
   - Segment all images.
   - Save RGB, NIR, and segmented PNGs.


Author: Himanshu
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(BASE_DIR, "sample")

K_MIN = 2
K_MAX = 8

RUN_ELBOW_ANALYSIS = True
RUN_FINAL_SEGMENTATION = True

ELBOW_PLOT_ROOT = os.path.join(ROOT_DIR, "elbow_plots")
SEGMENT_OUT_ROOT = os.path.join(ROOT_DIR, "outputs_k")

os.makedirs(ELBOW_PLOT_ROOT, exist_ok=True)
os.makedirs(SEGMENT_OUT_ROOT, exist_ok=True)



def build_features(rgb_vis: np.ndarray, nir: np.ndarray):
    rgb_f = rgb_vis.astype(np.float32) / 255.0
    nir_f = nir.astype(np.float32) / 255.0

    
    nir_f = cv2.resize(nir_f, (rgb_f.shape[1], rgb_f.shape[0]))

    H, W, _ = rgb_f.shape
    R = rgb_f[:, :, 0].reshape(-1)
    G = rgb_f[:, :, 1].reshape(-1)
    B = rgb_f[:, :, 2].reshape(-1)
    N = nir_f.reshape(-1)

    eps = 1e-6
    features = np.stack(
        [
            R,
            G,
            B,
            N,
            N / (R + eps),
            N - R,
            (R + G + B) / 3.0,
        ],
        axis=1,
    )
    return features, nir_f



def find_best_k_elbow(features, k_min, k_max, title, save_dir):

    Ks = list(range(k_min, k_max + 1))
    inertias = []

    print(f"    Elbow analysis: {title}")
    for k in Ks:
        print(f"      - KMeans K={k}")
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(features)
        inertias.append(km.inertia_)

    x = np.array(Ks, float)
    y = np.array(inertias, float)

    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    
    distances = np.abs((y2 - y1) * x - (x2 - x1) * y +
                       x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    best_index = int(np.argmax(distances))
    k_best = Ks[best_index]

    print(f"    → Best K = {k_best}")

   
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(Ks, inertias, marker="o")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title(f"Elbow Curve - {title}")
    plt.grid(True)
    for k, inertia in zip(Ks, inertias):
        plt.text(k, inertia, str(k))

    out_path = os.path.join(save_dir, f"elbow_{title}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"    Saved elbow plot to: {out_path}")
    return k_best



def segment_with_k(rgb_vis, nir_f, features, k):

    H, W, _ = rgb_vis.shape

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(features)
    centers = km.cluster_centers_

    
    nir_values = centers[:, 3]
    sorted_idx = np.argsort(nir_values)

    base_colors = [
        [0, 0, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0],
        [255, 0, 255], [0, 255, 255], [128, 128, 128], [255, 128, 0]
    ]

    color_map = {
        cluster_id: base_colors[i % len(base_colors)]
        for i, cluster_id in enumerate(sorted_idx)
    }

    seg_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for label in range(k):
        seg_rgb[labels.reshape(H, W) == label] = color_map[label]

    return seg_rgb



def run_elbow_analysis(root_dir):

    scene_folders = [
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
        and f not in ["elbow_plots", "outputs_k"]
        and not f.startswith(".")
    ]

    
    if not scene_folders:
        scene_folders = ["."]
        print("No subfolders found → using ROOT_DIR as a single scene.")

    print("Scene folders:", scene_folders)

    per_image_K = []

    for scene in scene_folders:
        if scene == ".":
            scene_dir = root_dir
            scene_name = "sample"
        else:
            scene_dir = os.path.join(root_dir, scene)
            scene_name = scene

        save_dir = os.path.join(ELBOW_PLOT_ROOT, scene_name)
        print(f"\n=== Scene: {scene_name} ===")

        files = sorted(os.listdir(scene_dir))
        rgb_files = [f for f in files if f.endswith("_rgb.tiff")]

        print(f"  Found {len(rgb_files)} RGB files")

        for file in rgb_files:
            rgb_path = os.path.join(scene_dir, file)
            nir_path = rgb_path.replace("_rgb.tiff", "_nir.tiff")

            rgb = cv2.imread(rgb_path)
            nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
            if rgb is None or nir is None:
                print(f"  [!] Skipping {file}")
                continue

            rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            features, _ = build_features(rgb_vis, nir)

            title = f"{scene_name}_{file.replace('_rgb.tiff','')}"
            k_best = find_best_k_elbow(features, K_MIN, K_MAX, title, save_dir)
            per_image_K.append(k_best)

    
    print("\n===== GLOBAL K STATS =====")
    per_image_K = np.array(per_image_K)

    if len(per_image_K) == 0:
        print("No images found → cannot compute global K.")
        return 4 

    unique, counts = np.unique(per_image_K, return_counts=True)

    for u, c in zip(unique, counts):
        print(f"K = {u}: {c} images")

    # BAR CHART
    plt.figure(figsize=(6, 4))
    plt.bar(unique, counts)
    plt.xlabel("K")
    plt.ylabel("Image Count")
    plt.title("Best K Frequency (Elbow Method)")
    plt.grid(axis="y")
    bar_path = os.path.join(ELBOW_PLOT_ROOT, "global_K_histogram.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()

    print(f"Saved global K histogram: {bar_path}")

    global_k = int(unique[np.argmax(counts)])
    print(f"\n>>> GLOBAL_K = {global_k}\n")

    return global_k



def run_final_segmentation(root_dir, global_k):

    scene_folders = [
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
        and f not in ["elbow_plots", "outputs_k"]
        and not f.startswith(".")
    ]

    if not scene_folders:
        scene_folders = ["."]
        print("No subfolders found → using ROOT_DIR as a single scene.")

    print(f"\nRunning final segmentation with K = {global_k}")

    for scene in scene_folders:

        if scene == ".":
            scene_dir = root_dir
            scene_name = "sample"
        else:
            scene_dir = os.path.join(root_dir, scene)
            scene_name = scene

        out_dir = os.path.join(SEGMENT_OUT_ROOT, scene_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== Scene: {scene_name} ===")

        files = sorted(os.listdir(scene_dir))
        rgb_files = [f for f in files if f.endswith("_rgb.tiff")]

        for file in rgb_files:

            rgb_path = os.path.join(scene_dir, file)
            nir_path = rgb_path.replace("_rgb.tiff", "_nir.tiff")

            rgb = cv2.imread(rgb_path)
            nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

            if rgb is None or nir is None:
                print(f"  [!] Skipping {file}")
                continue

            rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            features, nir_f = build_features(rgb_vis, nir)
            seg_rgb = segment_with_k(rgb_vis, nir_f, features, global_k)

            stem = file.replace("_rgb.tiff", "")

            cv2.imwrite(os.path.join(out_dir, f"{stem}_rgb.png"),
                        cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_dir, f"{stem}_nir.png"),
                        (nir_f * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(out_dir, f"{stem}_segmented_K{global_k}.png"),
                        seg_rgb)

            print(f"  Saved: {file}")



if __name__ == "__main__":

    if RUN_ELBOW_ANALYSIS:
        global_K = run_elbow_analysis(ROOT_DIR)
    else:
        global_K = 4
        print("Skipping elbow. Using default K=4.")

    if RUN_FINAL_SEGMENTATION:
        run_final_segmentation(ROOT_DIR, global_K)
    else:
        print("Segmentation skipped.")
