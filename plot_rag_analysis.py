import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Try importing PCA, handle if missing
try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. PCA plots will be skipped.")
    print("To install: pip install scikit-learn")

# ==========================================
# CONFIGURATION
# ==========================================

# Directory containing the dataset files
DATASET_DIR = "final_dataset"
OUTPUT_DIR = "plots-analysis"
TOP_K = 200

# Define your target parameters here for each environment.
# These should be lists of floats corresponding to the parameter vector size.
TARGET_PARAMS = {
    # Text key matches the prefix of the dataset file (e.g., 'hopper' for 'hopper_dataset.txt')
    "cliffwalking": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
    "hopper": [1.1, 0.0, 0.5, 0.0, 1.1, 1.1, 0.0, 0.6, 1.1, 0.6, 1.1, 0.6, 0.6, 0.0, 0.1, 0.6, 1.6, 1.6, 0.1, 0.6, 0.6, 1.1, 0.1, 0.1, -0.6, 0.6, 0.1, 0.6, 0.1, 0.6, 0.6, 0.1, 0.6, 0.6, 0.6, 0.6],
    "inverted_double_pendulum": [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0],
    "invertedpendulum": [0.0, 5.9, 0.5, 2.5, 0.0],
    "mountaincar": [0.0, -0.6, 0.0, 0.0, 4.5, 6.5, 0.0, -3.0, 0.0],
    "mountaincarcontinuous": [-5.5, -5.5, -3.1],
    "pong": [0.0, -1.0, -2.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 1.0],
    "reacher": [3.3, 3.3, -3.3, 6.7, 3.6, 3.4, 1.1, 4.3, 3.8, 6.1, 3.6, 7.2, -0.4, 0.9, 0.7, -0.2, 0.9, 1.7, -0.9, 1.7, 2.0, -2.7],
    "swimmer": [0.8, -0.9, 0.8, -0.9, 0.8, -0.9, 0.8, -0.9, 0.8, -0.9, 0.8, -0.9, 0.8, -0.9, 0.8, -0.9, 0.8, -0.9],
    "walker2d": [-1.1, -2.5, 0.5, -0.7, -0.6, -3.1, 0.3, 1.2, 2.5, 0.9, 2.9, -0.1, -0.4, 1.5, 2.0, 0.1, 0.5, -0.1, -1.3, 1.4, -0.9, 1.1, -1.5, -1.2, 1.2, 0.8, 0.3, 0.6, -0.7, 0.2, 3.5, -2.4, -0.6, 1.7, -1.5, 1.5, 4.2, 0.1, -2.3, 0.7, -0.5, 0.9, -1.7, 0.8, -3.2, -0.0, 1.4, -1.5, 2.9, 2.0, -0.6, -1.4, 0.5, -1.3, -0.4, 0.5, -1.5, 2.7, -2.3, 0.4, -0.7, -1.0, 3.1, -0.9, -0.6, 1.3, -2.1, -0.3, 0.4, -0.1, 2.6, 1.2, -2.7, -0.5, -1.0, 1.3, 0.2, 0.8, 0.7, -2.5, 1.6, -0.3, -1.3, -1.2, 0.9, 0.4, -1.7, 1.1, 0.6, 0.9, 0.3, 0.5, -1.5, -0.7, 1.0, -1.6, -0.1, 0.2, 0.3, -1.5, -0.2, -0.1, -2.1, 0.9, 0.1, 1.6, 0.7, 0.3],
    "nav": [2.6, 8.5, -5.3, 24.5, 16.5, 8.5, 11.1, -3.8, 10.8, 5.2, 19.4, 8.2, 5.7, 9.9, 5.7, 0.1, 3.4, 9.1],
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_dataset(env_name):
    """
    Loads proper dataset file for the given env_name.
    Expected format in file: param1, param2, ... | reward
    """
    # Construct filename
    filename = f"{env_name}_dataset.txt"
    filepath = os.path.join(DATASET_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"Dataset not found for {env_name}: {filepath}")
        return None

    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split('|')
                if len(parts) != 2:
                    continue
                
                # Parse Params
                params_str = parts[0].strip()
                params = [float(x.strip()) for x in params_str.split(',') if x.strip()]
                
                # Parse Reward
                reward = float(parts[1].strip())
                
                data.append((params, reward))
            except Exception as e:
                print(f"Error parsing line in {filename}: {line[:30]}... Error: {e}")
                continue
                
    return data

def perform_rag_retrieval(train_data, query_params, top_k=TOP_K):
    """
    Performs the RAG retrieval logic: Calculate Euclidean distance and return top_k.
    """
    query_vec = np.array(query_params)
    scored_examples = []
    
    for params, reward in train_data:
        params_vec = np.array(params)
        dist = np.linalg.norm(query_vec - params_vec)
        scored_examples.append((dist, params_vec, reward))

    # Sort by distance (ascending)
    scored_examples.sort(key=lambda x: x[0])
    
    # Return top K
    return scored_examples[:top_k]

def plot_distance_vs_reward(scored_examples, env_name, save_dir):
    """
    Scatterplot of Euclidean Distance (x) vs Reward (y).
    """
    distances = [x[0] for x in scored_examples]
    rewards = [x[2] for x in scored_examples]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, rewards, alpha=0.6, edgecolors='b')
    plt.xlabel(f"Euclidean Distance from Target (Top {len(scored_examples)})")
    plt.ylabel("Reward")
    plt.title(f"Reward vs Distance for '{env_name}'")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    filepath = os.path.join(save_dir, f"{env_name}_dist_reward.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved scatterplot to {filepath}")

def plot_pca_analysis(scored_examples, query_params, env_name, save_dir):
    """
    Perform PCA on [query_params + retrieved_params] and plot in 2D.
    """
    if not HAS_SKLEARN:
        return

    # Prepare data for PCA
    # Index 0 will be the query params
    # Indices 1..k will be the retrieved params
    
    retrieved_params = [x[1] for x in scored_examples]
    rewards = [x[2] for x in scored_examples]
    
    all_vectors = np.vstack([np.array(query_params), retrieved_params])
    
    # Run PCA
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(all_vectors)
    
    # Split back
    query_proj = transformed[0]
    retrieved_proj = transformed[1:]
    
    plt.figure(figsize=(10, 8))
    
    # Plot retrieved points - color coded by reward if desired, or just simple scatter
    sc = plt.scatter(retrieved_proj[:, 0], retrieved_proj[:, 1], 
                     c=rewards, cmap='viridis', s=50, alpha=0.8, label='Retrieved Params')
    plt.colorbar(sc, label='Reward')
    
    # Plot query point
    plt.scatter(query_proj[0], query_proj[1], c='red', marker='X', s=200, label='Target Param', edgecolors='black')
    
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"PCA Projection of Parameter Space for '{env_name}'\nExplained Variance: {pca.explained_variance_ratio_}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    filepath = os.path.join(save_dir, f"{env_name}_pca.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved PCA plot to {filepath}")

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Checking {len(TARGET_PARAMS)} environments...")
    
    for env_name, query_params in TARGET_PARAMS.items():
        if not query_params:
            print(f"Skipping '{env_name}': No target params provided.")
            continue
            
        print(f"\nProcessing '{env_name}'...")
        
        # 1. Load Data
        train_data = load_dataset(env_name)
        if not train_data:
            continue
            
        # Verify param dimensions match
        if len(train_data[0][0]) != len(query_params):
            print(f"Warning: Dimension mismatch for {env_name}. "
                  f"Target has {len(query_params)}, dataset has {len(train_data[0][0])}. Skipping.")
            continue

        # 2. Perform RAG
        top_k_results = perform_rag_retrieval(train_data, query_params, TOP_K)
        print(f"Retrieved {len(top_k_results)} examples.")
        
        # 3. Create env specific output folder
        env_plot_dir = os.path.join(OUTPUT_DIR, env_name)
        if not os.path.exists(env_plot_dir):
            os.makedirs(env_plot_dir)
            
        # 4. Generate Plots
        plot_distance_vs_reward(top_k_results, env_name, env_plot_dir)
        plot_pca_analysis(top_k_results, query_params, env_name, env_plot_dir)

if __name__ == "__main__":
    main()
