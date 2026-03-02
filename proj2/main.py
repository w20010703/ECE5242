import os
import pickle
import numpy as np
import pandas as pd

from hmm import GestureHMM 

columns = ['Time', 'Wx', 'Wy', 'Wz', 'Ax', 'Ay', 'Az']

def load_sensor_data(filepath):
    """Load test sensor data (same format as training)."""
    df = pd.read_csv(filepath, header=None, sep=r'\s+', engine='python', skip_blank_lines=True)
    if df.shape[1] != 7:
        df = pd.read_csv(filepath, header=None, sep=',', engine='python')
    df.columns = columns
    return df[['Wx', 'Wy', 'Wz', 'Ax', 'Ay', 'Az']].values

def calculate_log_likelihood(model, obs_seq):
    """Compute log-likelihood using forward scale factors: log P(O) = sum(log(scale))."""
    _, scale = model._forward_scaled(obs_seq)
    log_likelihood = np.sum(np.log(scale))
    return log_likelihood

def test_single_file(filepath, kmeans, hmm_dict):
    """Run classification on one file; return top 3 predictions."""
    data = load_sensor_data(filepath)
    obs_seq = kmeans.predict(data)
    results = {}
    for gesture_name, model in hmm_dict.items():
        ll = calculate_log_likelihood(model, obs_seq)
        results[gesture_name] = ll
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:3]

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Loading saved models...")
    with open(os.path.join(script_dir, 'kmeans_model.pkl'), 'rb') as f:
        kmeans_model = pickle.load(f)
    with open(os.path.join(script_dir, 'hmm_models.pkl'), 'rb') as f:
        hmm_models = pickle.load(f)
        
    test_folder = os.path.join(script_dir, 'ECE5242Proj2-train') 
    
    if os.path.exists(test_folder):
        print("\n=== Starting Classification ===")
        for file in sorted(os.listdir(test_folder)):
            if file.endswith('.txt'):
                filepath = os.path.join(test_folder, file)
                
                top_3 = test_single_file(filepath, kmeans_model, hmm_models)
                
                print(f"\nFile: {file}")
                print(f"  1st Prediction: {top_3[0][0]} (Log-Likelihood: {top_3[0][1]:.2f})")
                print(f"  2nd Prediction: {top_3[1][0]} (Log-Likelihood: {top_3[1][1]:.2f})")
                print(f"  3rd Prediction: {top_3[2][0]} (Log-Likelihood: {top_3[2][1]:.2f})")
    else:
        print(f"Test folder not found: {test_folder}. Please download the test data when released on 3/4[cite: 8].")