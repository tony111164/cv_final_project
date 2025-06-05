import numpy as np
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors
import argparse
import sys

def load_ply_points(ply_file_path):
    """
    Load point coordinates from a PLY file.
    
    Args:
        ply_file_path (str): Path to the PLY file
        
    Returns:
        np.ndarray: Array of 3D points with shape (N, 3)
    """
    try:
        plydata = PlyData.read(ply_file_path)
        vertex = plydata['vertex']
        
        # Extract x, y, z coordinates
        points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
        return points
    except Exception as e:
        print(f"Error loading PLY file {ply_file_path}: {e}")
        sys.exit(1)

def calculate_accuracy(predicted_points, ground_truth_points):
    """
    Calculate Accuracy metric: For each predicted point, find its nearest neighbor 
    in the ground-truth point cloud and compute Euclidean distance.
    Take the median of these distances.
    
    Args:
        predicted_points (np.ndarray): Predicted point cloud (P)
        ground_truth_points (np.ndarray): Ground-truth point cloud (G)
        
    Returns:
        float: Accuracy score
    """
    if len(predicted_points) == 0:
        return float('inf')
    
    # Build KNN model for ground-truth points
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth_points)
    
    # Find nearest neighbors for each predicted point
    distances, indices = nbrs.kneighbors(predicted_points)
    
    # Calculate median distance
    accuracy = np.median(distances.flatten())
    
    return accuracy

def calculate_completeness(predicted_points, ground_truth_points):
    """
    Calculate Completeness metric: For each ground-truth point, find its nearest 
    neighbor in the predicted point cloud and compute Euclidean distance.
    Take the median of these distances.
    
    Args:
        predicted_points (np.ndarray): Predicted point cloud (P)
        ground_truth_points (np.ndarray): Ground-truth point cloud (G)
        
    Returns:
        float: Completeness score
    """
    if len(ground_truth_points) == 0:
        return float('inf')
    
    # Build KNN model for predicted points
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(predicted_points)
    
    # Find nearest neighbors for each ground-truth point
    distances, indices = nbrs.kneighbors(ground_truth_points)
    
    # Calculate median distance
    completeness = np.median(distances.flatten())
    
    return completeness

def evaluate_point_clouds(predicted_ply_path, ground_truth_ply_path):
    """
    Evaluate predicted point cloud against ground truth using Accuracy and Completeness metrics.
    
    Args:
        predicted_ply_path (str): Path to predicted PLY file
        ground_truth_ply_path (str): Path to ground truth PLY file
        
    Returns:
        tuple: (accuracy, completeness) scores
    """
    print(f"Loading predicted point cloud from: {predicted_ply_path}")
    predicted_points = load_ply_points(predicted_ply_path)
    print(f"Predicted points: {len(predicted_points)}")
    
    print(f"Loading ground truth point cloud from: {ground_truth_ply_path}")
    ground_truth_points = load_ply_points(ground_truth_ply_path)
    print(f"Ground truth points: {len(ground_truth_points)}")
    
    print("\nCalculating metrics...")
    
    # Calculate Accuracy (Metric1)
    accuracy = calculate_accuracy(predicted_points, ground_truth_points)
    print(f"Accuracy (Metric1): {accuracy:.6f}")
    
    # Calculate Completeness (Metric2)
    completeness = calculate_completeness(predicted_points, ground_truth_points)
    print(f"Completeness (Metric2): {completeness:.6f}")
    
    return accuracy, completeness

def main():
    parser = argparse.ArgumentParser(description='Evaluate PLY point cloud using Accuracy and Completeness metrics')
    parser.add_argument('predicted_ply', help='Path to predicted PLY file')
    parser.add_argument('ground_truth_ply', help='Path to ground truth PLY file')
    parser.add_argument('--output', '-o', help='Output file to save results (optional)')
    
    args = parser.parse_args()
    
    # Evaluate point clouds
    accuracy, completeness = evaluate_point_clouds(args.predicted_ply, args.ground_truth_ply)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy (Metric1):    {accuracy:.6f}")
    print(f"Completeness (Metric2): {completeness:.6f}")
    print("="*50)
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write("PLY Point Cloud Evaluation Results\n")
            f.write("="*40 + "\n")
            f.write(f"Predicted PLY: {args.predicted_ply}\n")
            f.write(f"Ground Truth PLY: {args.ground_truth_ply}\n")
            f.write(f"Accuracy (Metric1): {accuracy:.6f}\n")
            f.write(f"Completeness (Metric2): {completeness:.6f}\n")
        print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()