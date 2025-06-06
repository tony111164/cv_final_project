import numpy as np
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors
import argparse
import os
import sys
from pathlib import Path

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
        return None

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

def get_ply_files(directory_path):
    """
    Get all PLY files from a directory.
    
    Args:
        directory_path (str): Path to directory containing PLY files
        
    Returns:
        list: List of PLY file paths
    """
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return []
    
    ply_files = list(directory.glob("*.ply"))
    return sorted([str(f) for f in ply_files])

def find_matching_files(predicted_files, ground_truth_files):
    """
    Find matching PLY files between predicted and ground truth directories based on filename.
    
    Args:
        predicted_files (list): List of predicted PLY file paths
        ground_truth_files (list): List of ground truth PLY file paths
        
    Returns:
        list: List of tuples (predicted_path, ground_truth_path) for matching files
    """
    predicted_names = {Path(f).name: f for f in predicted_files}
    ground_truth_names = {Path(f).name: f for f in ground_truth_files}
    
    matching_pairs = []
    for name in predicted_names:
        if name in ground_truth_names:
            matching_pairs.append((predicted_names[name], ground_truth_names[name]))
        else:
            print(f"Warning: No ground truth file found for predicted file: {name}")
    
    for name in ground_truth_names:
        if name not in predicted_names:
            print(f"Warning: No predicted file found for ground truth file: {name}")
    
    return matching_pairs

def evaluate_directories(predicted_dir, ground_truth_dir):
    """
    Evaluate all PLY files in predicted directory against corresponding files in ground truth directory.
    
    Args:
        predicted_dir (str): Path to directory containing predicted PLY files
        ground_truth_dir (str): Path to directory containing ground truth PLY files
        
    Returns:
        tuple: (average_accuracy, average_completeness, individual_results)
    """
    print(f"Searching for PLY files in predicted directory: {predicted_dir}")
    predicted_files = get_ply_files(predicted_dir)
    print(f"Found {len(predicted_files)} predicted PLY files")
    
    print(f"Searching for PLY files in ground truth directory: {ground_truth_dir}")
    ground_truth_files = get_ply_files(ground_truth_dir)
    print(f"Found {len(ground_truth_files)} ground truth PLY files")
    
    if not predicted_files:
        print("Error: No PLY files found in predicted directory")
        return None, None, []
    
    if not ground_truth_files:
        print("Error: No PLY files found in ground truth directory")
        return None, None, []
    
    # Find matching files
    matching_pairs = find_matching_files(predicted_files, ground_truth_files)
    print(f"Found {len(matching_pairs)} matching file pairs")
    
    if not matching_pairs:
        print("Error: No matching PLY files found between directories")
        return None, None, []
    
    accuracies = []
    completenesses = []
    individual_results = []
    
    print("\nEvaluating individual files:")
    print("-" * 60)
    
    for i, (pred_path, gt_path) in enumerate(matching_pairs, 1):
        filename = Path(pred_path).name
        print(f"\n[{i}/{len(matching_pairs)}] Processing: {filename}")
        
        # Load point clouds
        predicted_points = load_ply_points(pred_path)
        ground_truth_points = load_ply_points(gt_path)
        
        if predicted_points is None or ground_truth_points is None:
            print(f"Skipping {filename} due to loading error")
            continue
        
        print(f"  Predicted points: {len(predicted_points)}")
        print(f"  Ground truth points: {len(ground_truth_points)}")
        
        # Calculate metrics
        accuracy = calculate_accuracy(predicted_points, ground_truth_points)
        completeness = calculate_completeness(predicted_points, ground_truth_points)
        
        print(f"  Accuracy: {accuracy:.6f}")
        print(f"  Completeness: {completeness:.6f}")
        
        accuracies.append(accuracy)
        completenesses.append(completeness)
        individual_results.append({
            'filename': filename,
            'accuracy': accuracy,
            'completeness': completeness,
            'predicted_points': len(predicted_points),
            'ground_truth_points': len(ground_truth_points)
        })
    
    if not accuracies:
        print("Error: No files were successfully processed")
        return None, None, []
    
    # Calculate averages
    avg_accuracy = np.mean(accuracies)
    avg_completeness = np.mean(completenesses)
    
    return avg_accuracy, avg_completeness, individual_results

def save_results(results, output_file, predicted_dir, ground_truth_dir, avg_accuracy, avg_completeness):
    """
    Save evaluation results to a file.
    """
    with open(output_file, 'w') as f:
        f.write("PLY Point Cloud Batch Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Predicted Directory: {predicted_dir}\n")
        f.write(f"Ground Truth Directory: {ground_truth_dir}\n")
        f.write(f"Total Files Evaluated: {len(results)}\n")
        f.write("\n")
        f.write("AVERAGE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average Accuracy: {avg_accuracy:.6f}\n")
        f.write(f"Average Completeness: {avg_completeness:.6f}\n")
        f.write("\n")
        f.write("INDIVIDUAL FILE RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for result in results:
            f.write(f"File: {result['filename']}\n")
            f.write(f"  Accuracy: {result['accuracy']:.6f}\n")
            f.write(f"  Completeness: {result['completeness']:.6f}\n")
            f.write(f"  Predicted points: {result['predicted_points']}\n")
            f.write(f"  Ground truth points: {result['ground_truth_points']}\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Calculate average Accuracy and Completeness metrics for directories of PLY files')
    parser.add_argument('predicted_dir', help='Path to directory containing predicted PLY files')
    parser.add_argument('ground_truth_dir', help='Path to directory containing ground truth PLY files')
    parser.add_argument('--output', '-o', help='Output file to save detailed results (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed per-file results')
    
    args = parser.parse_args()
    
    # Evaluate directories
    avg_accuracy, avg_completeness, individual_results = evaluate_directories(
        args.predicted_dir, args.ground_truth_dir
    )
    
    if avg_accuracy is None or avg_completeness is None:
        print("Evaluation failed. Please check the directories and try again.")
        sys.exit(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH EVALUATION RESULTS")
    print("=" * 60)
    print(f"Files evaluated: {len(individual_results)}")
    print(f"Average Accuracy:    {avg_accuracy:.6f}")
    print(f"Average Completeness: {avg_completeness:.6f}")
    print("=" * 60)
    
    # Show individual results if verbose
    if args.verbose:
        print("\nINDIVIDUAL FILE RESULTS:")
        print("-" * 60)
        for result in individual_results:
            print(f"{result['filename']}: Acc={result['accuracy']:.6f}, Comp={result['completeness']:.6f}")
    
    # Save results if output file specified
    if args.output:
        save_results(individual_results, args.output, args.predicted_dir, 
                    args.ground_truth_dir, avg_accuracy, avg_completeness)
        print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()