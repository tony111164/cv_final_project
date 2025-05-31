import argparse
import os

def visualize_ply(PATH_TO_FILE):
    """
    Visualize a PLY file using Open3D.

    Parameters:
    -----------
    PATH_TO_FILE : str
        Path to the PLY file to visualize
    """
    try:
        import open3d as o3d
        import numpy as np

        # Check if file exists
        if not os.path.exists(PATH_TO_FILE):
            raise FileNotFoundError(f"The file {PATH_TO_FILE} does not exist")

        # Check if file is a PLY file
        if not PATH_TO_FILE.lower().endswith('.ply'):
            print("Warning: File may not be a PLY file. Attempting to load anyway.")

        # Load the PLY file
        print(f"Loading PLY file: {PATH_TO_FILE}")
        mesh = o3d.io.read_triangle_mesh(PATH_TO_FILE)

        # If it's a point cloud with no triangles, try loading as point cloud instead
        if len(np.asarray(mesh.triangles)) == 0:
            print("No triangles found, loading as point cloud instead")
            pcd = o3d.io.read_point_cloud(PATH_TO_FILE)

            # Estimate normals for better visualization if they don't exist
            if not pcd.has_normals():
                pcd.estimate_normals()

            # First try direct visualization (simpler approach)
            try:
                print("Attempting to visualize point cloud...")
                o3d.visualization.draw_geometries([pcd])
            except Exception as viz_error:
                print(f"Direct visualization failed: {str(viz_error)}")
                
                # Alternative: Save as image if visualization fails
                print("Saving point cloud as image instead...")
                output_path = os.path.splitext(PATH_TO_FILE)[0] + "_view.png"
                
                try:
                    # Try offscreen rendering
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(visible=False, width=800, height=600)
                    
                    # Only proceed if window creation was successful
                    if vis.get_render_option() is not None:
                        vis.add_geometry(pcd)
                        vis.get_render_option().point_size = 1.0
                        
                        # Update view
                        vis.poll_events()
                        vis.update_renderer()
                        
                        # Capture image
                        vis.capture_screen_image(output_path, True)
                        print(f"Point cloud image saved to: {output_path}")
                    else:
                        print("Failed to create offscreen rendering window")
                    
                    vis.destroy_window()
                except Exception as img_error:
                    print(f"Failed to save as image: {str(img_error)}")
                    print("\nTo fix this issue, try installing GUI support libraries:")
                    print("sudo apt-get update")
                    print("sudo apt-get install -y xvfb libgl1-mesa-glx libglu1-mesa")
        else:
            # If it's a mesh, compute normals if they don't exist
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()

            # Add color if it doesn't exist
            if not mesh.has_vertex_colors():
                mesh.paint_uniform_color([0.7, 0.7, 0.7])

            # Try to visualize the mesh
            try:
                o3d.visualization.draw_geometries([mesh])
            except Exception as viz_error:
                print(f"Mesh visualization failed: {str(viz_error)}")
                
                # Alternative: Save as image if visualization fails
                print("Saving mesh as image instead...")
                output_path = os.path.splitext(PATH_TO_FILE)[0] + "_view.png"
                
                try:
                    # Try offscreen rendering
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(visible=False, width=800, height=600)
                    
                    if vis.get_render_option() is not None:
                        vis.add_geometry(mesh)
                        vis.poll_events()
                        vis.update_renderer()
                        vis.capture_screen_image(output_path, True)
                        print(f"Mesh image saved to: {output_path}")
                    else:
                        print("Failed to create offscreen rendering window")
                        
                    vis.destroy_window()
                except Exception as img_error:
                    print(f"Failed to save as image: {str(img_error)}")
                    print("\nTo fix this issue, try installing GUI support libraries:")
                    print("sudo apt-get update")
                    print("sudo apt-get install -y xvfb libgl1-mesa-glx libglu1-mesa")

        print("Visualization process completed")

    except ImportError:
        print("Please install Open3D: pip install open3d")
    except Exception as e:
        print(f"Error visualizing PLY file: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a PLY file using Open3D.")
    parser.add_argument("ply_file", type=str, help="Path to the PLY file to visualize")
    args = parser.parse_args()
    visualize_ply(args.ply_file)