import trimesh
import pyrender
import numpy as np
import os
import imageio


def render_gltf_with_rotations(gltf_path, output_dir, angles):
    """
    Render a GLTF file with three planes of rotation (top, middle, bottom).

    Parameters:
        gltf_path (str): Path to the GLTF file.
        output_dir (str): Directory to save the rendered images.
        angles (list): List of rotation angles (in degrees) for the camera.
    """
    # Load the GLTF model as a Trimesh Scene
    scene = trimesh.load(gltf_path)

    # Compute the bounding box and center the model
    bounding_box = scene.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    center = (bounding_box[0] + bounding_box[1]) / 2.0
    scene.apply_translation(-center)

    # Compute the bounding sphere radius
    bounding_sphere_radius = np.linalg.norm(bounding_box[1] - bounding_box[0]) / 2.0

    # Create a Pyrender Scene
    render_scene = pyrender.Scene()

    # Add all meshes from the Trimesh scene to the Pyrender scene
    model_nodes = []
    for name, mesh in scene.geometry.items():
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        node = render_scene.add(pyrender_mesh)
        model_nodes.append(node)

    # Add lighting
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    render_scene.add(light, pose=np.eye(4))

    # Field of view in radians
    yfov = np.pi / 3.0  # 60 degrees

    # Calculate camera distance
    camera_distance = bounding_sphere_radius / np.sin(yfov / 2)

    # Create a camera and position it at a reasonable distance
    camera = pyrender.PerspectiveCamera(yfov=yfov)

    # Camera pose (along the Z-axis, looking at the origin)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Camera x position (centered)
        [0.0, 1.0, 0.0, 0.0],  # Camera y position (centered)
        [0.0, 0.0, 1.0, camera_distance],  # Camera z position
        [0.0, 0.0, 0.0, 1.0]
    ])
    render_scene.add(camera, pose=camera_pose)

    # Define rotation planes (top, middle, bottom)
    rotation_planes = [
        (1, 0, 0),  # Top: Rotate around X-axis (view from above)
        (0, 1, 0),  # Middle: Rotate around Y-axis (normal view)
        (-1, 0, 0)  # Bottom: Rotate around X-axis (view from below)
    ]

    # Loop through each plane and rotation angle
    for idx, angle in enumerate(angles):
        for plane in rotation_planes:
            # Calculate the rotation matrix for the model for this plane
            rotation_matrix = trimesh.transformations.rotation_matrix(
                np.radians(angle), plane
            )

            # Apply rotation to each model node
            for node in model_nodes:
                render_scene.set_pose(node, pose=rotation_matrix)

            # Render the scene
            renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)
            color, _ = renderer.render(render_scene)

            # Save the rendered image
            output_path = os.path.join(output_dir, f'render_plane_{plane}_{idx}.png')
            imageio.imwrite(output_path, color)

            # Clean up renderer for next loop
            renderer.delete()

    # Clean up the scene
    render_scene.clear()


# Example Usage
gltf_file = "extracted/scene.gltf"
output_directory = "images"
os.makedirs(output_directory, exist_ok=True)

# List of rotation angles (in degrees) for different perspectives
rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]

render_gltf_with_rotations(gltf_file, output_directory, rotation_angles)
