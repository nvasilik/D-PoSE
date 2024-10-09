import OpenEXR, Imath
import numpy as np
import cv2
from scipy.special import expit
from glob import glob
from math import floor, ceil
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
import pyrender
import colorsys
from matplotlib import cm as mpl_cm, colors as mpl_colors
def read_depth_exr(path):
    """Reads depth map from EXR file.
    Args:
        path: Path to the EXR file.
    Returns:
        depth: Depth map as a numpy array.
    """
    f = OpenEXR.InputFile(path)
    dw = f.header()['dataWindow']
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    c = f.channel('Depth', Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.fromstring(c, dtype=np.float32)
    depth = np.reshape(depth, (height, width))
    return depth

def read_segmentation_mask_env(path):
    """Reads segmentation mask from PNG file.
    Args:
        path: Path to the PNG file.
    Returns:
        segmentation_mask: Segmentation mask as a numpy array.
    """
    path = path + '_env.png'
    segmentation_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255
    segmentation_mask = 1 - segmentation_mask
    return segmentation_mask


def read_body_clothing_segm_mask_(path, keypoints):
    """Reads multiple segmentation masks from all PNG files for the same image.
    Args:
        path: Path to the PNG files.
        keypoint: Keypoint to locate the human in the segmentation mask.
    Returns:
        mask: Segmentation mask as a numpy array.
    """
    segmentation_masks = []
    #ls files with path + _body.png and _clothing.png
    list_files = glob(path + '*_body.png') + glob(path + '*_clothing.png')
    mask1_path=None
    mask1 = None
    keypoints = np.floor(keypoints).astype(int)
    if len(list_files) >2:
        for k in list_files:
            mask = cv2.imread(k, cv2.IMREAD_GRAYSCALE)/255
            mask = 1 - mask
            keypoints_m= keypoints.copy()
            if mask.shape[0] == 1280:
                #swap x and y
                keypoints_m[:,[0, 1]] = keypoints_m[:,[1, 0]]
            keypoints_m= keypoints_m[keypoints_m[:,1] < mask.shape[0]]
            keypoints_m= keypoints_m[keypoints_m[:,0] < mask.shape[1]]
            keypoints_m= keypoints_m[keypoints_m[:,1] > 0]
            keypoints_m= keypoints_m[keypoints_m[:,0] > 0]
            #Check if the keypoint is inside the mask
            #if mask[keypoint_floor[1], keypoint_floor[0]] == 0 or mask[keypoint_ceil[1], keypoint_ceil[0]] == 0:
            #    mask1 = mask
            #    mask1_path = k
            #Check if any of the keypoints is inside the mask
            if np.any(mask[keypoints_m[:,1], keypoints_m[:,0]] == 0.0):
                mask1 = mask
                mask1_path = k
        if mask1_path is None:
            mask= cv2.imread(path+'_env.png', cv2.IMREAD_GRAYSCALE)/255
            return 1-mask
            #print('No mask found for keypoints')
            #print(list_files)
            #print(mask.shape)
            #import ipdb; ipdb.set_trace()
        #if mask1_path ends with _body.png replace it with _clothing.png
        if mask1_path.endswith('_body.png'):
            mask2_path = mask1_path.replace('_body.png', '_clothing.png')
        elif mask1_path.endswith('_clothing.png'):
            mask2_path = mask1_path.replace('_clothing.png', '_body.png')
        if (mask2_path is not None) and mask2_path in list_files:
            mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)/255
            mask2 = 1 - mask2
            #combine the masks
            mask = mask1 
            mask[mask2 == 0] = 0
        else:
            mask = mask1
    else:
        if len(list_files) == 1:
            mask = cv2.imread(list_files[0], cv2.IMREAD_GRAYSCALE)/255
            if list_files[0].endswith('_env.png'):
                mask = 1 - mask
                return mask
            else:
                return mask

        elif len(list_files) == 2:
            mask1 = cv2.imread(list_files[0], cv2.IMREAD_GRAYSCALE)/255
            mask1 = 1 - mask1
            mask2 = cv2.imread(list_files[1], cv2.IMREAD_GRAYSCALE)/255
            mask2 = 1 - mask2
            mask = mask1
            mask[mask2 == 0] = 0
        else:
            mask = cv2.imread(path+'_env.png', cv2.IMREAD_GRAYSCALE)/255
            #mask = 1 - mask
    return 1-mask
    
def read_body_clothing_segm_mask(path, keypoints):
    """
    Reads and combines body and clothing segmentation masks for an image.

    Args:
        path (str): Path to the directory containing PNG mask files.
        keypoints (np.ndarray): Array of keypoints to locate the human in the segmentation masks.

    Returns:
        np.ndarray: Combined segmentation mask as a numpy array.
    """
    # List all body and clothing mask files in the directory.
    mask_files = glob(f"{path}*_body.png") + glob(f"{path}*_clothing.png")
    keypoints = np.floor(keypoints).astype(int)  # Round and convert keypoints to integer.
    
    selected_mask = None
    for file_path in mask_files:
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) / 255.0  # Normalize to range [0, 1].
        mask = 1 - mask  # Invert mask.
        
        if mask.shape[0] == 1280:  # Conditionally swap x and y for specific mask size.
            keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
        
        # Filter keypoints within mask bounds.
        valid_keypoints = keypoints[(keypoints[:, 1] < mask.shape[0]) & (keypoints[:, 0] < mask.shape[1]) & (keypoints[:, 1] > 0) & (keypoints[:, 0] > 0)]
        
        # Check if any keypoints are within the mask.
        if np.any(mask[valid_keypoints[:, 1], valid_keypoints[:, 0]] == 0.0):
            selected_mask = mask
            selected_mask_path = file_path
            break  # Stop once a matching mask is found.

    if selected_mask is None:  # Fallback to environment mask if no body/clothing mask matches keypoints.
        return 1 - cv2.imread(f"{path}_env.png", cv2.IMREAD_GRAYSCALE) / 255.0
    
    # Determine the path for the complementary mask (body/clothing).
    complementary_mask_path = selected_mask_path.replace('_body.png', '_clothing.png') if '_body.png' in selected_mask_path else selected_mask_path.replace('_clothing.png', '_body.png')
    
    if complementary_mask_path in mask_files:
        complementary_mask = cv2.imread(complementary_mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        complementary_mask = 1 - complementary_mask  # Invert mask.
        combined_mask = np.minimum(selected_mask, complementary_mask)  # Combine masks by taking the minimum.
    else:
        combined_mask = selected_mask

    return 1 - combined_mask  # Return the final combined and inverted mask.

def read_human_background_depth(path_depth, path_segmentation_mask):
    """Reads depth map from EXR file for both human and background.
    Args:
        path_depth: Path to the EXR file containing the depth map.
        path_segmentation_mask: Path to the PNG file containing the segmentation mask.
    Returns:
        depth: Depth map with background removed.
        depth_background: Depth map with only background.
    """
    #Read depth map
    depth = read_depth_exr(path_depth)
    #Read segmentation mask
    #segmentation_mask = read_multiple_segmentation_masks(path_segmentation_mask)
    segmentation_mask = read_segmentation_mask_env(path_segmentation_mask)
    segmentation_mask = segmentation_mask.astype('uint8')
    #depth_background = depth.copy()
    #depth_background[segmentation_mask != 0] = 0
    depth[segmentation_mask == 0] = 0
    return depth
def read_human_parts_depth(path_depth, part_segm,is_closeup=False):
    """Reads depth map from EXR file and returns a list of 25 normalized depth maps for each human part,
    based on a 3D array mask, with normalization done using the 5th and 95th percentiles to ensure a better spread of values.
    
    Args:
        path_depth: Path to the EXR file containing the depth map.
        part_segm: A 3D numpy array (25, w, h) representing masks for each part with zeros and ones.
    
    Returns:
        normalized_depth_segms: List of 25 numpy arrays, each normalized between 0 and 1 for each part,
        with non-part areas as 0, using the 5th and 95th percentile for normalization bounds.
    """
    depth = read_depth_exr(path_depth)  # Assuming this function is defined elsewhere
    if is_closeup:
        depth = np.rot90(depth, k=1, axes=(1, 0))
    normalized_depth_segms = []
    
    for i in range(23):  # Iterate over each part
        part_mask = part_segm[i].astype(bool)
        normalized_depth_map = np.zeros_like(depth)
        
        if np.any(part_mask):  # Check if there are any parts
            part_depth_values = depth[part_mask]
            # Calculate the 5th and 95th percentiles as new min and max
            p5, p95 = np.percentile(part_depth_values, [5, 95])
            
            # Avoid division by zero in case p5 == p95
            if p95 > p5:
                # Normalize values between the 5th and 95th percentiles
                normalized_values = (part_depth_values - p5) / (p95 - p5)
                # Clip values to ensure they stay within 0-1
                normalized_values = 0.8 * normalized_values + 0.2
                # Clip values to ensure they stay within 0.2-1
                normalized_values = np.clip(normalized_values, 0.2, 1)
                #normalized_values = np.clip(normalized_values, 0, 1)
                normalized_depth_map[part_mask] = 1-normalized_values

        
        normalized_depth_segms.append(normalized_depth_map.reshape(1,normalized_depth_map.shape[0],normalized_depth_map.shape[1]))
    normalized_depth_segms = np.concatenate(normalized_depth_segms, axis=0)
    #For first normalize depth keep only ones and zeros
    normalized_depth_segms[0][normalized_depth_segms[0] > 0] = 1
    #import ipdb; ipdb.set_trace()
    #cv2.imwrite('depth.png',normalized_depth_segms[0]*255)
    return normalized_depth_segms
def read_depth_percentile_norm(path_depth,mask_path,keypoints,is_closeup=False):
    """Reads depth map from EXR file and returns a list of 25 normalized depth maps for each human part,
    based on a 3D array mask, with normalization done using the 5th and 95th percentiles to ensure a better spread of values.
    
    Args:
        path_depth: Path to the EXR file containing the depth map.
        part_segm: A 3D numpy array (25, w, h) representing masks for each part with zeros and ones.
    
    Returns:
        normalized_depth_segms: List of 25 numpy arrays, each normalized between 0 and 1 for each part,
        with non-part areas as 0, using the 5th and 95th percentile for normalization bounds.
    """

    depth = read_depth_exr(path_depth)#read_human_background_depth(path_depth,mask_path)
    mask = read_body_clothing_segm_mask(mask_path, keypoints)
    depth[mask == 0] = 0
    if is_closeup:
        depth = np.rot90(depth, k=1, axes=(1, 0))
    #min-max normalization between 5th and 95th percentile but non-zero 0.1-1.0
    p5, p95 = np.percentile(depth[depth != 0], [5, 95])
    normalized_depth_map = np.zeros_like(depth)
    if p95 > p5:
        normalized_values = (depth - p5) / (p95 - p5)
        normalized_values = 0.9 * normalized_values + 0.1
        normalized_values = np.clip(normalized_values, 0, 1)

        normalized_depth_map = normalized_values
    return normalized_depth_map


def read_human_only_depth(path_depth, path_segmentation_mask, keypoint):
    """Reads depth map from EXR file and removes background.
    Args:
        path_depth: Path to the EXR file containing the depth map.
        path_segmentation_mask: Path to the PNG file containing the segmentation mask.
        keypoint: Keypoint to locate the human in the depth map.
    Returns:
        depth: Depth map with background removed.
    """
    #Read depth map
    depth = read_depth_exr(path_depth)
    #Read segmentation mask
    #segmentation_mask = read_multiple_segmentation_masks(path_segmentation_mask)
    segmentation_mask=read_body_clothing_segm_mask(path_segmentation_mask, keypoint)
    #segmentation_mask = read_segmentation_mask_env(path_segmentation_mask)
    #segmentation_mask_2 = read_segmentation_mask_env(path_segmentation_mask)
    #segmentation_mask_2 = segmentation_mask_2.astype('uint8')
    segmentation_mask = segmentation_mask.astype('uint8')
    #subplot Mask1 and Mask2 and depth[mask1] and depth[mask2]
    depth1 = depth.copy()
    depth1[segmentation_mask == 0] = 0
    
    nonzero = depth1 != 0
    if np.sum(nonzero) == 0:
        return depth1,segmentation_mask
    normalized = np.zeros_like(depth1)
    normalized[nonzero] = (depth1[nonzero]-depth1[nonzero].min())/(depth1[nonzero].max()-depth1[nonzero].min())

    #mean = np.mean(depth1[nonzero])
    #std = np.std(depth1[nonzero])
    #normalized = np.zeros_like(depth1)
    #normalized[nonzero] = (depth1[nonzero]-mean)/std
    #rescale to 0-1
    #normalized = (normalized - normalized.min())/(normalized.max()-normalized.min())
    #depth[segmentation_mask == 1] = 0

    return normalized,segmentation_mask

def imwrite_rgbd_inside_dataloader(rgb_img, depth, path,filename):
    """Writes RGB-D image to PNG file.
    Args:
        rgb: RGB image as a numpy array.
        depth: Depth map as a numpy array.
        path: Path to the output PNG file.
        filename: Name of the output PNG file.
    """
    rgbd = np.concatenate((rgb_img, depth   ), axis=0)
    #rgbd = np.transpose(rgbd.astype('float32'), (2, 0, 1))
    #Save rgbd image to file
    cv2.imwrite('test.exr' ,rgbd)

def foreground_human_depth(depth, keypoints2D):
    """Remove background from depth map.
    Args:
        depth: Depth map as a numpy array.
        keypoints2D: 2D keypoints as a numpy array.
    Returns:
        depth: Depth map with background removed.
    """
    #Locate the human in the depth map
    x = keypoints2D[:,0].astype(int)
    y = keypoints2D[:,1].astype(int)

    depth_values = depth[y,x]
    #Values significantly higher or lower than the mean are considered outliers
    mean = np.mean(depth_values)
    std = np.std(depth_values)
    depth_values = depth_values[np.abs(depth_values - mean) < 2 * std]
    return depth_values
def custom_sigmoid(x, mean, std):
    """Apply sigmoid function centered around a specific mean and adjusted by standard deviation."""
    epsilon = 1e-6
    return expit((x-mean)/(std+epsilon))

def dynamic_compress_human_depth_Sigmoid(depth, keypoints2D):
    """Dynamic range compression of the depth map.
    Args:
        depth: Depth map as a numpy array.
        keypoints2D: 2D keypoints as a numpy array.
    Returns:
        depth: Depth map with dynamic range compression.
    """
    #Locate the human in the depth map
    x = keypoints2D[:,0].astype(int)
    y = keypoints2D[:,1].astype(int)

    depth_values = depth[y,x]
    #Calculate mean and standard deviation of the depth values
    mean = np.mean(depth_values)
    std = np.std(depth_values)
    #Apply sigmoid function to the depth map
    depth_compressed = custom_sigmoid(depth, mean, std)
    return depth_compressed

def part_segm_to_vertex_colors(part_segm, n_vertices):
    # Generate 23 unique grayscale values and one extra for the background
    # The background is black
    # grayscale_values = np.linspace(0, 255, 23, dtype=np.uint8)
    grayscale_values = np.linspace(0, 0.88, 23)
    
    # Initialize the vertex colors array
    vertex_colors = np.zeros((n_vertices, 3))  # Include alpha for completeness
    
    # Assign a unique grayscale value to each part
    for part_idx, (_, vertices) in enumerate((part_segm.items())):
        # Ensure each vertex assigned to this part gets the correct grayscale value
        # and full opacity in the alpha channel
        vertex_colors[vertices, :3] = grayscale_values[part_idx+1]  # Apply grayscale value to RGB
        #vertex_colors[vertices, 3] = 255  # Full opacity
    
    return vertex_colors, grayscale_values

class SegmRenderer(object):
    def __init__(self, focal_length=600, img_w=512, img_h=512, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h, point_size=1.0)
        self.camera_center = [img_w // 2, img_h // 2]
        self.focal_length = focal_length
        self.faces = faces

    def render_part_segmentation(self, verts, part_segm):
        # Create a scene
        scene = pyrender.Scene(bg_color=np.array([0, 0, 0, 0]), ambient_light=np.ones(3) * 1)
        
        # Setup camera
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length, cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=np.eye(4))
        
        # Generate vertex colors for part segmentation
        vertex_colors = part_segm_to_vertex_colors(part_segm, verts[0].shape[0])
        # Create mesh with vertex colors

        mesh = trimesh.Trimesh(verts[0], self.faces, process=False, vertex_colors=vertex_colors)
        
        # Apply transformation if needed
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        
        # Add mesh to the scene
        scene.add(pyrender.Mesh.from_trimesh(mesh, wireframe=False))
        
        # Render the scene
        renderer = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=720, point_size=1.0)
        #color_rgba, _ = self.renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
        color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
        segmentation_map = color_rgba[:, :, 0]  # Extract the red channel as all channels are the same for grayscale
        return segmentation_map

    def delete(self):
        self.renderer.delete()