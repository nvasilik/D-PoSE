
import os
import trimesh
import pyrender
import numpy as np
import colorsys
import cv2

#import pyvista as pv
class Renderer(object):

    def __init__(self, focal_length=600, img_w=512, img_h=512, faces=None,
                 same_mesh_color=True):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        
        self.renderer = pyrender.OffscreenRenderer(viewport_width=1280,
                                                   viewport_height=720,
                                                   point_size=1.0)
        self.camera_center = [img_w // 2, img_h // 2]
        self.focal_length = focal_length
        self.faces = faces
        self.same_mesh_color = True#same_mesh_color
        self.camera = pyrender.camera.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                                         cx=self.camera_center[0], cy=self.camera_center[1])
        self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self.light_pose_neg = trimesh.transformations.rotation_matrix(np.radians(4-5), [1, 0, 0])
        self.light_pose_pos = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        self.rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        self.scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=np.ones(3) * 0)
        self.scene.add(self.camera, pose=np.eye(4))
        self.scene.add(self.light, pose=self.light_pose_neg)
        self.scene.add(self.light, pose=self.light_pose_pos)



    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(0, 0, 0, 0)):


        common_color = colorsys.hsv_to_rgb(0.6, 0.5, 1.0)
        mesh_list = []
        for n in range(len(verts)):
            m = trimesh.Trimesh(verts[n], self.faces, process=False)
            m.apply_transform(self.rot)
            mesh_list.append(m)
        # Merge all meshes into a single mesh
        merged_mesh = trimesh.util.concatenate(mesh_list)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=common_color)
        pyrender_mesh = pyrender.Mesh.from_trimesh(merged_mesh, material=material, wireframe=False,smooth=False)
        node=self.scene.add(pyrender_mesh, name='mesh')



        color_rgba, depth_map = self.renderer.render(self.scene,flags=pyrender.RenderFlags.OFFSCREEN)
        self.scene.remove_node(node)
        color_rgb = color_rgba[:, :, :3]
        mask = depth_map > 0
        bg_img_rgb[mask] = color_rgb[mask]
        return bg_img_rgb

    def render_side_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        side_view = self.render_front_view(pred_vert_arr_side)
        return side_view

    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()
