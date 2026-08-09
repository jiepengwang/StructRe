'''Borrowed from AutoSDF
'''

import re
import numpy as np
import mcubes
from termcolor import cprint

import os
import utils.utils_geometry as GeoUtils
import utils.utils_image as ImageUtils


import open3d as o3d
# from open3d.geometry.PointCloud import compute_point_cloud_distance

import torch
# import torchvision.utils as vutils

try:
    import pytorch3d
    from pytorch3d.structures import Pointclouds, Meshes

    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVOrthographicCameras, 
        PointsRasterizationSettings,
        PointsRenderer,
        PointsRasterizer,
        AlphaCompositor,
    )

    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras, 
        PointLights, 
        RasterizationSettings, 
        MeshRenderer, 
        MeshRasterizer,  
        HardPhongShader,
    )

    from pytorch3d.structures import Meshes

except (ImportError, OSError) as error:
    print(f'Failed to import PyTorch3D renderer: {error}')


# macros
# from chamferdist import ChamferDistance

# chamferDist = ChamferDistance()

# import torchvision.utils as vutils


'''Image-realted functions
'''
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def read_mesh_to_torch3D(path_mesh, use_mesh_texture=True, mesh=None, device='cuda'):

    if path_mesh is not None:
        mesh = o3d.io.read_triangle_mesh(path_mesh)
    
    p3d_mesh = convert_lis_mesh_to_p3d([mesh], use_mesh_texture=use_mesh_texture, device=device)
    return p3d_mesh, mesh

def convert_lis_mesh_to_p3d(lis_mesh, use_mesh_texture=True, device='cuda'):

    verts = []
    faces = []
    verts_rgb = []
    
    for i in range(len(lis_mesh)):
        mesh = lis_mesh[i]
        
        verts_i = np.asarray(mesh.vertices)
        verts_i = torch.from_numpy(verts_i).float().to(device).float()
        faces_i = np.asarray(mesh.triangles)
        faces_i = torch.from_numpy(faces_i.astype(np.int64)).to(device).float()

        if use_mesh_texture:
            text_i = np.asarray(mesh.vertex_colors)
            text_i = torch.from_numpy(text_i).to(device).float()

            if len(text_i)==0:
                text_i = torch.ones_like(verts_i).to(device)
        else:
            text_i = torch.ones_like(verts_i).to(device)
            
        verts.append(verts_i)
        faces.append(faces_i)
        verts_rgb.append(text_i)

    try:
        p3d_mesh = pytorch3d.structures.Meshes(verts, faces, textures=pytorch3d.renderer.Textures(verts_rgb=verts_rgb))
    except (RuntimeError, ValueError) as error:
        cprint(f'Failed to construct PyTorch3D mesh: {error}', 'yellow')
        p3d_mesh = None

    return p3d_mesh


def init_points_renderer(image_size=256, dist=1.7, elev=20, azim=20, camera='0', device='cuda:0', radius_point=0.003, points_per_pixel=1):
    # renderer
    # Initialize a camera.
    # cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

    # R, T = look_at_view_transform(1.0, 15, 60) 
    R, T = look_at_view_transform(dist, elev, azim) 

    if camera == '0':
        camera_cls = FoVPerspectiveCameras
    else:
        camera_cls = FoVOrthographicCameras

    cameras = camera_cls(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters. 
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        # radius = 0.0,
        radius=radius_point,
        # points_per_pixel = 10
        # bin_size=None,
        points_per_pixel = points_per_pixel
    )


    # Set the background color (e.g., red)  
    background_color = torch.tensor([1.0, 1.0, 1.0], device=torch.device("cpu"))  
    
    # Set up the blend parameters with the background color  
    # blend_params = BlendParams(background_color=background_color)  
    # compositor = AlphaCompositor(blend_params=blend_params) 

    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(),
        # comp=blend_params
    )

    return renderer

def init_mesh_renderer(image_size=512, dist=3.5, elev=90, azim=90, camera='0', device='cuda:0', return_cam=False, light_id = 0):
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 

    if camera == '0':
        # for vox orientation
        # dist, elev, azim = 1.7, 20, 20 # shapenet
        # dist, elev, azim = 3.5, 90, 90 # front view

        # dist, elev, azim = 3.5, 0, 135 # front view
        camera_cls = FoVPerspectiveCameras
    else:
        # dist, elev, azim = 5, 45, 135 # shapenet
        camera_cls = FoVOrthographicCameras
    
    R, T = look_at_view_transform(dist, elev, azim, device=device)
    cameras = camera_cls(znear=0.5, zfar=200.0, device=device, R=R, T=T)
    # trans_all = cameras.get_full_projection_transform()

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0, 
        faces_per_pixel=1, 
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    # lights = PointLights(device=device, location=[[1.5, 1.5, 0.5], [-1.5, 1.5, 0.5], [1.5, -1.5, 0.5], [-1.5, -1.5, 0.5]])
    lights = ((2, 3, 3),(-2, 3, -3))
    light = lights[light_id]
    lights = PointLights(device=device, location=(light,))

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    # renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(
    #         cameras=cameras, 
    #         raster_settings=raster_settings
    #     ),
    #     shader=SoftPhongShader(
    #         device=device, 
    #         cameras=cameras,
    #         lights=lights
    #     )
    # )
    renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
        )
    if return_cam:
        return renderer, cameras
    return renderer


def render_pcd(renderer, verts, color=None, normals=None):
    # verts: tensor of shape: B, V, 3
    # color: tensor of shape: B, V, 3
    
    if verts.dim() == 2:
        verts = verts[None, ...]

    if color is not None and color.dim() == 2:
        color = color[None, ...]
    
    if normals is not None and normals.dim() == 2:
        normals = normals[None, ...]
        
    verts = verts.float() 

    verts = verts.to(renderer.rasterizer.cameras.device)
    # verts = verts.cpu()

    
    # return: image tensor with shape: B, C, H, W
    V = verts.shape[1]
    B = verts.shape[0]
    features=torch.ones_like(verts).cuda()
    if color is not None:
        for i in range(3):
            features[:, :, i] = color[:, :, i]
    else:
        pass
        # features = 0.5*features
        
    pcl = Pointclouds(points=verts, features=features, normals=normals)
    images = renderer(pcl)
    # try:
    #     images = renderer(pcl)
    # except:
    #     images = renderer(pcl, gamma=(1e-4,),)
    images = images.cpu().numpy()
    images = 1.0 - images
    return images*255

def render_mesh(renderer, mesh_p3d=None, mesh_o3d=None, path_mesh=None, color=None, norm=True):
    # mesh: an open3d mesh or a list of open3d meshes
    
    # verts: tensor of shape: B, V, 3
    # return: image tensor with shape: B, H, W, C
    if path_mesh is not None:
        mesh, _ = read_mesh_to_torch3D(path_mesh)
    
    if mesh_o3d is not None:
        if not isinstance(mesh_o3d, list):
            lis_mesh_o3d = [mesh_o3d]
        else:
            lis_mesh_o3d = mesh_o3d
        mesh = convert_lis_mesh_to_p3d(lis_mesh_o3d, use_mesh_texture=True, device='cuda')

    if mesh_p3d is not None:
        mesh = mesh_p3d

    if mesh is None:
        return None
    
    if mesh.textures is None:
        verts = mesh.verts_list()
        verts_rgb_list = []
        for i in range(len(verts)):
        # print(verts.min(), verts.max())
            verts_rgb_i = torch.ones_like(verts[i])
            if color is not None:
                for i in range(3):
                    verts_rgb_i[:, i] = color[i]
            verts_rgb_list.append(verts_rgb_i)

        texture = pytorch3d.renderer.Textures(verts_rgb=verts_rgb_list)
        mesh.textures = texture

    images = renderer(mesh) # B*H*W*4
    images = images.cpu().numpy()*255
    return images[..., :3]

def sdf_to_mesh(sdf, level=0.01, color=None, render_all=False, scale_mesh=2.3):
    # device='cuda'
    device=sdf.device

    # extract meshes from sdf
    n_cell = sdf.shape[-1]
    bs, nc = sdf.shape[:2]

    if nc != 1:
        raise ValueError(
            f"Expected SDF tensor shape [batch, 1, depth, height, width]; "
            f"got {nc} channels"
        )

    nimg_to_render = bs
    if not render_all:
        if bs > 16:
            cprint('Warning! Will not return all meshes', 'red')
        nimg_to_render = min(bs, 16) # no need to render that much..

    verts = []
    faces = []
    verts_rgb = []

    for i in range(nimg_to_render):
        sdf_i = sdf[i, 0].detach().cpu().numpy()
        # verts_i, faces_i = mcubes.marching_cubes(sdf_i, 0.02)
        verts_i, faces_i = mcubes.marching_cubes(-sdf_i, -level)
        verts_i = verts_i / n_cell - .5 
        verts_i = verts_i*scale_mesh

        verts_i = torch.from_numpy(verts_i).float().to(device)
        faces_i = torch.from_numpy(faces_i.astype(np.int64)).to(device)
        text_i = torch.ones_like(verts_i).to(device)
        if color is not None:
            for i in range(3):
                text_i[:, i] = color[i]

        verts.append(verts_i)
        faces.append(faces_i)
        verts_rgb.append(text_i)

    try:
        p3d_mesh = pytorch3d.structures.Meshes(verts, faces, textures=pytorch3d.renderer.Textures(verts_rgb=verts_rgb))
    except (RuntimeError, ValueError) as error:
        cprint(f'Failed to construct PyTorch3D mesh: {error}', 'yellow')
        p3d_mesh = None

    return p3d_mesh

# rendering
def render_sdf(mesh_renderer, sdf, level=0.02, color=None, render_imsize=256, render_all=False, min_shapes_render = 8):
    """ 
        shape of sdf:
        - bs, 1, nC, nC, nC 

        return a tensor of image rendered according to self.renderer
        shape of image:
        - bs, rendered_imsize, rendered_imsize, 4

        ref: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/base_3d.py
    """
    # device='cuda'
    device = sdf.device
    bs = sdf.shape[0]

    # if not render_all:
    nimg_to_render = min(bs, min_shapes_render) # no need to render that much..
    
    p3d_mesh = sdf_to_mesh(sdf[:nimg_to_render], level=level, color=color, render_all=render_all)

    if p3d_mesh is not None:
        out = mesh_renderer(p3d_mesh)
        # rendered_im = einops.rearrange(out, 'b h w c-> b c h w').contiguous() # bs, h, w, c
        rendered_im = out.contiguous() # bs, h, w, c
    else:
        rendered_im = torch.zeros(nimg_to_render, render_imsize, render_imsize, 4).to(device)
    
    # img = tensor2im(rendered_im.data)
    img = rendered_im.data.cpu().numpy()*255

    return img[..., :3]


def render_lis_mesh(dir_mesh, path_img, renderer_mesh):
    lis_name = sorted(os.listdir(dir_mesh))[::-1]
    
    lis_imgs = []
    for item in lis_name:
        if not item.endswith('.ply'):
            continue
        print(f'Process {item}')
        
        lis_comp = list(filter(None, re.split('_|-', item[:-4])))
        n_parent, n_child = int(lis_comp[-2]), int(lis_comp[-1])
        
        path_mesh = f'{dir_mesh}/{item}'
        mesh = GeoUtils.read_triangle_mesh(path_mesh)
        if len(mesh.vertices) == 0:
            print(f'empty mesh: {path_mesh}')
            continue    
        img_curr = render_mesh(renderer_mesh, mesh_o3d=mesh)[0]
        
        img_curr = ImageUtils.add_text_on_image(img_curr.astype(np.uint8), f'Part{n_child}')
        
        
        lis_imgs.append(img_curr)
        # ImageUtils.write_image(f'{dir_img}/{item[:-4]}.png', img_curr, color_space='RGB')
    
    # os.makedirs(path_img, exist_ok=True)
    ImageUtils.write_image_lis(path_img, lis_imgs, cat_mode='horizontal', color_space='RGB', add_black_border=True)
