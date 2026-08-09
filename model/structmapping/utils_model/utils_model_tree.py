import numpy as np

import utils.utils_geometry as GeoUtils
import utils.utils_io as IOUtils
import model.structmapping.tree as TreeUtils
import utils.utils_image as ImageUtils


from termcolor import cprint

def construct_lis_tree_nodes(dict_out_shape, curr_level_hier):
    
    lis_tree_nodes = []
    for i in range(len(dict_out_shape['lis_mesh'])):
        is_leaf = True
        bb_parent = dict_out_shape['lis_bbox_centerlwh'][i]
        code_parent = None #dict_nodes_code[parent_id]
        
        if 'lis_img_part' in dict_out_shape:
            mesh_render = dict_out_shape['lis_img_part'][i]
        else:
            mesh_render = None
            
        mesh_norm_parent = dict_out_shape['lis_mesh'][i]
        try:
            points_sample, _ = GeoUtils.sample_points_on_mesh(mesh_norm_parent, n_sample=1024, return_points=True)
        except Exception as error:
            points_sample = np.ones((1024, 3))
            print(f'No mesh here: {error}')
        
        code_noise = None
        if 'lis_code_noise' in dict_out_shape:
            code_noise = dict_out_shape['lis_code_noise'][i]

        verts, faces = None, None
        node_curr_parent = TreeUtils.Node(part_id=f'{curr_level_hier:02d}_{i:02d}', is_leaf=is_leaf, box=bb_parent, 
                                        label=None, children=None, 
                                        edges=None, full_label=None, 
                                        mesh_v=verts, mesh_f=faces,
                                        geo=points_sample, geo_feat=code_parent, geo_render=mesh_render,
                                        code_noise=code_noise)
        lis_tree_nodes.append(node_curr_parent)
    return lis_tree_nodes

def construct_local_tree_pair(dict_info_shape_prev, dict_info_shape_curr, mapping_direction_infer):
    
    lis_nodes_prev = dict_info_shape_prev['lis_tree_nodes']
    lis_nodes_curr = dict_info_shape_curr['lis_tree_nodes']
    
    dict_info_link = dict_info_shape_curr['dick_link_cp']
    if mapping_direction_infer==0:
        dict_link_nodes = dict_info_link['dict_p2c']
        TreeUtils.link_tree_nodes(lis_nodes_prev, lis_nodes_curr, dict_link_nodes)
        
    elif mapping_direction_infer==1:
        dict_link_nodes = dict_info_link['dict_c2p']
        TreeUtils.link_tree_nodes(lis_nodes_curr, lis_nodes_prev, dict_link_nodes)


def update_tree_hierarchy(
    dict_hierarchy_levels,
    i_level_hier,
    data_out_tmp,
    dict_out_shape,
    mapping_direction_infer,
):
    lis_tree_nodes = construct_lis_tree_nodes(dict_out_shape, curr_level_hier=i_level_hier)
    dict_out_shape['lis_tree_nodes'] = lis_tree_nodes

    dict_hierarchy_levels[f'level_{i_level_hier}'] = dict_out_shape
    
    if i_level_hier==0 and mapping_direction_infer==0:
        return dict_hierarchy_levels
    else:
        # build pointer
        dick_link_cp = data_out_tmp['dick_link_cp']
        dict_hierarchy_levels[f'level_{i_level_hier}']['dick_link_cp'] = dick_link_cp
        
        if len(dict_hierarchy_levels)==1:
            return dict_hierarchy_levels
        
        # construct local tree
        construct_local_tree_pair(
            dict_hierarchy_levels[f'level_{i_level_hier-1}'],
            dict_hierarchy_levels[f'level_{i_level_hier}'],
            mapping_direction_infer
        )

    return dict_hierarchy_levels

def merge_root_tree(dict_info_tree):
    
    
    curr_level_hier = len(dict_info_tree)
    curr_level_hier_last = curr_level_hier -1
    
    lis_tree_nodes_final = dict_info_tree[f'level_{curr_level_hier_last}']['lis_tree_nodes']
    n_node_final = len(lis_tree_nodes_final)
    if n_node_final==1:
        return dict_info_tree
    
    cprint(f'[Rewriting] Merge root tree ...\n', 'yellow')

    # Construct root node
    mesh_norm_parent = dict_info_tree[f'level_{curr_level_hier-1}']['mesh_merge']
    min_bound, max_bound, _ = GeoUtils.get_aabb(mesh_norm_parent)
    bb_parent = np.concatenate([min_bound, max_bound])[None, ...]
    bb_parent = GeoUtils.get_lwh_from_aabb(bb_parent)
    
    try:
        points_sample, _ = GeoUtils.sample_points_on_mesh(mesh_norm_parent, n_sample=1024, return_points=True)
    except Exception as error:
        points_sample = np.ones((1024, 3))
        print(f'No mesh here: {error}')
    
    code_noise = None
    verts, faces = None, None
    code_parent, mesh_render = None, None
    i = 0
    node_curr_parent = TreeUtils.Node(part_id=f'{curr_level_hier:02d}_{i:02d}', is_leaf=False, box=bb_parent, 
                                    label=None, children=None, 
                                    edges=None, full_label=None, 
                                    mesh_v=verts, mesh_f=faces,
                                    geo=points_sample, geo_feat=code_parent, geo_render=mesh_render,
                                    code_noise=code_noise)
    
    
    # add child node
    lis_tree_nodes_child = dict_info_tree[f'level_{curr_level_hier-1}']['lis_tree_nodes']
    for i in range(len(lis_tree_nodes_child)):
        node_curr_parent.children.append(lis_tree_nodes_child[i])

    dict_info_tree[f'level_{curr_level_hier}'] = {}
    dict_info_tree[f'level_{curr_level_hier}']['lis_tree_nodes'] = [node_curr_parent]
    
    return dict_info_tree


def dump_tree(path_tree, dict_info_tree, mapping_direction_infer,
                    path_img_hier, dir_mesh_merge, 
                    should_dump_tree = False,
                    should_save_images=False,
                    should_save_mesh=False):
    
    # save tree
    if should_dump_tree:
        if mapping_direction_infer == 1:
            tree_pred = dict_info_tree['level_-1']['lis_tree_nodes'][0]
            
            IOUtils.save_class_instance_to_pickle(path_tree, tree_pred)
        else:
            n_levels = len(dict_info_tree)
            tree_pred = dict_info_tree[f'level_{n_levels-1}']['lis_tree_nodes'][0]
            IOUtils.save_class_instance_to_pickle(path_tree, tree_pred)

    if should_save_images:
        save_images_render(path_img_hier, dict_info_tree=dict_info_tree)

    if should_save_mesh:
        save_mesh_merged_parts(dir_mesh_merge, dict_info_tree=dict_info_tree)
        
        

def save_images_render(path_img_hier, dict_info_tree):       
    # save images
    n_levels = len(dict_info_tree)
    lis_images_merge = []
    for i in range(-1, n_levels):
        if f'level_{i}' in dict_info_tree:

            if len(lis_images_merge)==0:
                img_gt = dict_info_tree[f'level_{i}']['img_gt']
                lis_images_merge.append(img_gt)
            
            if 'lis_img_part' not in dict_info_tree[f'level_{i}']:
                continue
            
            n_parts = len(dict_info_tree[f'level_{i}']['lis_img_part'])
            img_part_merge = dict_info_tree[f'level_{i}']['img_part_merge']  
            
            img_part_merge = ImageUtils.add_text_on_image(img_part_merge.astype(np.uint8), f'Hier{i}-Part{n_parts}')
            lis_images_merge.append(img_part_merge)
            
    ImageUtils.write_image_lis(path_img_hier, lis_images_merge, color_space='RGB',
                                add_black_border=True, interval_img=20)


def save_mesh_merged_parts(dir_mesh_part, dict_info_tree):       
    # save images
    n_levels = len(dict_info_tree)
    for i in range(-1, n_levels):
        if f'level_{i}' in dict_info_tree:

            if 'mesh_merge' in dict_info_tree[f'level_{i}']:
                mesh_merge = dict_info_tree[f'level_{i}']['mesh_merge']
                
                path_mesh_merge = f'{dir_mesh_part}/level_{i}.ply'
                GeoUtils.write_triangle_mesh(path_mesh_merge, mesh_merge)
            
