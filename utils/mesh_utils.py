import open3d as o3d
import numpy as np
from scipy.interpolate import griddata
import os
from psbody.mesh import Mesh
import scipy.sparse as sp

def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))


def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v),len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv

def extract_texture(mesh_path, resolution=256, is_interpolate_position=False, is_interpolate_normal=True, force=False): 
    npz_path = mesh_path.replace(".obj", ".npz")
    if not os.path.exists(npz_path) or force:
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_path))
        mesh.compute_vertex_normals()
        mesh.compute_uvatlas(size=resolution, max_stretch=1.0, gutter=1.0)
        normals = mesh.vertex.normals.numpy()
        
        uv_coordinates = mesh.triangle.texture_uvs.numpy().reshape(-1, 2)
        vertices = mesh.vertex.positions.numpy()
        triangles = mesh.triangle.indices.numpy().reshape(-1)
        np.savez(npz_path, normals=normals, uv_coordinates=uv_coordinates, vertices=vertices, triangles=triangles)
    else:
        npz = np.load(npz_path)
        normals = npz["normals"]
        uv_coordinates = npz["uv_coordinates"]
        vertices = npz["vertices"]
        triangles = npz["triangles"]

    # mesh = o3d.io.read_triangle_mesh(mesh_path)
    # normals = np.asarray(mesh.vertex_normals)
    # vertices = np.asarray(mesh.vertices)
    # triangles = np.asarray(mesh.triangles).reshape(-1)
    # uv_coordinates = np.asarray(mesh.triangle_uvs)
    print("Ver num:", vertices.shape)
    uv_coordinates_ = np.zeros((vertices.shape[0], 2))
    for triangle_ind, uv_coord in zip(triangles, uv_coordinates):
        uv_coordinates_[triangle_ind] = uv_coord
    uv_coordinates = uv_coordinates_
    
    uv_mapped = (uv_coordinates[:, :2] * (resolution-1)).astype(int)
    uv_grid = np.mgrid[0:resolution, 0:resolution].T.reshape(-1, 2)
    if is_interpolate_position:
        uv_color = np.concatenate([uv_coordinates, np.zeros((uv_coordinates.shape[0], 1), dtype=np.float32)], axis=1)
        
        position_texture_values = griddata(uv_mapped, uv_color, uv_grid, method='linear', fill_value=0.0)
        # position_texture_values = position_texture_values * 2 - 1
        position_texture = position_texture_values.reshape(resolution, resolution, 3)
    else:
        position_texture = np.zeros((resolution, resolution, 3), dtype=np.float32)
        uv_color = uv_coordinates
        uv_color = np.concatenate([uv_color, np.zeros((uv_color.shape[0], 1), dtype=np.float32)], axis=1)
        uv_color = uv_color * 2 - 1
        position_texture[uv_mapped[:, 1], uv_mapped[:, 0], :] = uv_color

    if is_interpolate_normal:
        normals_mapped = griddata(uv_mapped, normals, uv_grid, method='linear', fill_value=0.0)
        normals_mapped = (normals_mapped + 1) / 2.0
        normal_texture = normals_mapped.reshape(resolution, resolution, 3)
    else:
        normal_texture = np.zeros((resolution, resolution, 3), dtype=np.float32)
        normal_texture[uv_mapped[:, 1], uv_mapped[:, 0], :] = normals

    map_3D = np.zeros((resolution, resolution, 3), dtype=np.float32)
    
    map_3D[uv_mapped[:, 1], uv_mapped[:, 0], :] = vertices

    available_index = np.unique(uv_mapped, axis=0)


    return position_texture, normal_texture, available_index, map_3D

def extract_graph(mesh_path, force=False): 
    npz_path = mesh_path.replace(".obj", ".npz")
    npz_path = npz_path.replace(".ply", ".npz")
    if not os.path.exists(npz_path) or force:
        mesh = Mesh(filename=mesh_path)
        vertices = mesh.v.astype(np.float32)
        normals = mesh.estimate_vertex_normals().astype(np.float32)
        adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
        edge_index = np.vstack((adjacency.row, adjacency.col)).astype(np.float32)
        vertice_feature = np.concatenate([vertices, normals], axis=-1)

        # mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_path))
        # mesh.compute_vertex_normals()
        # vertices = mesh.vertex.positions.numpy()
        # normals = mesh.vertex.normals.numpy()
        # triangles = mesh.triangle.indices.numpy()
        # vertice_feature = np.concatenate([vertices, normals], axis=-1)
        
        # edge_index = [[],[]]
        # for triangle in triangles:
        #     edge_index[0].append(triangle[0])
        #     edge_index[1].append(triangle[1])
        #     edge_index[0].append(triangle[1])
        #     edge_index[1].append(triangle[0])

        #     edge_index[0].append(triangle[0])
        #     edge_index[1].append(triangle[2])
        #     edge_index[0].append(triangle[2])
        #     edge_index[1].append(triangle[0])

        #     edge_index[0].append(triangle[2])
        #     edge_index[1].append(triangle[1])
        #     edge_index[0].append(triangle[1])
        #     edge_index[1].append(triangle[2])
        # edge_index = np.array(edge_index)

        np.savez(npz_path, vertice_feature=vertice_feature, edge_index=edge_index)
    else:
        npz = np.load(npz_path)
        vertice_feature = npz["vertice_feature"]
        edge_index = npz["edge_index"]
    
    return vertice_feature, edge_index

# test
if __name__ == '__main__':
    # extract_texture("/data/wxb/face/gaussian-head/insta-dataset/justin/meshes/00000.obj",is_interpolate_position=True, is_interpolate_normal=False)
    # position_texture, normal_texture, available_index, map_3D = extract_texture("/data/wxb/face/gaussian-head/blendshape_template/render.obj",is_interpolate_position=False, is_interpolate_normal=False)
    
    # position_texture, normal_texture, available_index, map_3D = extract_texture("/data/wxb/face/gaussian-head/insta-dataset/justin/meshes/00000.obj",resolution=256, \
    #                                                                             is_interpolate_position=False, is_interpolate_normal=True, force=True)

    # import cv2
    # cv2.imwrite("./normal_0.png", ((normal_texture + 1) / 2.0 * 255.0).astype(np.uint8))
    # cv2.imwrite("./pos_0.png", ((position_texture + 1) / 2.0 * 255.0).astype(np.uint8))
    # print("pos max:", position_texture.max(), " pos min:", position_texture.min(), " nor max:", normal_texture.max(), " nor min:", normal_texture.min())
    # print("available_index num:", available_index.shape)

    # position_texture, normal_texture, available_index, map_3D = extract_texture("/data/wxb/face/gaussian-head/insta-dataset/justin/meshes/00001.obj",resolution=256, \
    #                                                                             is_interpolate_position=False, is_interpolate_normal=True, force=True)

    # import cv2
    # cv2.imwrite("./normal_1.png", ((normal_texture + 1) / 2.0 * 255.0).astype(np.uint8))
    # cv2.imwrite("./pos_1.png", ((position_texture + 1) / 2.0 * 255.0).astype(np.uint8))
    # print("pos max:", position_texture.max(), " pos min:", position_texture.min(), " nor max:", normal_texture.max(), " nor min:", normal_texture.min())
    # print("available_index num:", available_index.shape)

    vertice_feature, edge_index = extract_graph("/data/wxb/face/gaussian-head/insta-dataset/justin/meshes/00000.obj", force=True)
    print(vertice_feature.shape)
    print(edge_index.shape)