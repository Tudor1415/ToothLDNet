import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors


def adjust_mesh_faces(mesh, target_face_count=10000):
    current_face_count = len(mesh.faces)

    if current_face_count > target_face_count:
        mesh = mesh.simplify_quadric_decimation(face_count=target_face_count)

    if current_face_count < target_face_count:
        while len(mesh.faces) < target_face_count:
            mesh = mesh.subdivide()
        if len(mesh.faces) > target_face_count:
            mesh = mesh.simplify_quadric_decimation(face_count=target_face_count)

    return mesh


def find_closest_faces(patch_mesh, ori_mesh, writh_file):
    sam_c = patch_mesh.triangles_center
    ori_c = ori_mesh.triangles_center
    ori_c = ori_c[:9998]

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(ori_c)
    _, indices = nbrs.kneighbors(sam_c)
    indices = indices.astype(int).flatten()
    np.save(writh_file, indices)
