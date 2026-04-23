import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors


def adjust_mesh_faces(mesh, target_face_count=10000):
    current_face_count = len(mesh.faces)

    # 1. 如果面片数量高于10000，则随机选择10000个面
    if current_face_count > target_face_count:
    mesh = mesh.simplify_quadric_decimation(target_face_count)
    # 2. 如果面片数量低于10000，则通过细分网格增加面片数
    if current_face_count < target_face_count:
        # 增加更多面片
        while len(mesh.faces) < target_face_count:
            mesh = mesh.subdivide()  # 增加面片数
        if len(mesh.faces) > target_face_count:
            mesh = mesh.simplify_quadratic_decimation(target_face_count)

    return mesh


def find_closest_faces(patch_mesh, ori_mesh, writh_file):
    # 获取每个网格的面中心（质心）
    sam_c = patch_mesh.triangles_center  # 获取patch_mesh的面中心
    ori_c = ori_mesh.triangles_center  # 获取ori_mesh的面中心
    # 只考虑ori_mesh的前9998个面
    ori_c = ori_c[:9998]  # 限制ori_mesh的面为前9998个

    # 使用NearestNeighbors查找最近邻面
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ori_c)
    _, indices = nbrs.kneighbors(sam_c)
    # 转换为整数并返回
    indices = indices.astype(int).flatten()
    # print(len(indices), max(indices))
    np.save(writh_file, indices)
