import torch
import numpy as np
import os
import trimesh
from pl_model_gnn import LitModel
from data.gnn.common import calc_features
from utils.gnn.cluster import Cluster
from scripts.graph_cut import graph_cut


class Predictor:
    def __init__(self, weight_dir):
        weights = os.path.join(weight_dir, 'checkpoints/last.ckpt')
        self.teethgnn = LitModel.load_from_checkpoint(weights).cuda()
        self.args = self.teethgnn.hparams.args
        self.teethgnn.eval()
        self.cluster = Cluster()

    @torch.no_grad()
    def infer(self, vs, ts):
        fea = calc_features(vs, ts)     # (15, n)
        pred, _, offsets = self.teethgnn(fea[None])
        pred = pred[0].softmax(0).cpu().detach().T.numpy() # (n, 17)
        offsets = offsets[0].cpu().detach().T.numpy()   # (n, 3)
        return pred, offsets

    def get_k_nearest(self, pts, pts_ref, k=5):
        """
        :param pts: (n1, 3)
        :param pts_ref: (n2, 3)
        :return: idx: (n1, k)
        """
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k).fit(pts_ref)
        _, idx = nbrs.kneighbors(pts)
        return idx

    def run(self, m:trimesh.Trimesh, sample_num, sample_times=0):
        """
        :param m:
        :param sample_num:
        :param sample_times:  0 indicating sampling all faces.
        :return:
        """
        num_faces = len(m.faces)
        if sample_times == 0:
            sample_times = 1000
        sample_times = min(sample_times, (num_faces+sample_num-1)//sample_num)

        perm_idx = np.random.permutation(num_faces)
        if sample_times * sample_num > num_faces:
            sampled_idx = np.concatenate([perm_idx, perm_idx[:sample_num - len(perm_idx) % sample_num]])
        else:
            sampled_idx = perm_idx[:sample_num * sample_times]

        all_vs = torch.tensor(m.vertices, dtype=torch.float).cuda()
        all_ts = torch.tensor(m.faces, dtype=torch.long).cuda()
        all_vs = all_vs - all_vs.mean(0)  # preprocess

        # 1. network inference
        all_pred = np.zeros((num_faces, 17))
        all_offsets = np.zeros((num_faces, 3))

        for idx in np.split(sampled_idx, len(sampled_idx)//sample_num):
            pred, offsets = self.infer(all_vs, all_ts[idx])
            all_pred[idx] = pred
            all_offsets[idx] = offsets

        # 2. clustering
        face_centers = m.triangles_center
        probs = self.cluster.cluster(all_pred, all_offsets, face_centers)

        if len(sampled_idx) < num_faces:
            # knn mapping
            unsampled_idx = list(set(range(num_faces)) - set(sampled_idx.tolist()))        # (n1, )
            idx = self.get_k_nearest(face_centers[unsampled_idx], face_centers[sampled_idx], k=5)        # (n1, k)
            probs[unsampled_idx] = probs[sampled_idx][idx].mean(1)       # (n1, 17)

        # 3. Graph cut, optimization for scattered faces

        # 4. Fuzzy clustering

        # 5. Boundary smoothing: smoothing jagged tooth boundaries

        labels = probs.argmax(-1)       # (n,

        return labels

    def infer_with_sampling(self, m:trimesh.Trimesh, sample_num, sample_times=0):
        """
        :param m:
        :param sample_num:
        :param sample_times:  0 indicating sampling all faces.
        :return: probs
                 offsets
        """
        num_faces = len(m.faces)
        if sample_times == 0:
            sample_times = 1000
        sample_times = min(sample_times, (num_faces+sample_num-1)//sample_num)

        perm_idx = np.random.permutation(num_faces)
        if sample_times * sample_num > num_faces:
            sampled_idx = np.concatenate([perm_idx, perm_idx[:sample_num - len(perm_idx) % sample_num]])
        else:
            sampled_idx = perm_idx[:sample_num * sample_times]

        all_vs = torch.tensor(m.vertices, dtype=torch.float).cuda()
        all_ts = torch.tensor(m.faces, dtype=torch.long).cuda()
        all_vs = all_vs - all_vs.mean(0)  # preprocess

        # 1. network inference
        all_pred = np.zeros((num_faces, 17))
        all_offsets = np.zeros((num_faces, 3))

        for idx in np.split(sampled_idx, len(sampled_idx)//sample_num):
            pred, offsets = self.infer(all_vs, all_ts[idx])
            all_pred[idx] = pred
            all_offsets[idx] = offsets

        face_centers = m.triangles_center
        if len(sampled_idx) < num_faces:
            # knn mapping
            unsampled_idx = list(set(range(num_faces)) - set(sampled_idx.tolist()))        # (n1, )
            idx = self.get_k_nearest(face_centers[unsampled_idx], face_centers[sampled_idx], k=5)        # (n1, k)
            all_pred[unsampled_idx] = all_pred[sampled_idx][idx].mean(1)       # (n1, 17)

        return all_pred, all_offsets


def gnn_run(mesh_sim, weight_dir):
    model = LitModel.load_from_checkpoint(weight_dir).cuda()
    model.eval()
    args = model.hparams.args

    vs, fs = mesh_sim.vertices, mesh_sim.faces
    fids = np.arange(args.num_points)
    fs = fs[fids]
    mesh_sim = trimesh.Trimesh(vertices=vs, faces=fs)

    vs = torch.tensor(vs, dtype=torch.float32)
    fs = torch.tensor(fs, dtype=torch.long)
    features = calc_features(vs, fs)  # (15, nf)
    features = features.unsqueeze(0).cuda()

    # pred
    with torch.no_grad():
        p_labels = model.infer(features)  # M,N

    all_labels = []
    p_labels = resolve_duplicates(p_labels, 0.3)
    for i in range(len(p_labels)):
        labels_sim = p_labels[i]
        if sum(labels_sim) == 0:
            continue
        # labels_sim = graph_cut(fs, mesh_sim.triangles_center, mesh_sim.face_normals, labels_sim)
        labels_sim = np.squeeze(labels_sim)

        if sum(labels_sim) >= 120:
            all_labels.append(labels_sim)
    all_labels = np.array(all_labels)
    return all_labels, mesh_sim


def compute_overlap_with_self(label1, label2):
    """
    计算两个标签之间的交集占各自标签1的比例。
    :param label1: 牙齿模型1的标签 (1维数组)
    :param label2: 牙齿模型2的标签 (1维数组)
    :return: 交集占牙齿1和牙齿2自身1点比例的元组 (overlap1, overlap2)
    """
    intersection = np.sum(np.logical_and(label1, label2))  # 计算交集
    overlap1 = intersection / np.sum(label1) if np.sum(label1) > 0 else 0  # 计算牙齿1的重叠比例
    overlap2 = intersection / np.sum(label2) if np.sum(label2) > 0 else 0  # 计算牙齿2的重叠比例
    return overlap1, overlap2


def resolve_duplicates(labels, threshold=0.5):
    """
    处理重复的牙齿模型，超过重复率阈值的较小模型会归零重复点
    :param labels: K * N 矩阵，表示K个牙齿模型的标签，每个模型有N个点
    :param threshold: 重复率阈值，超过该值的牙齿模型会被认为是重复的
    :return: 处理后的标签矩阵
    """
    K, N = labels.shape
    labels_copy = labels.copy()  # 避免直接修改原始数据

    for i in range(K):
        for j in range(i + 1, K):
            # 计算牙齿i和牙齿j的交集与各自牙齿1点数量的比例
            overlap_i, overlap_j = compute_overlap_with_self(labels[i], labels[j])

            if overlap_i > threshold or overlap_j > threshold:
                # 找到重复的点
                common_points = np.logical_and(labels[i], labels[j])

                # 将重复的点归零，较小的牙齿模型
                if np.sum(labels[i]) < np.sum(labels[j]):
                    labels_copy[i][common_points] = 0
                else:
                    labels_copy[j][common_points] = 0

    return labels_copy
