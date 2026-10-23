"""RAPTOR 聚类的离线单测（纯函数，无网络）。"""
from rag.raptor import cosine, greedy_cluster


def test_cosine():
    assert abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(cosine([1, 0], [0, 1])) < 1e-9  # 正交
    assert cosine([], [1]) == 0.0  # 空向量


def test_greedy_cluster_groups_similar():
    vecs = [[1, 0], [0.9, 0.1], [0, 1]]
    clusters = greedy_cluster(vecs, threshold=0.8)
    # 前两个相似应聚成一簇，第三个独立成簇
    assert sorted(clusters) == sorted([[0, 1], [2]])


def test_greedy_cluster_all_separate_when_threshold_high():
    vecs = [[1, 0], [0, 1], [0.5, 0.5]]
    clusters = greedy_cluster(vecs, threshold=1.01)  # 阈值超过最大相似度 → 全部分离
    assert sorted(clusters) == [[0], [1], [2]]
