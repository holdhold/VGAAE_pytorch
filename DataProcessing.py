from CreateGraph import *


# 得到相似性矩阵S
def similary_metrix(X_hvg):
    m, n = X_hvg.shape
    distance = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            distance[i, j] = np.sqrt(np.dot((X_hvg[i] - X_hvg[j]), (X_hvg[i] - X_hvg[j]).T))
    similarity = np.zeros((m, m))
    for p in range(m):
        for q in range(m):
            similarity[p, q] = 1. / (distance[p, q] + 1)
    return similarity


# CMF--协作矩阵分解
def CMF(X_HVG, lambda1, lambda2, lambdaC, lambdaG):
    # 不能在原数据上直接修改，所以需要复制操作
    X_copy=X_HVG.copy()
    m, n = X_copy.shape
    r = int(n / 10)
    print(f'Start calculating cell similarity and gene similarity........')
    # GENE SIMILARITY
    G = similary_metrix(X_copy.T)
    # CELL SIMILARITY
    C = similary_metrix(X_copy)
    # 初始化插补矩阵，后续只需插补丢失值--0即可
    X_impute = X_copy
    _, p = C.shape
    _, q = G.shape
    print(f'p={p},q={q}')
    H = np.random.rand(p, r)
    W = np.random.rand(q, r)
    I = np.eye(r)
    k = 1
    err = 1e-5
    insweep = 2000
    print(f'Start calculating the imputation matrix......')
    while k < insweep:
        k = k + 1
        H = np.dot((np.dot(X_copy, W) + lambdaC * np.dot(C, H)),
                   np.linalg.inv(np.dot(W.T, W) + lambda1 * I + lambdaC * np.dot(H.T, H)))

        W = np.dot((np.dot(X_copy.T, H) + lambdaG * np.dot(G, W)),
                   np.linalg.inv(np.dot(H.T, H) + lambda2 * I + lambdaG * np.dot(W.T, W)))

        error = np.mean(np.mean(np.abs(X_copy - np.dot(H, W.T)), axis=1)) / np.mean(np.mean(X_copy, axis=1))
        if error < err:
            break
        print(f'k--{k}:error--{error}')
    similarity = np.matmul(H, W.T)
    # <class 'anndata._core.views.ArrayView'>
    print(type(similarity))
    print('impute starting,wait for a moment......')
    # 将similarity中小于0的元素置为0
    similarity[similarity < 0] = 0
    for i in range(m):
        for j in range(n):
            print(f'X_impute[{i},{j}]={X_impute[i,j]},similarity[{i},{j}]={similarity[i,j]}')
            if X_impute[i, j] == 0:
                X_impute[i, j] = similarity[i, j]
                print(f'X_impute[{i},{j}]={X_impute[i,j]}')
        print(f'第{i}个细胞的丢失值插补完成')
    print('impute endding............')
    return X_impute

