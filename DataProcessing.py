from CreateGraph import *


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


def CMF(X_HVG, lambda1, lambda2, lambdaC, lambdaG):
    X_copy=X_HVG.copy()
    m, n = X_copy.shape
    r = int(n / 10)
    print(f'Start calculating cell similarity and gene similarity........')
    # GENE SIMILARITY
    G = similary_metrix(X_copy.T)
    # CELL SIMILARITY
    C = similary_metrix(X_copy)
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
    similarity = np.matmul(H, W.T)
    print('impute starting,wait for a moment......')
    similarity[similarity < 0] = 0
    for i in range(m):
        for j in range(n):
            if X_impute[i, j] == 0:
                X_impute[i, j] = similarity[i, j]
                print(f'X_impute[{i},{j}]={X_impute[i,j]}')
    print('impute endding............')
    return X_impute

