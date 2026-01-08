# Import
import numpy as np
import pandas as pd

# Define functions
def make_random_projection_matrix(K: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """
    K차원 -> d차원으로 줄이는 랜덤 투영 행렬 R (shape: K x d)
    - gaussian: N(0, 1)/sqrt(d)
    """
    R = rng.normal(0.0, 1.0, size=(K, d)) / np.sqrt(d)
    return R

# Define GPC Encoder (Geoemetry Preserving Category Encoder)
def gpc_encoder(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list[str], d: int = 16, r: float = 1.0, seed: int = 42,) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    train 기준으로 각 cat_col의 'category -> vector' 매핑을 만들고
    train/test 모두 동일 매핑으로 변환한 뒤, d차원 벡터를 컬럼으로 펼쳐 반환.
    """
    rng = np.random.default_rng(seed)

    Xtr_num = X_train.drop(columns=cat_cols)
    Xte_num = X_test.drop(columns=cat_cols)

    tr_blocks = [Xtr_num]
    te_blocks = [Xte_num]

    for col in cat_cols:
        # (중요) 결측을 하나의 범주로 처리해 매핑 일관성 유지
        tr = X_train[col].astype("object")
        te = X_test[col].astype("object")

        # 4.1 (train 기준) 범주 목록 확정
        cats = pd.unique(tr)
        K = len(cats)
        if K < 2:
            # 범주가 1개면 분리 정보가 없으니 상수 벡터로 처리
            const_vec = np.zeros(d, dtype=float)
            const_vec[0] = r
            mapping = {cats[0]: const_vec}
            fallback = const_vec
        else:
            # 4.1 e_i: K차원 표준기저를 직접 만들 필요는 없지만,
            #      수식 흐름을 그대로 반영하기 위해 I(K)를 사용한다.
            I = np.eye(K, dtype=float)                   # e_i 들이 행(row)로 들어있다고 보면 됨
            ones = np.ones((K,), dtype=float) / K        # (1/K)*1 벡터

            # 4.2 u_i: 중심을 원점으로(평행이동) -> u_i = e_i - (1/K)1
            U = I - ones[None, :]                        # shape (K, K)

            # 4.3 v_i: 길이를 r로 맞추는 스케일
            #    ||u_i|| = sqrt((K-1)/K) 이므로, v_i = r * sqrt(K/(K-1)) * u_i
            scale = r * np.sqrt(K / (K - 1))
            V = scale * U                                 # shape (K, K), 각 row가 v_i

            # 4.4 Random Projection: K -> d
            #     (컬럼별로 다른 R을 쓰고 싶으면 col별 seed를 섞음)
            col_seed = int(rng.integers(0, 2**31 - 1))
            R = make_random_projection_matrix(K=K, d=d, rng=np.random.default_rng(col_seed))
            Z = V @ R                                     # shape (K, d)

            # 5.4 투영 후 다시 길이를 r로 맞춤(정규화)
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            Z = (Z / norms) * r

            # 5.5 category -> vector 매핑 테이블 생성
            mapping = {cat: Z[i] for i, cat in enumerate(cats)}

            # unseen category 처리용 fallback (중립 벡터: 평균 후 정규화)
            mean_vec = Z.mean(axis=0)
            mv = np.linalg.norm(mean_vec)
            if mv < 1e-12:
                tmp = np.random.default_rng(col_seed).normal(size=d)
                mean_vec = (tmp / np.linalg.norm(tmp)) * r
            else:
                mean_vec = (mean_vec / mv) * r
            fallback = mean_vec

        # (train/test) 변환: 값 -> 벡터 -> 컬럼 펼치기
        def to_matrix(series: pd.Series) -> np.ndarray:
            out = np.zeros((len(series), d), dtype=float)
            for i, v in enumerate(series):
                out[i] = mapping.get(v, fallback)
            return out

        Ztr = to_matrix(tr)
        Zte = to_matrix(te)

        new_cols = [f"{col}__gpce_{j}" for j in range(d)]

        tr_blocks.append(pd.DataFrame(Ztr, columns=new_cols, index=X_train.index))
        te_blocks.append(pd.DataFrame(Zte, columns=new_cols, index=X_test.index))

    Xtr_out = pd.concat(tr_blocks, axis=1)
    Xte_out = pd.concat(te_blocks, axis=1)

    return Xtr_out, Xte_out