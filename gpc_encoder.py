# Import
import numpy as np
import pandas as pd

# Define functions
def make_random_projection_matrix(K: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Create a random projection matrix R of shape (K, d) to project
    K-dimensional vectors into d-dimensional space.

    The matrix entries are sampled from a Gaussian distribution:
        N(0, 1) / sqrt(d)

    Parameters
    ----------
    K : int
        Original dimensionality (number of categories).
    d : int
        Target embedding dimension.
    rng : np.random.Generator
        NumPy random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Random projection matrix of shape (K, d).
    """
    R = rng.normal(0.0, 1.0, size=(K, d)) / np.sqrt(d)
    return R

# Define GPC Encoder (Geoetry Preserving Category Encoder)
def gpc_encoder(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list[str], d: int = 16, r: float = 1.0, seed: int = 42,) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Geometry Preserving Category Encoder (GPCE).

    For each categorical column:
    - Categories are mapped (based on TRAIN data only) to vectors
      lying on a hypersphere of radius r.
    - Category geometry is preserved by:
        1) Centering one-hot vectors
        2) Scaling to fixed norm
        3) Applying random projection
        4) Re-normalizing after projection
    - The same mapping is applied to both train and test sets.

    The resulting d-dimensional vectors are expanded into d numerical
    columns and concatenated with the original numerical features.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.
    cat_cols : list[str]
        List of categorical column names to encode.
    d : int, default=16
        Target embedding dimension for each categorical feature.
    r : float, default=1.0
        Radius of the hypersphere on which category vectors lie.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Encoded training and test feature matrices.
    """

    rng = np.random.default_rng(seed)

    # Separate numerical features
    Xtr_num = X_train.drop(columns=cat_cols)
    Xte_num = X_test.drop(columns=cat_cols)

    tr_blocks = [Xtr_num]
    te_blocks = [Xte_num]

    for col in cat_cols:
        # Treat missing values as a separate category to ensure consistency
        tr = X_train[col].astype("object")
        te = X_test[col].astype("object")

        # 1. Determine category set from TRAIN data only
        cats = pd.unique(tr)
        K = len(cats)
        if K < 2:
            # If there is only one category, no discriminative information exists.
            # Assign a constant vector on the hypersphere.
            const_vec = np.zeros(d, dtype=float)
            const_vec[0] = r
            mapping = {cats[0]: const_vec}
            fallback = const_vec
        else:
            # 2. Construct standard basis vectors e_i (identity matrix)
            I = np.eye(K, dtype=float)

            # 3. Compute centroid (1/K) * 1 vector
            ones = np.ones((K,), dtype=float) / K

            # 4. Center the basis vectors: u_i = e_i - (1/K) * 1
            U = I - ones[None, :] # shape (K, K)

            # 5. Scale vectors so that ||v_i|| = r
            #    ||u_i|| = sqrt((K - 1) / K)
            #    v_i = r * sqrt(K/(K-1)) * u_i
            scale = r * np.sqrt(K / (K - 1))
            V = scale * U # shape (K, K)

            # 6. Random projection from K -> d dimensions
            #    (each column uses an independent projection matrix)
            col_seed = int(rng.integers(0, 2**31 - 1))
            R = make_random_projection_matrix(K=K, d=d, rng=np.random.default_rng(col_seed))
            Z = V @ R # shape (K, d)

            # 7. Re-normalize projected vectors to radius r
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            Z = (Z / norms) * r

            # 8. Build category -> vector mapping
            mapping = {cat: Z[i] for i, cat in enumerate(cats)}

            # 9. Fallback vector for unseen categories (mean direction)
            mean_vec = Z.mean(axis=0)
            mv = np.linalg.norm(mean_vec)

            if mv < 1e-12:
                # Degenerate case: generate random direction
                tmp = np.random.default_rng(col_seed).normal(size=d)
                mean_vec = (tmp / np.linalg.norm(tmp)) * r
            else:
                mean_vec = (mean_vec / mv) * r
            fallback = mean_vec

        # Transform a categorical series into a matrix of shape (N, d)
        def to_matrix(series: pd.Series) -> np.ndarray:
            out = np.zeros((len(series), d), dtype=float)
            for i, v in enumerate(series):
                out[i] = mapping.get(v, fallback)
            return out

        # Encode train and test columns
        Ztr = to_matrix(tr)
        Zte = to_matrix(te)

        # Generate column names for expanded vectors
        new_cols = [f"{col}__gpce_{j}" for j in range(d)]

        tr_blocks.append(pd.DataFrame(Ztr, columns=new_cols, index=X_train.index))
        te_blocks.append(pd.DataFrame(Zte, columns=new_cols, index=X_test.index))

    # Concatenate numerical and encoded categorical features
    Xtr_out = pd.concat(tr_blocks, axis=1)
    Xte_out = pd.concat(te_blocks, axis=1)

    return Xtr_out, Xte_out
