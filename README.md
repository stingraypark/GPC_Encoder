# GPC Encoder (Geometry Preserving Category Encoder)

## Introduction & Purpose

In ML/DL preprocessing, categorical data is commonly encoded using **one-hot**, **label**, or **target mean encoding**, each of which has clear limitations:

- **One-Hot Encoding**
  - Causes a rapid increase in dimensionality for high-cardinality features
  - Increases model complexity and memory usage

- **Label Encoding**
  - Introduces artificial ordinal relationships between categories
  - Can mislead linear or distance-based models

- **Target Mean Encoding**
  - Prone to **data leakage**, as global target statistics influence encoding
  - Unstable for rare categories or small datasets, leading to bias

The purpose of **GPC Encoder** is to address all of these issues by designing an encoding method that:

1. Does **not impose any ordering** on categories  
2. Does **not drastically increase dimensionality**  
3. **Avoids data leakage** by relying only on training data statistics  

---

## Requirements

- **Python 3.11.4**
- See `requirements.txt` for full dependency details

