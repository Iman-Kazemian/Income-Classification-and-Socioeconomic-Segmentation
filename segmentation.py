import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_data(data_path: Path, cols_path: Path):
    columns = [line.strip() for line in cols_path.read_text().splitlines() if line.strip()]
    df = pd.read_csv(
        data_path,
        header=None,
        names=columns,
        na_values=["?", " ?"],
        skipinitialspace=True,
    )
    return df


def build_numeric_matrix(df: pd.DataFrame, feature_names):
    X = df[feature_names].copy()
    before = X.shape[0]
    X = X.dropna()
    after = X.shape[0]
    print("Shape before dropping missing:", (before, len(feature_names)))
    print("Shape after dropping missing:", X.shape)
    return X


def run_segmentation(df: pd.DataFrame, n_components: int, n_clusters: int):
    features = [
        "age",
        "wage per hour",
        "capital gains",
        "capital losses",
        "weeks worked in year",
        "detailed industry recode",
        "detailed occupation recode",
        "num persons worked for employer",
    ]
    print("Selected features for clustering:", features)
    X_num = build_numeric_matrix(df, features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num.values)
    print("Scaled feature matrix shape:", X_scaled.shape)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)
    print("\nExplained variance ratio per component:")
    for i, v in enumerate(var, start=1):
        print(f"PC{i}: {v:.4f}")
    print("\nCumulative variance explained:")
    for i, c in enumerate(cum, start=1):
        print(f"PC1..PC{i}: {c:.4f}")

    inertias = []
    print("\nElbow scan (k=2..10) on PCA space:")
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca[:, :n_components])
        inertias.append((k, km.inertia_))
        print(f"k={k}, inertia={km.inertia_:.2f}")

    print("\nSilhouette scores (k=2..10) on PCA space:")
    sil_scores = {}
    max_samples = 20000  # you can lower this (e.g. 10000) if still slow
    
    X_pca_4 = X_pca[:, :4]
    
    for k in range(2, 11):
        print(f"Computing silhouette for k = {k} ...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca_4)
    
        score = silhouette_score(
            X_pca_4,
            labels,
            sample_size=min(max_samples, X_pca_4.shape[0]),
            random_state=42,
        )
        sil_scores[k] = score
        print(f"  silhouette_score = {score:.4f}")
    
    print("\nSilhouette scores by k:")
    for k, s in sil_scores.items():
        print(f"k={k}: {s:.4f}")
        km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_final = km_final.fit_predict(X_pca[:, :n_components])
        X_num_with_clusters = X_num.copy()
        X_num_with_clusters["cluster"] = labels_final

    print("\nCluster sizes:")
    print(X_num_with_clusters["cluster"].value_counts().sort_index())

    print("\nCluster-wise means in original feature space:")
    cluster_means = X_num_with_clusters.groupby("cluster")[features].mean()
    print(cluster_means)

    return inertias, sil_scores, cluster_means, X_num_with_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="census-bureau.data")
    parser.add_argument("--columns", type=str, default="census-bureau.columns")
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--out-clusters", type=str, default="clusters.csv")
    args = parser.parse_args()

    df = load_data(Path(args.data), Path(args.columns))
    inertias, sil_scores, cluster_means, df_clusters = run_segmentation(
        df,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
    )

    df_clusters.to_csv(args.out_clusters, index=False)
    print(f"\nClustered subset saved to {args.out_clusters}")


if __name__ == "__main__":
    main()

