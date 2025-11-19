pip install -r requirements.txt

# 1) Run classification (all models + deep MLP)
python classification.py --data census-bureau.data --columns census-bureau.columns

# 2) Run segmentation (PCA + KMeans, default 5 clusters)
python segmentation.py --data census-bureau.data --columns census-bureau.columns --n-components 4 --n-clusters 5
