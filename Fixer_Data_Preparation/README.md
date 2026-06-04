## Train/Test Split JSON Generation
```
# Example
python create_fixer_dataset_json.py --root DF3DV-1K-Fixer --output dataset_df3dv1k.json
mv dataset_df3dv1k.json ./DF3DV-1K-Fixer/
```

## Image Neighbor Generation for Reference Selection
```
# Generate NN/nn_rank_by_name.json from COLMAP images.txt in undistortion_sparse/0/
# Example
python compute_pose_neighbors.py --root DF3DV-1K-Fixer --overwrite
```
