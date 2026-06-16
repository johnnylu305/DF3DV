## Image Dataset Generation
You can follow the [instructions](https://github.com/johnnylu305/DF3DV/tree/main/DF3DV_Benchmark) here to generate `MODELS/<method>/renders/extra_*.png`.
```
# convert DF3DV-1K into DF3DV-1K-Fixer format.
# extract the ground-truth and rendering halves from MODELS/<method>/renders/extra_*.png,
# save them as paired source/target images, and copy undistortion_sparse.
python build_fixer_data.py --src_root DF3DV-1K --dst_root DF3DV-1K-Fixer --run_star --run_41  --start 0 --end 24 --num_workers 8 --overwrite
```

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
