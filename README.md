# Requirements
```xml
faiss                              1.7.2
numpy                              1.20.1
pandas                             1.2.4
scanpy                             1.8.1
scikit-learn                       0.24.1
scipy                              1.6.2
torch                              1.10.0+cpu
torch-cluster                      1.5.9
torch-geometric                    2.0.3
torch-scatter                      2.0.9
torch-sparse                       0.6.12
torch-spline-conv                  1.2.1
```
# Usage
All datasets are available for download at https://www.ncbi.nlm.nih.gov/
You can run the clustering results of VGAAE on Baron-Human2 using the following command. 
```python
python VGAAE_Train.py --datasetType Baron_Human --datasetName Baron_Human2 --num_clusters 7
```
The clustering labels and representation learning are stored in the './results' folder.

