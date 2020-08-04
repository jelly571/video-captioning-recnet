# Missing files
./data/MSVD/features/MSVD_InceptionV4.hdf5
./data/MSVD/metadata/MSR Video Description Corpus.csv
./pycocoevalcap
# Performances

* MSVD

  | Model | BLEU4 | CIDEr | METEOR | ROUGE_L | pretrained |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | RecNet (global) | 51.1 | 79.7 | 34.0 | 69.4 | - |
  | RecNet (local) | **52.3** | **80.3** | **34.1** | **69.8** | - |
  |  |  |  |  |  |  |
  | (Ours) RecNet (global) |46.5 |	55.0 | |67.7 |	|
  | (Ours) RecNet (local) |   |  |	 |	  | |



# References
https://github.com/hobincar/RecNet
