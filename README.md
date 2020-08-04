# Missing files
https://drive.google.com/file/d/18aZ8AdFeJ8h2wPR3YMnZNHnw7ebtfGih/view

./data/MSVD/features/MSVD_InceptionV4.hdf5

./data/MSVD/metadata/MSR Video Description Corpus.csv

Extract the video features from here
https://github.com/hobincar/pytorch-video-feature-extractor

https://github.com/xiadingZ/video-caption.pytorch/tree/master/coco-caption

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
