# RecNet

This project tries to implement *RecNet* proposed in **[Reconstruction Network for Video Captioning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Reconstruction_Network_for_CVPR_2018_paper.pdf) [1], *CVPR 2018***.


# Requirements

* Python3
  * PyTorch

## Step 1. Prepare Data

1. Extract Inception-v4 [2] features from datasets, and locate them at `<PROJECT ROOT>/<DATASET>/features/<DATASET>_InceptionV4.hdf5`. I extracted the Inception-v4 features from [here](https://github.com/hobincar/pytorch-video-feature-extractor).

   | Dataset | Inception-v4 |
   | :---: | :---: |
   | MSVD | [link](https://drive.google.com/open?id=18aZ8AdFeJ8h2wPR3YMnZNHnw7ebtfGih) | 
   | MSR-VTT | [link](https://drive.google.com/open?id=1pFh4u-KwSnCFRl6UJgg7yeaLo2GbxkVT) |

2. Split the dataset along with the official splits by running following:

   ```
   (.env) $ python -m splits.MSVD
   (.env) $ python -m splits.MSR-VTT
   ```
   

## Step 2. Prepare Evaluation Codes

Clone evaluation codes from [the official coco-evaluation repo](https://github.com/tylin/coco-caption).

   ```
   (.env) $ git clone https://github.com/tylin/coco-caption.git
   (.env) $ mv coco-caption/pycocoevalcap .
   (.env) $ rm -rf coco-caption
   ```

## Step 3. Train

* Stage 1 (Encoder-Decoder)

   ```
   (.env) $ python train.py -c configs.train_stage1
   ```

* Stage 2 (Encoder-Decoder-Reconstructor

   Set the `pretrained_decoder_fpath` of `TrainConfig` in `configs/train_stage2.py` as the checkpoint path saved at stage 1, then run

   ```
   (.env) $ python train.py -c configs.stage2
   ```
   
You can change some hyperparameters by modifying `configs/train_stage1.py` and `configs/train_stage2.py`.


## Step 4. Inference

1. Set the checkpoint path by changing `ckpt_fpath` of `RunConfig` in `configs/run.py`.
2. Run
   ```
   (.env) $ python run.py
   ```


# Performances

\* *NOTE: As you can see, the performance of RecNet does not outperform SA-LSTM. Better hyperparameters should be found out.*

* MSVD

  | Model | BLEU4 | CIDEr | METEOR | ROUGE_L | pretrained |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | RecNet (global) | 51.1 | 79.7 | 34.0 | 69.4 | - |
  | RecNet (local) | **52.3** | **80.3** | **34.1** | **69.8** | - |
  |  |  |  |  |  |  |
  | (Ours) RecNet (global) |46.5 |	55.0 | |67.7 |	|
  | (Ours) RecNet (local) |   |  |	 |	  | |


* MSR-VTT

  | Model | BLEU4 | CIDEr | METEOR | ROUGE_L | pretrained |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | RecNet (global) | 38.3 | 41.7 | 26.2 | 59.1 | - |
  | RecNet (local) | **39.1** | **42.7** | **26.6** | **59.3** | - |
  |  |  |  |  |  |  |
  | (Ours) RecNet (global) |  |		| 	|  |  |
  | (Ours) RecNet (local) |  |		| 	|  |  |


# References
https://github.com/hobincar/RecNet
