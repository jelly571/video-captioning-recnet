

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
