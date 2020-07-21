参考的这个代码库https://github.com/hobincar/RecNet
环境pytorch
python3
1.将数据集拆分：
python -m splits.MSVD
python -m splits.MSR-VTT
2.训练编码器-解码器
 CUDA_VISIBLE_DEVICES=2  python train.py -c configs.train_stage1
3.训练编码器-解码器-重构器
 CUDA_VISIBLE_DEVICES=2  python train.py -c configs.stage2
4.测试
 CUDA_VISIBLE_DEVICES=2  python run.py
