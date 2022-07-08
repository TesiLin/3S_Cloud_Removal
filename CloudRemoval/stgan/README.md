# STGAN

> 来自[stgan 项目](https://github.com/VSAnimator/stgan)

## Usage

1. 从[https://doi.org/10.7910/DVN/BSETKZ](https://doi.org/10.7910/DVN/BSETKZ)获取多时像数据集, 解压至stgan文件夹外
2. `python divide.py`进行取样、分割;
3. `python datasets/combine_temporal.py --fold_A ../divide_multi/cloudy/ --fold_B ../divide_multi/clear/ --fold_AB ../divide_multi/combined/`进行 combine 操作。
4. `python train.py --dataroot ../divide_multi/combined --name stgan --model temporal_branched --netG unet_256_independent --dataset_mode temporal --input_nc 3 `进行训练
   - 训练模型及配置文件保存在`./checkpoints/stgan/`。(如果是cpu加上`--gpu_ids -1`,`--netG`可选参数参考`./options/base_options.py`文件)
5. `python test.py --dataroot ../divide_multi/combined --name stgan --model temporal_branched --netG unet_256_independent --dataset_mode temporal --input_nc 3`进行图像去云测试
   - 结果文件保存在`./results/stgan/test_latest/index.html`
6. `python test_quantification.py --dataroot ../divide_multi/combined --name stgan --model temporal_branched --netG unet_256_independent --dataset_mode temporal --input_nc 3`进行结果量化测试