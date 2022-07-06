# Pix2Pix

> 来自[stgan 项目](https://github.com/VSAnimator/stgan)

## Usage

1. 从[https://doi.org/10.7910/DVN/BSETKZ](https://doi.org/10.7910/DVN/BSETKZ)获取单时相数据集。
2. `python split.py`进行取样、分割。
3. `python datasets/combine_normal.py --fold_A divide_single/cloudy/ --fold_B divide_single/clear/ --fold_AB ./single_combined`进行 combine 操作。
4. `python train.py --dataroot ./single_combined --name pix2pix --model pix2pix`训练。
5. `python test.py --dataroot ./single_combined --name pix2pix --model pix2pix`测试，结果文件保存在`./results/stgan/test_latest/index.html`
