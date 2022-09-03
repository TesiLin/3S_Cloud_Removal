# How to run?

### 现在支持的predict模式

- 不确定传入一张图还是一个文件夹：`predict-batchOr1dehazed.py`
- 传入一张图：`predict-SingleDehazed.py`

### 命令行参数设置

```python
# 要设置
# test_dir和test_file同时设置or同时不设置：会报错
parser.add_argument('--test_dir', type=str, default=None, required=False) # Single测试代码不传这个，batchOr1可传
parser.add_argument('--test_file', type=str, default=None, required=False) # batchOr1不传这个，Single测试必设
parser.add_argument('--out_dir', type=str, required=True) # 必设
```

### 格式参考

可以直接运行

#### 1 单张图去云

首选`predict-SingleDehazed.py`

```bash
python predict-SingleDehazed.py --out_dir ./results/test --test_file ./data/RICE_DATASET/RICE2/cloudy_image/102.png
```

也可以使用`predict-batchOr1dehazed.py`

```bash
python predict-batchOr1dehazed.py --out_dir ./results/test --test_file ./data/RICE_DATASET/RICE2/cloudy_image/154.png
```

两者区别：针对单张图处理，`predict-batchOr1dehazed.py`经过更复杂封装，更慢。

#### 2 一个文件夹的图去云

使用`predict-batchOr1dehazed.py`，需要设置test_dir

且文件夹格式为：test_dir/cloudy_image/xxx.png

```bash
python predict-batchOr1dehazed.py --out_dir ./results/test --test_dir ./data/RICE_DATASET/RICE2
```



### 其他备注

- 这里采用作者预训练的模型[RICE2]((./pretrained_models/RICE1/))(`./pretrained_models/RICE2/gen_model_epoch_200.pth`)和对应的config文件

- 若最终采用单图处理，则保留`predict-SingleDehazed.py`，删除没用了的`DataManager.py`

