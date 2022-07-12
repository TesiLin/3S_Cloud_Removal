## TRAIN

Modify the `config.yml` to set your parameters and run:

```bash
python train.py
```

## TEST

现在支持只测试一张图/一个文件夹的所有图

这里采用作者预训练的模型[RICE2]((./pretrained_models/RICE1/))(`./pretrained_models/RICE2/gen_model_epoch_200.pth`).

测试格式（一个文件夹的图）：设置test_dir

```bash
python predict-outputdehazed.py --config <path_to_config.yml_in_the_out_dir> --test_dir <path_to_a_directory_stored_test_data> --out_dir <path_to_an_output_directory> --pretrained <path_to_a_pretrained_model> --cuda
```

测试格式（单张图）：设置test_file

以测试RICE2的154.png为例

```bash
python predict-outputdehazed.py --config pretrained_models/RICE2/config.yml --out_dir ./results/test --pretrained ./pretrained_models/RICE2/gen_model_epoch_200.pth --cuda --test_file ./data/RICE_DATASET/RICE2/cloudy_image/154.png
```

同时设置或者同时不设置--test_dir和--test_file会报错



具体传参个数表：

```python
# 要设置
parser.add_argument('--test_dir', type=str, default=None, required=False)
parser.add_argument('--test_file', type=str, default=None, required=False)
parser.add_argument('--out_dir', type=str, required=True)

# 固定
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--pretrained', type=str, required=True)
parser.add_argument('--cuda', action='store_true')

# 可选
parser.add_argument('--gpu_ids', type=int, default=[0])
parser.add_argument('--manualSeed', type=int, default=0)
```

