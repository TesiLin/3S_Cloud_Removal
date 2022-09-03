class settingObj:
    def __init__(self, config, pretrained, gpu_ids, manualSeed, cuda):
        self.config=config
        self.pretrained = pretrained
        self.gpu_ids = gpu_ids
        self.manualSeed = manualSeed
        self.cuda = cuda

    def append(self, test_dir=None, test_file=None, out_dir='./results/test'):
        if (test_dir and test_file) or (test_dir is None and test_file is None):
            print("[Error] Please input either test_dir or test_file.")
            exit()
        self.test_dir = test_dir
        self.test_file = test_file
        self.out_dir = out_dir