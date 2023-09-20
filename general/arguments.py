class Arguments:
    root: str = ""
    output_dir: str = ""
    resume: str = ""
    seed: int = -1
    source_domains: str = None
    target_domains: str = None
    transforms: str = None
    config_file: str = "configs/trainers/CoOp/"
    dataset_config_file: str = "configs/datasets/"
    trainer: str = "CoOp"
    backbone: str = ""
    head: str = ""
    eval_only: bool = False
    model_dir: str = ""
    load_epoch: int = None
    no_train: bool = False
    n_ctx: int = 16
    csc: bool = False
    ctp: str = "end"
    shots: int = 1
    opts: list = [] # opts=['TRAINER.COOP.N_CTX', '16', 'TRAINER.COOP.CSC', 'False', 'TRAINER.COOP.CLASS_TOKEN_POSITION', 'end', 'DATASET.NUM_SHOTS', '1']
    open_clip: bool = True
    pretrained: str = ""

    def __init__(self, trainer: str, root: str, dataset: str, config: str, ctp: str, n_ctx: int, shots: int, csc: bool, output_dir: str, open_clip: bool, pretrained: str):
        self.trainer = str(trainer)
        self.root = str(root)
        self.dataset_config_file += str(dataset) + ".yaml"
        self.config_file += str(config) + ".yaml"
        self.ctp = str(ctp)
        self.n_ctx = n_ctx
        self.shots = shots
        self.csc = csc
        self.opts = ['TRAINER.COOP.N_CTX', str(n_ctx), 'TRAINER.COOP.CSC', str(csc), 'TRAINER.COOP.CLASS_TOKEN_POSITION',
                     str(ctp), 'DATASET.NUM_SHOTS', str(shots)]
        self.output_dir = output_dir
        self.open_clip = open_clip
        self.pretrained = pretrained

    def __str__(self):
        return (f"backbone={self.backbone}, config_file={self.config_file}, dataset_config_file={self.dataset_config_file}, "
                f"eval_only={self.eval_only}, head={self.head}, load_epoch={self.load_epoch}, model_dir={self.model_dir}, "
                f"no_train={self.no_train},  opts: {self.opts}, output_dir={self.output_dir}, resume={self.resume}, "
                f"root={self.root}, seed={self.seed}, source_domains={self.source_domains}, "
                f"target_domains={self.target_domains}, trainer={self.trainer}, transforms={self.transforms}, open_clip={self.open_clip}, pretrained={self.pretrained}")
