import os
import argparse
file_dir = os.path.dirname(__file__)


class FlexdepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Flexdepth options")

        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data_png"))  #
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(file_dir, "logs"))


        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "cityscapes_preprocessed"],
                                 default="eigen_zhou")
        self.parser.add_argument("--split_path",
                                 type=str,
                                 help="path to data splits directory",
                                 default=os.path.join(file_dir, "splits"))
        self.parser.add_argument("--make3d_path",
                                 type=str,
                                 help="path to Make3D dataset",
                                 default=os.path.join(file_dir, "make3d"))
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test",
                                          "cityscapes_preprocessed"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--use_gmsd",
                                 help="if set, use gmsd in the loss",
                                 action="store_true")
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")

        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)

        self.parser.add_argument("--mu_learning_rate",
                                 type=float,
                                 help="mu_learning rate",
                                 default=1e-3)

        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)

        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)



        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")


        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        self.parser.add_argument("--encoder_model_type",
                                 type=str,
                                 help="encoder model type resnet or yolo11-seg",
                                 default="resnet",
                                 choices=["resnet", "yolo11n-seg", "yolo11s-seg", "yolo11m-seg",
                                          "yolo11l-seg", "yolo11x-seg"])

        self.parser.add_argument("--decoder_model_type",
                                 type=str,
                                 help="decoder model type",
                                 default="ori",
                                 choices=["ori", "flexn", "flexs", "flexm", "flexl", "flexx"])
        self.parser.add_argument("--upsample", type=str,
                                 help="upsampling method for decoder",
                                 default="nearest",
                                 choices=["bilinear", "nearest", "dysample_pl", "dysample_lp"])
        self.parser.add_argument("--HPB",
                                 help="High-Performance Bottleneck (auto-set for flexm/flexl/flexx)",
                                 action="store_true")
        self.parser.add_argument("--HEB",
                                 help="High-Efficiency Bottleneck (auto-set for flexn/flexs)",
                                 action="store_true")
        self.parser.add_argument("--inv_upsample",
                                 help="Inverted prediction head with dynamic upsampling (auto-set for flexm/flexl/flexx)",
                                 action="store_true")

        self.parser.add_argument("--optim",
                                 type=str,
                                 help="optimizer select ",
                                 default="Adam",
                                 choices=["Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--same_lr",
                                 help="if set encoder and decoder same lr",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)
        self.parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        self.parser.add_argument("--save_predictions",
                                 action="store_true",
                                 help="if set, save depth visualizations and error maps")

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        self.parser.add_argument("--save_best",
                                 help="only save best epoch not every epoch",
                                 action="store_true",
                                 )

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--use_lstsq_alignment",
                                 help="if set, use least-squares alignment in disparity space instead of median scaling",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                     "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10",
                                     "cityscapes_preprocessed", "cityscapes"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

        self.parser.add_argument('--sync_bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        self.parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.parser.add_argument("--amp",
                                 help="if set will use amp", action="store_true")
        self.parser.add_argument("--apex",
                                 help="if set will use apex", action="store_true")
        self.parser.add_argument("--all_mem",
                                 help="if set data all use mem if your pc have big ram ", action="store_true")
        self.parser.add_argument("--nbs",
                                 help=" Nominal batch size for normalization of loss. when amp and batchsize", type=int,

                                 default=12)
        self.parser.add_argument("--useTensorRT",
                                 help="useTensorRT only avalible in eval or test",
                                 action="store_true")
        self.parser.add_argument("--seed", help="set seed", type=int, default=1)

        self.parser.add_argument("--lrf",
                                 type=float,
                                 help="Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time. rate",
                                 default=0.01)
        self.parser.add_argument("--momentum",
                                 type=float,
                                 help="momentum",
                                 default=0.937)
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="weight_decay",
                                 default=0.0005)

        self.parser.add_argument("--warmup_momentum",
                                 type=float,
                                 help="warmup_momentum",
                                 default=0.8)

        self.parser.add_argument("--warmup_bias_lr",
                                 type=float,
                                 help="weight_decay",
                                 default=0.1)

        self.parser.add_argument("--warmup_epochs",
                                 type=int,
                                 help="number of warmup_epochs",
                                 default=0)
        self.parser.add_argument("--start_epoch",
                                 type=int,
                                 help="start of epochs",
                                 default=0)

        self.parser.add_argument("--scheduler",
                                 help="Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs. Helps in managing learning rate for better convergence.",
                                 type=str,
                                 default="step",
                                 choices=["step", "cos", "linear", "ExponentialLR"])
        self.parser.add_argument("--step2_scheduler",
                                 help="Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs. Helps in managing learning rate for better convergence.",
                                 type=str,
                                 default="ExponentialLR_08",
                                 choices=["step", "cos", "linear", "ExponentialLR_08", "ExponentialLR_09"])

        self.parser.add_argument("--use_step_2",
                                 help="if set, use two-stage training (step 2)",
                                 action="store_true")

        self.parser.add_argument("--skip_step1",
                                 help="if set, skip step 1 and start from step 2 directly",
                                 action="store_true")

        self.parser.add_argument("--step_2_epoch",
                                 help="epoch to start step 2 training",
                                 type=int,
                                 default=20)
        self.parser.add_argument("--start_opt_epoch",
                                 help="epoch to start optimizer for step 2 freeze",
                                 type=int,
                                 default=19)
        self.parser.add_argument("--for_debug",
                                 help="use a small subset of data for debugging",
                                 action="store_true")
        self.parser.add_argument("--use_var_net",
                                 help="if set, uses variance for training",
                                 action="store_true")

        self.parser.add_argument("--freeze_encoder",
                                 help="if set, will freeze_encoder backbone",
                                 action="store_true")

        self.parser.add_argument("--resume",
                                 help="if set, resume from last",
                                 action="store_true")

        self.parser.add_argument("--c3k",
                                 help="if set, decoder c3k with True",
                                 action="store_true")
        self.parser.add_argument("--all_dysample",
                                 help="if set, decoder use all dysample",
                                 action="store_true")
        self.parser.add_argument("--dysample_dynamic",
                                 help="if set, decoder use dysample with dynamic scope (experimental)",
                                 action="store_true")
        self.parser.add_argument("--dysample_group8",
                                 help="if set, decoder use  dysample with dynamic group8",
                                 action="store_true")

        self.parser.add_argument("--fuse_dysample",
                                 help="if set, decoder use fuse_dysample (experimental)",
                                 action="store_true")
        self.parser.add_argument("--c3mixk",
                                 help="if set, decoder use mix_c3k (experimental)",
                                 action="store_true")
        self.parser.add_argument("--dy_mu",
                                 help="if set, train use dy_mu",
                                 action="store_true")
        self.parser.add_argument("--ada_dy_mu",
                                 help="if set, train use ada_dy_mu with self learning (experimental)",
                                 action="store_true")

        self.parser.add_argument("--diff_dy_mu",
                                 help="if set, train use diff_dy_mu (experimental)",
                                 action="store_true")

        self.parser.add_argument("--optimize-eval",
                                 help="if set, uses GPU-accelerated chunked evaluation",
                                 action="store_true")

        self.parser.add_argument("--export_name",
                                 help="export onnx model name",
                                 type=str,
                                 default="monodepth",
                                 )

        self.parser.add_argument("--use_wb",
                                 help="use weight and bias ",
                                 action="store_true")


    def parse(self):
        self.options = self.parser.parse_args()

        self._apply_decoder_defaults()
        self._validate_decoder_options()

        if isinstance(self.options.scales, list) and len(self.options.scales) > 1:
            pass
        else:
            if isinstance(self.options.scales, list):
                self.options.scales = list(range(int(self.options.scales[0])))
            else:
                self.options.scales = list(range(int(self.options.scales)))
        # Set DDP variables
        self.options.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        self.options.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
        print("[WORLD_SIZE]", self.options.world_size)
        print("[global_rank]", self.options.global_rank)
        self.options.total_batch_size = self.options.batch_size
        return self.options

    def _apply_decoder_defaults(self):
        """Auto-apply decoder configuration based on decoder_model_type.
        Paper: Scale-Driven Decoder with HEB/HPB, Dynamic Upsampling, Inverted Head."""
        opt = self.options
        if "flex" not in opt.decoder_model_type:
            return

        scale = opt.decoder_model_type.replace("flex", "")
        hpb_scales = {"m", "l", "x"}
        heb_scales = {"n", "s"}

        # All flex decoders use dynamic upsampling
        if opt.upsample == "nearest":
            opt.upsample = "dysample_pl"
            print(f"[Auto-default] --upsample dysample_pl (auto-set for {opt.decoder_model_type})")

        if scale in hpb_scales:
            if not opt.c3k:
                opt.c3k = True
                print(f"[Auto-default] --c3k (HPB, auto-set for {opt.decoder_model_type})")
            if not opt.dysample_group8:
                opt.dysample_group8 = True
                print(f"[Auto-default] --dysample_group8 (auto-set for {opt.decoder_model_type})")
            if not opt.all_dysample:
                opt.all_dysample = True
                print(f"[Auto-default] --all_dysample (Inverted Head, auto-set for {opt.decoder_model_type})")
            if not opt.HPB:
                opt.HPB = True
            if not opt.inv_upsample:
                opt.inv_upsample = True

        elif scale in heb_scales:
            if opt.c3k:
                opt.c3k = False
                print(f"[Auto-default] --c3k disabled (HEB, auto-set for {opt.decoder_model_type})")
            if opt.dysample_group8:
                opt.dysample_group8 = False
                print(f"[Auto-default] --dysample_group8 disabled (auto-set for {opt.decoder_model_type})")
            if opt.all_dysample:
                opt.all_dysample = False
                print(f"[Auto-default] --all_dysample disabled (Normal Head, auto-set for {opt.decoder_model_type})")
            if not opt.HEB:
                opt.HEB = True

    def _validate_decoder_options(self):
        """Validate decoder configuration matches paper constraints."""
        opt = self.options
        if "flex" not in opt.decoder_model_type:
            return

        scale = opt.decoder_model_type.replace("flex", "")
        hpb_scales = {"m", "l", "x"}
        heb_scales = {"n", "s"}

        # HEB/HPB conflict check
        if scale in hpb_scales and opt.HEB:
            raise ValueError(
                f"Conflict: --HEB cannot be used with {opt.decoder_model_type} "
                f"(uses HPB - High-Performance Bottleneck)")
        if scale in heb_scales and opt.HPB:
            raise ValueError(
                f"Conflict: --HPB cannot be used with {opt.decoder_model_type} "
                f"(uses HEB - High-Efficiency Bottleneck)")

        # Encoder-decoder scale matching
        if "yolo11" in opt.encoder_model_type and "flex" in opt.decoder_model_type:
            enc_scale = opt.encoder_model_type.replace("yolo11", "").split("-")[0]
            dec_scale = opt.decoder_model_type.replace("flex", "")
            if enc_scale != dec_scale:
                raise ValueError(
                    f"Encoder scale '{enc_scale}' must match decoder scale '{dec_scale}'. "
                    f"Got --encoder_model_type {opt.encoder_model_type} with "
                    f"--decoder_model_type {opt.decoder_model_type}")
