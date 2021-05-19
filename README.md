# Code using V-Net to segment blood vessels from large 3D phase contrast CT images
A [PyTorch](https://pytorch.org/) Implementation of the [V-Net](https://arxiv.org/abs/1606.04797)
based on [github.com/mattmacy/vnet.pytorch](https://github.com/mattmacy/vnet.pytorch) using [PyTorch Lightning](https://www.pytorchlightning.ai/).

Since the 3D images we train on are too large to process on the GPU directly, we train on batches of random 3D crops.
We ensure that the extracted crops always contain labelled blood vessels before we send it to the network.
Furthermore, to avoid having to annotate full 3D images, we only annotate small parts of 2D slices.
Annotations are stored in 3D images where 0 means 'not-annotated', 1 means 'blood vessel', and 2 means 'background tissue'.
Inference is performed using an overlapping sliding window approach.

**Note:** Many of the shell scripts were made to interface with the HPC system at the Technical University of Denmark (DTU) and may therefore not work for your system.
More info on the DTU HPC system is available at [www.hpc.dtu.dk](https://www.hpc.dtu.dk).

## How to run training
Data must be put in a directory containing two sub-directories `train` and `val` for training and validation data, respectively.
Each 3D image should be stored as a `.npy` file as pairs of files called `data_<X>.npy` and `mask_<X>.npy`, where `<X>` is an index.

To perform training, run:
```
python run_training.py [LIGHTNING_OPTIONS] {vnet,unet} [NETWORK_OPTIONS]
```
where `NETWORK_OPTIONS` are:
```
-h, --help            show this help message and exit
--lr LR
--num_loader_workers NUM_LOADER_WORKERS
--batch_size BATCH_SIZE
--crop_size CROP_SIZE
--samples_per_volume SAMPLES_PER_VOLUME
--data_dir DATA_DIR
--min_lr MIN_LR
--lr_reduce_factor LR_REDUCE_FACTOR
--normalization {b,g}
```
The `LIGHTNING_OPTIONS` are shown in their own section below.

For example usage, see `submit_train.sh`.

## How to run inference
All 3D images to use for inference must be in a directory and stored as `.npy` files. Names do not matter.
Furthermore, a model checkpoint (as saved by PyTorch Lightning) must be available.
For each input 3D image `X.npy`, two new ones will be saved: `<CHKPT>.X.npy.pred.npy` and `<CHKPT>.X.npy.mask.npy`,
where `<CHKPT>` is the name of the checkpoint file.
These files contain, respectively, the model prediction probabilities and prediction mask (which is the probablity image thresholded at 0.5).

To perform inference, run:
```
python run_inference.py CHECKPOINT [OPTIONS]
```
where `CHECKPOINT` is the path to the checkpoint file and `OPTIONS` are:
```
-h, --help            show this help message and exit
--save_dir SAVE_DIR
--data_dir DATA_DIR
--crop_size CROP_SIZE
--file_type {npy,raw}
--model {vnet,unet}
--border BORDER
```
For example usage, see `submit_inference.sh`.

## PyTorch Lightning options
```
  -h, --help            show this help message and exit
  --logger_save_dir LOGGER_SAVE_DIR
  --monitor_loss MONITOR_LOSS
  --save_top_k SAVE_TOP_K
  --experiment_name EXPERIMENT_NAME
  --date_time DATE_TIME
  --checkpoint_path CHECKPOINT_PATH
  --logger [LOGGER]     Logger (or iterable collection of loggers) for
                        experiment tracking.
  --checkpoint_callback [CHECKPOINT_CALLBACK]
                        If ``True``, enable checkpointing. It will configure a
                        default ModelCheckpoint callback if there is no user-
                        defined ModelCheckpoint in :paramref:`~pytorch_lightni
                        ng.trainer.trainer.Trainer.callbacks`. Default:
                        ``True``. .. warning:: Passing a ModelCheckpoint
                        instance to this argument is deprecated since v1.1.0
                        and will be unsupported from v1.3.0.
  --default_root_dir DEFAULT_ROOT_DIR
                        Default path for logs and weights when no
                        logger/ckpt_callback passed. Default: ``os.getcwd()``.
                        Can be remote file paths such as `s3://mybucket/path`
                        or 'hdfs://path/'
  --gradient_clip_val GRADIENT_CLIP_VAL
                        0 means don't clip.
  --process_position PROCESS_POSITION
                        orders the progress bar when running multiple models
                        on same machine.
  --num_nodes NUM_NODES
                        number of GPU nodes for distributed training.
  --num_processes NUM_PROCESSES
  --gpus GPUS           number of gpus to train on (int) or which GPUs to
                        train on (list or str) applied per node
  --auto_select_gpus [AUTO_SELECT_GPUS]
                        If enabled and `gpus` is an integer, pick available
                        gpus automatically. This is especially useful when
                        GPUs are configured to be in "exclusive mode", such
                        that only one process at a time can access them.
  --tpu_cores TPU_CORES
                        How many TPU cores to train on (1 or 8) / Single TPU
                        to train on [1]
  --log_gpu_memory LOG_GPU_MEMORY
                        None, 'min_max', 'all'. Might slow performance
  --progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE
                        How often to refresh progress bar (in steps). Value
                        ``0`` disables progress bar. Ignored when a custom
                        callback is passed to :paramref:`~Trainer.callbacks`.
  --overfit_batches OVERFIT_BATCHES
                        Overfit a percent of training data (float) or a set
                        number of batches (int). Default: 0.0
  --track_grad_norm TRACK_GRAD_NORM
                        -1 no tracking. Otherwise tracks that p-norm. May be
                        set to 'inf' infinity-norm.
  --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        Check val every n train epochs.
  --fast_dev_run [FAST_DEV_RUN]
                        runs 1 batch of train, test and val to find any bugs
                        (ie: a sort of unit test).
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Accumulates grads every k batches or as set up in the
                        dict.
  --max_epochs MAX_EPOCHS
                        Stop training once this number of epochs is reached.
  --min_epochs MIN_EPOCHS
                        Force training for at least these many epochs
  --max_steps MAX_STEPS
                        Stop training after this number of steps. Disabled by
                        default (None).
  --min_steps MIN_STEPS
                        Force training for at least these number of steps.
                        Disabled by default (None).
  --limit_train_batches LIMIT_TRAIN_BATCHES
                        How much of training dataset to check (floats =
                        percent, int = num_batches)
  --limit_val_batches LIMIT_VAL_BATCHES
                        How much of validation dataset to check (floats =
                        percent, int = num_batches)
  --limit_test_batches LIMIT_TEST_BATCHES
                        How much of test dataset to check (floats = percent,
                        int = num_batches)
  --val_check_interval VAL_CHECK_INTERVAL
                        How often to check the validation set. Use float to
                        check within a training epoch, use int to check every
                        n steps (batches).
  --flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS
                        How often to flush logs to disk (defaults to every 100
                        steps).
  --log_every_n_steps LOG_EVERY_N_STEPS
                        How often to log within steps (defaults to every 50
                        steps).
  --accelerator ACCELERATOR
                        Previously known as distributed_backend (dp, ddp,
                        ddp2, etc...). Can also take in an accelerator object
                        for custom hardware.
  --sync_batchnorm [SYNC_BATCHNORM]
                        Synchronize batch norm layers between process
                        groups/whole world.
  --precision PRECISION
                        Full precision (32), half precision (16). Can be used
                        on CPU, GPU or TPUs.
  --weights_summary WEIGHTS_SUMMARY
                        Prints a summary of the weights when training begins.
  --weights_save_path WEIGHTS_SAVE_PATH
                        Where to save weights if specified. Will override
                        default_root_dir for checkpoints only. Use this if for
                        whatever reason you need the checkpoints stored in a
                        different place than the logs written in
                        `default_root_dir`. Can be remote file paths such as
                        `s3://mybucket/path` or 'hdfs://path/' Defaults to
                        `default_root_dir`.
  --num_sanity_val_steps NUM_SANITY_VAL_STEPS
                        Sanity check runs n validation batches before starting
                        the training routine. Set it to `-1` to run all
                        batches in all validation dataloaders. Default: 2
  --truncated_bptt_steps TRUNCATED_BPTT_STEPS
                        Truncated back prop breaks performs backprop every k
                        steps of much longer sequence.
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        To resume training from a specific checkpoint pass in
                        the path here. This can be a URL.
  --profiler [PROFILER]
                        To profile individual steps during training and assist
                        in identifying bottlenecks. Passing bool value is
                        deprecated in v1.1 and will be removed in v1.3.
  --benchmark [BENCHMARK]
                        If true enables cudnn.benchmark.
  --deterministic [DETERMINISTIC]
                        If true enables cudnn.deterministic.
  --reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]
                        Set to True to reload dataloaders every epoch.
  --auto_lr_find [AUTO_LR_FIND]
                        If set to True, will make trainer.tune() run a
                        learning rate finder, trying to optimize initial
                        learning for faster convergence. trainer.tune() method
                        will set the suggested learning rate in self.lr or
                        self.learning_rate in the LightningModule. To use a
                        different key set a string instead of True with the
                        key name.
  --replace_sampler_ddp [REPLACE_SAMPLER_DDP]
                        Explicitly enables or disables sampler replacement. If
                        not specified this will toggled automatically when DDP
                        is used. By default it will add ``shuffle=True`` for
                        train sampler and ``shuffle=False`` for val/test
                        sampler. If you want to customize it, you can set
                        ``replace_sampler_ddp=False`` and add your own
                        distributed sampler.
  --terminate_on_nan [TERMINATE_ON_NAN]
                        If set to True, will terminate training (by raising a
                        `ValueError`) at the end of each training batch, if
                        any of the parameters or the loss are NaN or +/-inf.
  --auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]
                        If set to True, will `initially` run a batch size
                        finder trying to find the largest batch size that fits
                        into memory. The result will be stored in
                        self.batch_size in the LightningModule. Additionally,
                        can be set to either `power` that estimates the batch
                        size through a power search or `binsearch` that
                        estimates the batch size through a binary search.
  --prepare_data_per_node [PREPARE_DATA_PER_NODE]
                        If True, each LOCAL_RANK=0 will call prepare data.
                        Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare
                        data
  --amp_backend AMP_BACKEND
                        The mixed precision backend to use ("native" or
                        "apex")
  --amp_level AMP_LEVEL
                        The optimization level to use (O1, O2, etc...).
  --distributed_backend DISTRIBUTED_BACKEND
                        deprecated. Please use 'accelerator'
  --automatic_optimization [AUTOMATIC_OPTIMIZATION]
                        If False you are responsible for calling .backward,
                        .step, zero_grad. Meant to be used with multiple
                        optimizers by advanced users.
```
