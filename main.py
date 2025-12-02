import argparse
import logging
import os
import pprint
import shutil
import time
import torch
import yaml
from data.data_builder import build_dataloader
from easydict import EasyDict
from tensorboardX import SummaryWriter
from utils.loss import build_criterion
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    to_device
)
from utils.optimizer import get_optimizer
from utils.vis_helper import visualize_compound, visualize_single, visualize_compound_aug
from setproctitle import setproctitle
from models.contrastive.model_builder import ContrastiveModelBuilder
from models.diffusion.model_builder import DiffusionModelBuilder

setproctitle(f"thuannt is training")    

class_name_list = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="./config/config.yml")
parser.add_argument("--training_phase", default="diffusion", choices=["contrastive", "diffusion"])
parser.add_argument("--class_name", default="bottle", help="Class name for separate training")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--local_rank", default=None)
parser.add_argument("--single_gpu", action="store_true")

def train_contrastive():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    if not args.class_name:
        raise ValueError("Class name is required for contrastive training")
    dataset_class_name = args.class_name
    print("Contrastive training for class: {}".format(dataset_class_name))

    if args.class_name:
        config.exp_path = os.path.join(os.path.dirname(args.config), args.class_name)
    else:
        config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, "contrastive", config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, "contrastive", config.saver.log_dir)
    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)

    current_time = get_current_time()
    tb_logger = SummaryWriter(config.log_path + "/events_contrastive/" + current_time)
    logger = create_logger(
        "global_logger", config.log_path + "/contrastive_{}.log".format(current_time)
    )
    logger.info("args: {}".format(pprint.pformat(args)))
    logger.info("config: {}".format(pprint.pformat(config)))

    if args.class_name:
        logger.info(f"Training separate class: {args.class_name}")
        logger.info(f"Experiment path: {config.exp_path}")

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    set_random_seed(random_seed, reproduce)

    # create model
    model = ContrastiveModelBuilder(config)
    model.cuda()
    # optimizer
    parameters = [{ 'params': model.parameters() }]
    optimizer = get_optimizer(parameters, config.contrastive_trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.contrastive_trainer.lr_scheduler)

    last_epoch = 0
    best_loss = float('inf')
    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)

    lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")
    best_model = os.path.join(config.save_path, "ckpt_best.pth.tar")
    
    if auto_resume:
        if os.path.exists(lastest_model):
            resume_model = lastest_model
        elif os.path.exists(best_model):
            resume_model = best_model
    
    if resume_model:
        checkpoint_data = load_state(resume_model, model, optimizer=optimizer, return_full_checkpoint=True)
        if isinstance(checkpoint_data, tuple):
            best_metric, last_epoch = checkpoint_data
        else:
            best_metric = checkpoint_data.get("best_metric", 0)
            last_epoch = checkpoint_data.get("epoch", 0)
            # Load best_loss if available, otherwise keep as inf
            best_loss = checkpoint_data.get("best_loss", float('inf'))
        logger.info(f"Resumed training from epoch {last_epoch} with best loss {best_loss}")
        logger.info(f"Resume model path: {resume_model}")
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)
        logger.info(f"Loaded model from: {load_path}")
    # dataloader
    trainer_loader, _ = build_dataloader(config.dataset, class_name=dataset_class_name)
    # loss
    criterion = build_criterion(config.train_contrastive_criterion)

    # training loop
    for epoch in range(last_epoch, config.contrastive_trainer.max_epoch):
        if epoch == last_epoch:
            logger.info(f'Stating contrastive training from epoch {epoch + 1}/{config.contrastive_trainer.max_epoch}')
        last_iter = epoch * len(trainer_loader)
        # train
        epoch_loss = train_one_epoch_contrastive(
            trainer_loader, 
            model, 
            optimizer, 
            lr_scheduler, 
            epoch, 
            last_iter, 
            tb_logger, 
            criterion
        )
        lr_scheduler.step()
        
        # Check if this is the best loss (lower is better)
        is_best_loss = epoch_loss < best_loss
        if is_best_loss:
            best_loss = epoch_loss
            logger.info(f"New best loss: {best_loss:.5f} at epoch {epoch + 1}")
        # Prepare checkpoint state
        checkpoint_state = {
            "epoch": epoch + 1,
            "arch": config.get('contrastive_net', []),
            "state_dict": model.state_dict(),
            "best_loss": best_loss,
            "optimizer": optimizer.state_dict(),
        }
        # Save best loss checkpoint immediately when we get a new best
        if is_best_loss:
            # Save latest checkpoint as "ckpt.pth.tar" and best as "ckpt_best.pth.tar"
            save_checkpoint(checkpoint_state, True, config)
            logger.info(f"Saved best loss checkpoint at epoch {epoch + 1} with loss {best_loss:.5f}")
        
        # Save latest checkpoint at specified intervals
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config.contrastive_trainer.max_epoch:
            if not is_best_loss:
                save_checkpoint(checkpoint_state, False, config)
                logger.info(f"Saved checkpoint at epoch {epoch + 1} (current loss: {epoch_loss:.5f}, best loss: {best_loss:.5f})")
            else:
                logger.info(f"Saved checkpoint at epoch {epoch + 1} (current loss: {epoch_loss:.5f}, best loss: {best_loss:.5f})") 
    

def train_one_epoch_contrastive(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion
):
    batch_time = AverageMeter(config.contrastive_trainer.print_freq_step)
    data_time = AverageMeter(config.contrastive_trainer.print_freq_step)
    losses = AverageMeter(config.contrastive_trainer.print_freq_step)

    model.train()
    logger = logging.getLogger("global_logger")
    end = time.time()

    for i, input in enumerate(train_loader):
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else lr_scheduler.get_lr()[0]

        # loading time
        data_time.update(time.time() - end)
        # input to device
        input_dev = to_device(input, device='cuda')
        image = input_dev["image"] # [ B, C, H, W ]

        # forward contrastive
        embeddings = model(image) # [B, h_dim*2]

        B = embeddings.shape[0]
        h_dim = embeddings.shape[1] // 2
        embeddings_aug = embeddings[:, :h_dim]  # [B, h_dim] - first h_dim are augmented
        embeddings_orig = embeddings[:, h_dim:]  # [B, h_dim] - second h_dim are original
        # concat embeddings_aug and embeddings_orig
        embeddings_reshaped = torch.cat([embeddings_aug, embeddings_orig], dim=0)

        # compute loss
        loss = 0
        # NTXentLoss [B*2, h_dim] tensor
        # criterion is a dict, so we need to iterate over it or access by name
        for name, criterion_loss in criterion.items():
            loss += criterion_loss(embeddings_reshaped) 
        losses.update(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        if (curr_step + 1) % config.contrastive_trainer.print_freq_step == 0:
            if tb_logger:
                tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
                tb_logger.add_scalar("lr", current_lr, curr_step + 1)
                tb_logger.flush()
            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.10f} ({loss.avg:.5f})\t"
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.contrastive_trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.contrastive_trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=current_lr,
                )
            )
        end = time.time()
    return losses.avg

"""
==========================================================================
"""

def train_diffusion():
    global args, config, key_metric, best_metric
    args = parser.parse_args()
    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    dataset_class_name = args.class_name if args.class_name else None
    if args.class_name:
        print(f"Training separate class: {args.class_name}")
    print("config: {}".format(pprint.pformat(config)))
    # setup experiment path
    if args.class_name:
        config.exp_path = os.path.join(os.path.dirname(args.config), args.class_name)
    else:
        config.exp_path = os.path.dirname(args.config)
    
    config.save_path = os.path.join(config.exp_path, "diffusion", config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, "diffusion", config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, "diffusion", config.evaluator.save_dir)
    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)

    current_time = get_current_time()
    tb_logger = SummaryWriter(config.log_path + "/events_diffusion/" + current_time)
    logger = create_logger(
        "global_logger", config.log_path + "/diffusion_{}.log".format(current_time)
    )
    logger.info("args: {}".format(pprint.pformat(args)))
    logger.info("config: {}".format(pprint.pformat(config)))

    # Log class-specific training info
    if args.class_name:
        logger.info(f"Training separate class: {args.class_name}")
        logger.info(f"Experiment path: {config.exp_path}")

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    phase1_checkpoint_dir = os.path.join(config.exp_path, "contrastive", config.saver.save_dir)
    phase1_checkpoint_path = os.path.join(phase1_checkpoint_dir, "ckpt.pth.tar")
    
    # check phase 1 checkpoint exists
    if not os.path.exists(phase1_checkpoint_path):
        error_msg = (
            f"Phase 1 (contrastive) checkpoint not found at: {phase1_checkpoint_path}\n"
            f"Please ensure contrastive training has been completed for class: {args.class_name or 'default'}\n"
            f"Expected path: {phase1_checkpoint_path}"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    logger.info(f"Found phase 1 checkpoint at: {phase1_checkpoint_path}")
    # Load pretrained contrastive model to extract gumbel_aug weights
    contrastive_model = ContrastiveModelBuilder(config)
    checkpoint_data = load_state(phase1_checkpoint_path, contrastive_model, return_full_checkpoint=True)
    contrastive_model.eval()  
    logger.info(f"Loaded contrastive model from: {phase1_checkpoint_path}")
    # build diffusion model
    model = DiffusionModelBuilder(config, contrastive_model=contrastive_model)
    model.cuda()
    if logger:
        logger.info("Phase 1 weights loaded successfully")

    # Get all layers from diffusion_net config
    layers = [module["name"] for module in config.get('diffusion_net', [])]
    # Get frozen layers from config
    frozen_layers = config.get("frozen_layers", [])
    # Calculate active layers (layers not in frozen_layers)
    active_layers = list(set(layers) - set(frozen_layers))
    # Remove sub-modules that should not be trained
    sub_modules_to_exclude = ["linear-augmentation", "cnn-augmentation"]
    active_layers = [layer for layer in active_layers if layer not in sub_modules_to_exclude]
    
    if logger:
        logger.info("layers: {}".format(layers))
        logger.info("frozen layers (from config): {}".format(frozen_layers))
        logger.info("active layers: {}".format(active_layers))
    
    parameters = []
    for layer in active_layers:
        if hasattr(model, layer):
            layer_module = getattr(model, layer)
            # Check if layer has any trainable parameters
            if any(p.requires_grad for p in layer_module.parameters()):
                parameters.append({"params": layer_module.parameters()})
    
    if not parameters: raise ValueError("No trainable parameters found! All layers are frozen.")

    optimizer = get_optimizer(parameters, config.diffusion_trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.diffusion_trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0
    best_std_metric = 0
    best_max_metric = 0
    best_pixel_metric = 0

    # load model: auto_resume > resume_model > load_path
    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)
    lastest_model = os.path.join(config.save_path, "ckpt_best.pth.tar")
    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model
    if resume_model:
        # best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)
        checkpoint_data = load_state(resume_model, model, optimizer=optimizer, return_full_checkpoint=True)
        if isinstance(checkpoint_data, tuple):
            best_metric, last_epoch = checkpoint_data
        else:
            best_metric = checkpoint_data.get("best_metric", 0)
            last_epoch = checkpoint_data.get("epoch", 0)
            # Load additional best metrics if available
            best_std_metric = checkpoint_data.get("best_std_metric", 0)
            best_max_metric = checkpoint_data.get("best_max_metric", 0)
            best_pixel_metric = checkpoint_data.get("best_pixel_metric", 0)
            
        if logger:
            logger.info(f"Resumed training from epoch {last_epoch} with best metric {best_metric}")
            logger.info(f"Best std metric: {best_std_metric}, Best max metric: {best_max_metric}, Best pixel metric: {best_pixel_metric}")
            logger.info(f"Resume model path: {resume_model}")
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)
        if logger:
            logger.info(f"Loaded model from: {load_path}")

    # Build dataloader
    train_loader, val_loader = build_dataloader(config.dataset, class_name=dataset_class_name)
    if args.evaluate:
        validate(val_loader, model)
        return

    criterion = build_criterion(config.train_diffusion_criterion)

    for epoch in range(last_epoch, config.diffusion_trainer.max_epoch):
        # Log current epoch info at start of training
        if epoch == last_epoch and logger:
            logger.info(f"Starting training from epoch {epoch + 1}/{config.diffusion_trainer.max_epoch}")
        
        last_iter = epoch * len(train_loader)
        train_diffusion_one_epoch(
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            tb_logger,
            criterion,
            frozen_layers,
        )
        lr_scheduler.step(epoch)

        if (epoch + 1) % config.diffusion_trainer.val_freq_epoch == 0:
            ret_metrics = validate(val_loader, model, epoch)
            # Extract individual metrics
            ret_std_metric = ret_metrics.get("mean_std_auc", 0)
            ret_max_metric = ret_metrics.get("mean_max_auc", 0)
            ret_pixel_metric = ret_metrics.get("mean_pixel_auc", 0)
            
            ret_key_metric = ret_metrics[key_metric]
            # Check if each metric is the best
            is_best_std = ret_std_metric >= best_std_metric
            is_best_max = ret_max_metric >= best_max_metric
            is_best_pixel = ret_pixel_metric >= best_pixel_metric
            is_best_default = ret_key_metric >= best_metric
            
            # Update best metrics
            best_std_metric = max(ret_std_metric, best_std_metric)
            best_max_metric = max(ret_max_metric, best_max_metric)
            best_pixel_metric = max(ret_pixel_metric, best_pixel_metric)
            
            best_metric = max(ret_key_metric, best_metric)
            checkpoint_state = {
                "epoch": epoch + 1,
                "arch": config.get('diffusion_net', []),
                "state_dict": model.state_dict(),
                "best_metric": best_metric,
                "best_std_metric": best_std_metric,
                "best_max_metric": best_max_metric,
                "best_pixel_metric": best_pixel_metric,
                "optimizer": optimizer.state_dict(),
            }
            # Save best models for each metric
            # save_checkpoint(checkpoint_state, is_best_std, config, "std")
            # save_checkpoint(checkpoint_state, is_best_max, config, "max")
            # save_checkpoint(checkpoint_state, is_best_pixel, config, "pixel")
            save_checkpoint(checkpoint_state, is_best_default, config, "default")


def train_diffusion_one_epoch(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    frozen_layers,
):
    batch_time = AverageMeter(config.diffusion_trainer.print_freq_step)
    data_time = AverageMeter(config.diffusion_trainer.print_freq_step)
    losses = AverageMeter(config.diffusion_trainer.print_freq_step)
    model.train()
    # freeze selected layers
    for layer in frozen_layers:
        module = getattr(model, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
    
    logger = logging.getLogger("global_logger")
    end = time.time()

    for i, input in enumerate(train_loader):
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_lr()[0]
        data_time.update(time.time() - end)
        # forward
        input_dev = to_device(input, device='cuda')
        image = input_dev["image"]

        outputs = model(image)
        loss = 0
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss += weight * criterion_loss(outputs)
        
        losses.update(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        if config.diffusion_trainer.get("clip_max_norm", None):
            max_norm = config.diffusion_trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        if (curr_step + 1) % config.diffusion_trainer.print_freq_step == 0:
            if tb_logger:
                tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
                tb_logger.add_scalar("lr", current_lr, curr_step + 1)
                tb_logger.flush()
            if logger:
                logger.info(
                    "Epoch: [{0}/{1}]\t"
                    "Iter: [{2}/{3}]\t"
                    "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                    "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                    "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                    "LR {lr:.5f}\t".format(
                        epoch + 1,
                        config.diffusion_trainer.max_epoch,
                        curr_step + 1,
                        len(train_loader) * config.diffusion_trainer.max_epoch,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        lr=current_lr,
                    )
                )
        end = time.time()


def validate(val_loader, model, epoch=None):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)
    model.eval()
    
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.test_criterion)
    end = time.time()

    os.makedirs(config.evaluator.eval_dir, exist_ok=True)
    aug_images_list = [] if (args.evaluate and config.evaluator.get("vis_compound", None)) else None

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            input_dev = to_device(input, device=getattr(model, 'device', 'cuda'))
            image = input_dev["image"] # norm [0,1]

            aug_image = model.gumbel_aug(image)
            reconstructed = model.diffusion.reconstruct(
                    x=aug_image, 
                    y0=aug_image, 
                    timesteps=None
            )
            # MSE between original and reconstructed
            recon_error = torch.mean((aug_image - reconstructed) ** 2, dim=1, keepdim=True)  # [B, 1, H, W]
            outputs = {
                    "pred": recon_error,  # Anomaly map
                    "filename": input_dev["filename"],
                    "height": input_dev["height"],
                    "width": input_dev["width"],
                    "clsname": input_dev["clsname"],
                    "mask": input_dev["mask"],
            }

            dump(config.evaluator.eval_dir, outputs)
            # record loss using original model outputs
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)  
            num = len(outputs["filename"])
            losses.update(loss.item(), num)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % config.diffusion_trainer.print_freq_step == 0 and logger:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )
                
    # gather final results
    final_loss = losses.avg
    total_num = losses.count

    if logger:
        logger.info("Gathering final results ...")
        # total loss
        logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num))
    fileinfos, preds, masks = merge_together(config.evaluator.eval_dir)
    shutil.rmtree(config.evaluator.eval_dir)
    # evaluate, log & vis
    ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)
    log_metrics(ret_metrics, config.evaluator.metrics)
    
    if args.evaluate and config.evaluator.get("vis_compound", None):
        # visualize_compound(
        #     fileinfos,
        #     preds,
        #     masks,
        #     config.evaluator.vis_compound,
        #     config.dataset.image_reader,
        # )
        if aug_images_list is not None and len(aug_images_list) > 0:
            # Concatenate all augmented images
            aug_images_tensor = torch.cat(aug_images_list, dim=0)
            visualize_compound_aug(
                fileinfos,
                aug_images_tensor,
                preds,
                masks,
                config.evaluator.vis_compound,
                config.dataset.image_reader
            )
    if args.evaluate and config.evaluator.get("vis_single", None):
        visualize_single(
            fileinfos,
            preds,
            config.evaluator.vis_single,
            config.dataset.image_reader,
        )
    model.train()
    return ret_metrics


def main():
    args = parser.parse_args()
    if args.training_phase == "contrastive":
        train_contrastive()
    elif args.training_phase == "diffusion":
        train_diffusion()
    else:
        raise ValueError(f"Unknown training phase: {args.training_phase}")


if __name__ == "__main__":
    main()



