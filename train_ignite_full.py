# %%
import os.path
import sys
from ignite.engine import Events, Engine
import torch
from utils.get_vkitti_dataset_full import get_dataloaders
from utils.tensorize_batch import tensorize_batch
from eval_coco import evaluate

import temp_variables
import constants
import models
from utils import map_hasty
from utils import get_splits
import config_kitti_2 as config_kitti
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


## this is panoptic segmentation training with no depth estimation using "vkitti full" dataset
# %%

def __update_model(trainer_engine, batch):
    model.train()
    optimizer.zero_grad()
    # imgs, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt, _ = batch
    imgs, ann, _, _, _, _, _, _ = batch
    # imgs, annotations = batch[0], batch[1]

    imgs = list(img for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()}
                   for t in ann]

    imgs = tensorize_batch(imgs, device)

    loss_dict = model(imgs, anns=annotations)

    losses = sum(loss for loss in loss_dict.values())
    
    i = trainer_engine.state.iteration
    writer.add_scalar("Loss/train/iteration", losses, i)

    for key in loss_dict.keys():
        writer.add_scalar("Loss/train/{}".format(key), loss_dict[key], i)

    losses.backward()

    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

    optimizer.step()

    return losses




# %% Define Event listeners


def __log_training_loss(trainer_engine):

    current_lr = None

    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']

    batch_loss = trainer_engine.state.output
    state_epoch = trainer_engine.state.epoch
    max_epochs = trainer_engine.state.max_epochs
    i = trainer_engine.state.iteration
    text = "Epoch {}/{} : {} - batch loss: {:.2f}, LR:{}".format(
        state_epoch, max_epochs, i, batch_loss, current_lr)

    sys.stdout = open(train_res_file, 'a+')
    print(text)


def __log_validation_results(trainer_engine):
    batch_loss = trainer_engine.state.output
    state_epoch = trainer_engine.state.epoch
    max_epochs = trainer_engine.state.max_epochs
    weights_path = "{}{}_{}_loss_{}.pth".format(
        constants.MODELS_LOC, config_kitti.MODEL, config_kitti.BACKBONE, batch_loss)
    state_dict = model.state_dict()
    torch.save(state_dict, weights_path)

    sys.stdout = open(train_res_file, 'a+')
    print("Model weights filename: ", weights_path)
    text = "Validation Results - Epoch {}/{} batch_loss: {:.2f}".format(
        state_epoch, max_epochs, batch_loss)
    sys.stdout = open(train_res_file, 'a+')
    print(text)

    # evaluate(model=model, weights_file=weights_path,
    #          data_loader_val=data_loader_val)
   
    writer.add_scalar("Loss/train/epoch", batch_loss, state_epoch)

    scheduler.step()

    with torch.no_grad():

        model.eval()
        # torch.cuda.empty_cache()
        # for imgs, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt, _ in data_loader_train:
        for imgs, anns, _, _, _, _, _, _ in data_loader_train:

        # imgs, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt = next(
        #     iter(data_loader_train))

            imgs = list(img for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()}
                   for t in anns]

            imgs = tensorize_batch(imgs, device)



            outputs = model(imgs, anns=annotations)
            
            

            for idx in range(len(outputs)):
                
                

                img = imgs[idx]
                
                writer.add_image("eval/src_img", img, state_epoch, dataformats="CHW")

                label = anns[idx]["semantic_mask"].cpu().numpy()
                pred = outputs[idx]["semantic_logits"]
                pred = F.softmax(pred, dim=0)
                pred = torch.argmax(pred, dim=0).squeeze(1)
                pred = pred.cpu().numpy()

                writer.add_image("eval/semantic_gt", label, state_epoch, dataformats="HW")
                writer.add_image("eval/semantic_pred", pred, state_epoch, dataformats="HW")
            
            break



if __name__ == "__main__":

    train_res_file = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), constants.RES_LOC, constants.VKITTI_TRAIN_RES_FILENAME_EffPS_NO_INSTANCE)

    with open(train_res_file, "w+") as training_results:
        training_results.write(
            "----- TRAINING RESULTS - VKitti Semantic----"+"\n")
    # Set device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ", device)
    temp_variables.DEVICE = device
    # Empty cuda cache
    torch.cuda.empty_cache()

    # Get model according to config
    model = models.get_model_by_name(config_kitti.MODEL)
    # move model to the right device
    model.to(device)

    print(torch.cuda.memory_allocated(device=device))
    # Define params
    params = [p for p in model.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(
    #     params, lr=0.005, momentum=0.9, weight_decay=0.00005)

    optimizer = torch.optim.SGD(
        params, lr=0.0016, momentum=0.9, weight_decay=0.00005)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65, 80, 85, 90], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    data_loader_train = None
    data_loader_val = None

    if config_kitti.USE_PREEXISTING_DATA_LOADERS:
        data_loader_train = torch.load(config_kitti.DATA_LOADER_TRAIN_FILANME)
        data_loader_val = torch.load(config_kitti.DATA_LOADER_VAL_FILENAME)

    # else:
    #     imgs_root = os.path.join(os.path.dirname(os.path.abspath(
    #         __file__)), "data_vkitti/vkitti_2.0.3_rgb/")

    #     depth_root = os.path.join(os.path.dirname(os.path.abspath(
    #         __file__)), "data_vkitti/vkitti_2.0.3_depth/")

    #     semseg_root = os.path.join(os.path.dirname(os.path.abspath(
    #         __file__)), "data_vkitti/semseg_bin/")

    #     data_loader_train, data_loader_val = get_dataloaders(batch_size=config_kitti.BATCH_SIZE,
    #                                                          imgs_root=imgs_root,
    #                                                          depth_root=depth_root,
    #                                                          semseg_root=semseg_root,
    #                                                          annotation=config_kitti.COCO_ANN,
    #                                                          split=True,
    #                                                          val_size=0.20,
    #                                                          n_samples=config_kitti.MAX_TRAINING_SAMPLES)

    #     # save data loaders
    #     data_loader_train_filename = os.path.join(os.path.dirname(
    #         os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.KITTI_DATA_LOADER_TRAIN_FILANME)

    #     data_loader_val_filename = os.path.join(os.path.dirname(
    #         os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.KITTI_DATA_LOADER_VAL_FILENAME)

    #     torch.save(data_loader_train, data_loader_train_filename)
    #     torch.save(data_loader_val, data_loader_val_filename)

    scheduler = MultiStepLR(optimizer, milestones=[65, 80, 85, 90], gamma=0.1)
    ignite_engine = Engine(__update_model)

    # ignite_engine.add_event_handler(Events.ITERATION_STARTED, scheduler)
    ignite_engine.add_event_handler(
        Events.ITERATION_COMPLETED(every=50), __log_training_loss)
    ignite_engine.add_event_handler(
        Events.EPOCH_COMPLETED, __log_validation_results)
    ignite_engine.run(data_loader_train, config_kitti.MAX_EPOCHS)
    writer.flush()



# data_loader_train, data_loader_val = get_dataloaders(batch_size=config_kitti.BATCH_SIZE,
#                                                              imgs_root=imgs_root,
#                                                              depth_root=depth_root,
#                                                              annotation=config_kitti.COCO_ANN
#                                                              split=True,
#                                                              val_size=0.20,
#                                                              n_samples=config_kitti.MAX_TRAINING_SAMPLES)
