import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import torchvision.transforms as transforms
from medpy.metric.binary import dc, asd  # 需要安装medpy库
import torch.nn.functional as F

def adjust_targets(msk):
    # 灰度值为0的像素转换为255
    msk[msk == 0] = 1
    # 灰度值在50到255之间的像素转换为0
    mask = (msk >= 50) & (msk <= 255)
    msk[mask] = 0
    mask=(msk==1)
    msk[mask]=255

    return msk


transform_images = transforms.Compose([
    transforms.Resize((256, 256)),
])

transform_targets = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(lambda img: adjust_targets(img))
])

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        images, targets = data
        images = transform_images(images)
        targets = transform_targets(targets) 

        step += iter
        optimizer.zero_grad()

        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def adjust_mask(msk):
    # 灰度值为0的像素转换为255
    msk[msk == 0] = 1
    # 灰度值在50到255之间的像素转换为0
    mask = (msk >= 50) & (msk <= 255)
    msk[mask] = 0
    mask=(msk==1)
    msk[mask]=255

    return msk



def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            # Resize 输入图像到256x256
            img_resized = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
            img_resized = img_resized.cuda(non_blocking=True).float()

            # 模型预测
            out_resized = model(img_resized)

            # Resize 模型输出回800x800
            out = F.interpolate(out_resized, size=(800, 800), mode='bilinear', align_corners=False)

            # 调整msk的灰度值
            msk_adjusted = adjust_mask(msk.cpu().numpy())
            msk_adjusted = torch.from_numpy(msk_adjusted).cuda(non_blocking=True).float()


            # 计算loss
            loss = criterion(out, msk_adjusted)
            loss_list.append(loss.item())

            # 保存预测和真实值用于计算指标
            out = out.squeeze(1).cpu().detach().numpy()
            msk = msk.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            gts.append(msk_adjusted)

    

        # 计算Dice coefficient 和 ASD
        if epoch % config.val_interval == 0:
            preds = np.vstack(preds)
            gts = np.vstack(gts)
            
            dice_coefficients = [dc(pred, gt) for pred, gt in zip(preds, gts)]
            average_surface_distances = [asd(pred, gt) for pred, gt in zip(preds, gts)]

            mean_dice = np.mean(dice_coefficients)
            mean_asd = np.mean(average_surface_distances)

            log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, mean_dice: {mean_dice}, mean_asd: {mean_asd}'
            print(log_info)
            logger.info(log_info)

        else:
            log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
            print(log_info)
            logger.info(log_info)

    return np.mean(loss_list)

def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            # Resize 输入图像到256x256
            img_resized = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
            img_resized = img_resized.cuda(non_blocking=True).float()

            # 模型预测
            out_resized = model(img_resized)

            # Resize 模型输出回800x800
            out = F.interpolate(out_resized, size=(800, 800), mode='bilinear', align_corners=False)

            # 调整msk的灰度值
            msk_adjusted = adjust_mask(msk.cpu().numpy())
            msk_adjusted = torch.from_numpy(msk_adjusted).cuda(non_blocking=True).float()

            

            # 计算loss
            loss = criterion(out, msk_adjusted)
            loss_list.append(loss.item())

            # 保存预测和真实值用于计算指标
            msk = msk.squeeze(1).cpu().detach().numpy()
            out = out.squeeze(1).cpu().detach().numpy()
            gts.append(msk_adjusted)
            preds.append(out)

            # 每隔一定间隔保存图像
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        # 计算Dice coefficient 和 ASD
        dice_coefficients = [dc(pred, gt) for pred, gt in zip(preds, gts)]
        average_surface_distances = [asd(pred, gt) for pred, gt in zip(preds, gts)]

        mean_dice = np.mean(dice_coefficients)
        mean_asd = np.mean(average_surface_distances)

        # 日志信息
        log_info = f'Test of model, loss: {np.mean(loss_list):.4f}, mean_dice: {mean_dice}, mean_asd: {mean_asd}'
        if test_data_name is not None:
            log_info += f', Test datasets name: {test_data_name}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)