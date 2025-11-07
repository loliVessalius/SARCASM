import os
import argparse
import time
import random
import pickle
import numpy as np
import torch
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch import cos_, optim

from data_utils import RGDataset,RG2Dataset
from modely_test1 import DMSD
import torch.nn.functional as F
from losses import SupConLoss2
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, accuracy,macro_f1,weighted_f1
from util import set_optimizer, save_model
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from ps_loss import Corr_loss,Cos_loss
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix
import skimage.transform

os.environ["CUDA_VISIBLE_DEVICES"] ="0"
log_dir = '/root/autodl-tmp/DMSD-CL/main/logs/tensorboard'
save_folder = '/root/autodl-tmp/DMSD-CL/main/saved_models'
save_dir = '/root/autodl-tmp/DMSD-CL/main/saved_models/attention_maps'
confusion_dir ='/root/autodl-tmp/DMSD-CL/main/saved_models/confusion_matrix'
distance_dir='/root/autodl-tmp/DMSD-CL/main/saved_models/distance'
batch_size=18
writer = SummaryWriter(log_dir)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument("--data_path",default= '/root/autodl-tmp/DMSD-CL/data',type=str)
    parser.add_argument("--image_path",default='/root/autodl-tmp/dataset_image',type=str)
    parser.add_argument("--save_folder",default=save_folder,type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_len",default=100,type=int,
                        help="Total number of text.can not alter")
    
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch_size')
    parser.add_argument('--seed',type=int,default=42,
                        help="random seed for initialization")
    parser.add_argument("--alpha", default= '0.1',
                        type=float)
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')
    # optimization
    parser.add_argument("--global_learning_rate",
                        default=1e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--global_pre",
                        default=1e-5, 
                        type=float)
    parser.add_argument("--global_weight_decay",
                        default=1e-4,
                        type=float)
    parser.add_argument('--lr_decay_epochs', type=str, default='50',
                        help='where to decay lr, can be a list')
    
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

    # other setting
    parser.add_argument('--cosine', action='store_true',default=True,
                        help='using cosine annealing')

    opt = parser.parse_args()
    
    # 创建保存路径如果不存在
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    return opt

# 数据加载器
def set_loader(opt):
    # construct data loader
    train_dataset = RGDataset(os.path.join(opt.data_path, 'train_id.txt'),opt.image_path, opt.max_len)
    valid_dataset = RG2Dataset(os.path.join(opt.data_path, 'valid_id.txt'),opt.image_path, opt.max_len)
    test_dataset = RG2Dataset(os.path.join(opt.data_path, 'test_id.txt'),opt.image_path, opt.max_len)
    ood_dataset = RG2Dataset(os.path.join(opt.data_path, 'ood_id.txt'),opt.image_path, opt.max_len)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                    num_workers=opt.num_workers, pin_memory=True)
    
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    
    ood_loader = DataLoader(ood_dataset, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    return train_loader,valid_loader,test_loader,ood_loader

# 构建模型和损失函数
def set_model(opt):
    device = torch.device("cuda:0")

    model = DMSD().to(device)
    ce_criterion = torch.nn.CrossEntropyLoss() #交叉熵损失函数
    # cl_criterion = SupConLoss2(temperature=opt.temp,t=0.8) #对比学习损失函数
    # cos_loss = Cos_loss()
    # corr_loss = Corr_loss()
    

    if torch.cuda.is_available():
        model = model.cuda()
        ce_criterion = ce_criterion.cuda()
        cl_criterion = cl_criterion.cuda()
        cos_loss = cos_loss.cuda()
        corr_loss = corr_loss.cuda()
        cudnn.benchmark = True

    return model, ce_criterion, cl_criterion,cos_loss,corr_loss


#执行一个训练周期
def train(train_loader, model, ce_criterion, cl_criterion, optimizer, epoch, opt,cos_loss,corr_loss):
    """one epoch training"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, batch in enumerate(tqdm(train_loader)):
        bert_mask,bert_mask_add, bert_indices_add, images,labels,glm_mask_add, glm_indices_add=batch
        
        bert_mask=torch.cat([bert_mask[0], bert_mask[1],bert_mask[2]], dim=0)
        bert_mask_add=torch.cat([bert_mask_add[0], bert_mask_add[1],bert_mask_add[2]], dim=0)
        bert_indices_add=torch.cat([bert_indices_add[0], bert_indices_add[1],bert_indices_add[2]], dim=0)
        images = torch.cat([images[0], images[1],images[2]], dim=0)  #[bsz*3,3,224,224]
        labels = torch.cat([labels[0], labels[1],labels[2]], dim=0)  #[bsz*3,3,224,224]
        glm_mask_add = torch.cat([glm_mask_add[0],glm_mask_add[1],glm_mask_add[2]],dim=0)
        glm_indices_add = torch.cat([glm_indices_add[0],glm_indices_add[1],glm_indices_add[2]])

        if torch.cuda.is_available():
            bert_mask=bert_mask.cuda(non_blocking=True)
            bert_mask_add=bert_mask_add.cuda(non_blocking=True)
            bert_indices_add=bert_indices_add.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)         #[bsz*3,1]
            glm_mask_add = glm_mask_add.cuda(non_blocking=True)
            glm_indices_add = glm_indices_add.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        output,scale,polar_vector,vs,ts,vl,tl= model(bert_mask,bert_mask_add, bert_indices_add, images,glm_mask_add, glm_indices_add,"train")    #[bsz*3,feature_dim]
        ce_loss = ce_criterion(output,labels)
        # cl_loss = cl_criterion(features,labels)
        # p_loss = cos_loss(polar_vector,labels,output)
        # s_loss = corr_loss(scale,labels)

        #总损失
        # loss = opt.alpha*ce_loss+(1-opt.alpha)*cl_loss
        # loss = ce_loss+0.05*p_loss+0.05*s_loss
        loss = ce_loss
        # update metric
        losses.update(loss.item(), bsz)
        acc1= accuracy(output, labels)
        top1.update(acc1[0], bsz)

        # Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg, top1.avg,vs,ts,vl,tl

def eval(val_loader, model, ce_criterion, opt,cos_loss,corr_loss,epoch,mode):
    """validation"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    y_true=[]
    y_pred=[]
    with torch.no_grad():
        
        for idx, batch in enumerate(tqdm(val_loader)):
            bert_mask,bert_mask_add, bert_indices_add, images,labels,glm_mask_add, glm_indices_add,img,tokens,name=batch
            y_true.append(labels.numpy())
            if torch.cuda.is_available():
                bert_mask=bert_mask.cuda()
                bert_mask_add=bert_mask_add.cuda()
                bert_indices_add=bert_indices_add.cuda()
                images = images.cuda()
                labels = labels.cuda()         #[bsz,1]
                glm_mask_add = glm_mask_add.cuda(non_blocking=True)
                glm_indices_add = glm_indices_add.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # forward
            output,scale,polar_vector,attention_map_t,attention_map_v,vs,ts,cs,tl,vl,cl= model(bert_mask,bert_mask_add, bert_indices_add, images,glm_mask_add, glm_indices_add,'test')
            ce_loss = ce_criterion(output, labels)
            # p_loss = cos_loss(polar_vector,labels,output)
            # s_loss = corr_loss(scale,labels)
            loss = ce_loss
            y_pred.append(output.to('cpu').numpy())
            # update metric
            losses.update(loss.item(), bsz)
            acc1= accuracy(output, labels)
            top1.update(acc1[0], bsz)

            # 如果是测试集或OOD测试集，执行可视化
            if mode in ["test", "ood"]:
                visualize_attention_on_image_batch(img, attention_map_t, tokens, epoch, sample_name=name, mode=mode)


   
        y_true=np.concatenate(y_true)
        y_pred=np.concatenate(y_pred)
        precision, recall, F_score = macro_f1(y_true, y_pred)
        w_pre,w_rec,w_f1 = weighted_f1(y_true, y_pred)



            
    return losses.avg, top1.avg, precision, recall, F_score, w_pre,w_rec,w_f1,attention_map_t,attention_map_v, y_true, y_pred,vs,ts,cs,tl,vl,cl,img,tokens,name

# 定义t-SNE降维并绘制函数
def tsne_visualization(vs, ts, vl, tl, epoch, distance_dir,perplexity=30, learning_rate=100):

    vs = vs.detach().cpu().numpy()
    ts = ts.detach().cpu().numpy()
    vl = vl.detach().cpu().numpy()
    tl = tl.detach().cpu().numpy()
    
    # 将所有数据进行合并，便于t-SNE处理
    all_data = np.concatenate([vs, ts, vl, tl], axis=0)
    
    # 使用t-SNE将数据降维到2D
    tsne = TSNE(n_components=2, random_state=42,perplexity=perplexity, learning_rate=learning_rate)
    all_data_2d = tsne.fit_transform(all_data)
    
    # 将数据拆分回四个类别
    vs_con_2d = all_data_2d[:len(vs)]
    ts_con_2d = all_data_2d[len(vs):len(vs) + len(ts)]
    vl_con_2d = all_data_2d[len(vs) + len(ts):len(vs) + len(ts) + len(vl)]
    tl_con_2d = all_data_2d[len(vs) + len(ts) + len(vl):]
    
    # 创建颜色列表
    colors = sns.color_palette("deep", 4)
    
    # 创建绘图
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 绘制2D散点图，叠加每个类别的点
    ax.scatter(vs_con_2d[:, 0], vs_con_2d[:, 1], c=[colors[0]], label='Visual Inter-Modality')
    ax.scatter(ts_con_2d[:, 0], ts_con_2d[:, 1], c=[colors[1]], label='Textual Inter-Modality')
    ax.scatter(vl_con_2d[:, 0], vl_con_2d[:, 1], c=[colors[2]], label='Visual Intra-Modality')
    ax.scatter(tl_con_2d[:, 0], tl_con_2d[:, 1], c=[colors[3]], label='Textual Intra-Modality')

    # 设置标题和图例
    # ax.set_title(f'2D t-SNE Projection of Tensors (Epoch {epoch})')
    ax.legend()

    # 确保输出目录存在
    if not os.path.exists(distance_dir):
        os.makedirs(distance_dir)
    
    # 保存图片到指定路径
    tsne_save_path = os.path.join(distance_dir, f'2d_tsne_visualization_epoch_{epoch}.png')
    plt.savefig(tsne_save_path)
    plt.close()  # 关闭图形防止内存泄漏

def visualize_attention_on_image(img, attention_map, tokens, epoch,sample_name,mode, patch_size=14):

    # 限制 tokens 数量不超过 18 个
    if len(tokens) > 10:
        print(f"Skipping visualization for {sample_name}: token count {len(tokens)} exceeds 10.")
        return  # 直接跳过可视化


    img_height, img_width, _ = img.shape
    
    # 创建子图
    fig, axs = plt.subplots(1, len(tokens), figsize=(20, 5))
    
    # 处理每个token
    for idx, token in enumerate(tokens):
        ax = axs[idx]
        ax.imshow(img)
        
        # 生成对应的注意力图
        attention = attention_map[:, idx].reshape((patch_size, patch_size))
        attention_resized = skimage.transform.resize(attention, (img_height, img_width), mode='reflect', anti_aliasing=True)
        # 叠加注意力权重到图像上，使用Alpha通道来突出显示
        ax.imshow(attention_resized , alpha=0.5)
        ax.set_title(f'{token} ({attention.mean()*10:.2f})')
        ax.axis('off')
    

    epoch_dir = os.path.join(save_dir, str(epoch))  # 以 epoch 为目录名
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
        # 确保输出目录存在
    if mode =="test":
        save_path = os.path.join(epoch_dir, f'{sample_name}.png')
    if mode == "ood":
        save_path = os.path.join(epoch_dir, f'ood{sample_name}.png')
    plt.savefig(save_path)  # 保存热力图为图片
    plt.close() 


def visualize_attention_on_image_batch(img_batch, attention_map_batch, tokens, epoch, sample_name,mode, patch_size=14):

    batch_size = img_batch.shape[0]  # 获取批次大小

    tokens_list = [sentence.split() for sentence in tokens]

    
    for i in range(batch_size):
        tokens = tokens_list[i]
        name = sample_name[i]
        img = img_batch[i]  # 获取当前批次中的第 i 个图像
        attention_map = attention_map_batch[i]  # 获取当前图像的注意力图


        # if len(tokens) == 1 and isinstance(tokens[0], str):
        #     tokens = tokens[0].split()  # 将句子拆分为单词列表

        
        # 限制 tokens 数量不超过 10 个
        if len(tokens) > 10:
            # print(f"Skipping visualization for sample {i}: token count {len(tokens)} exceeds 10.")
            continue  # 直接跳过可视化

        img_height, img_width, _ = img.shape  # 获取图像的高度、宽度和通道数

        # 创建子图
        if len(tokens) == 1:
            fig, axs = plt.subplots(1, 1, figsize=(20, 5))  # 只有一个 token
            axs = [axs]  # 转换为列表以便统一处理
        else:
            fig, axs = plt.subplots(1, len(tokens), figsize=(20, 5))  # 多个 token
        


        # # 创建子图
        # fig, axs = plt.subplots(1, len(tokens), figsize=(20, 5))
        
        # 处理每个 token
        for idx, token in enumerate(tokens):
            ax = axs[idx]
            ax.imshow(img)  # 显示原始图像
            
            # 生成对应的注意力图
            attention = attention_map[:, idx].reshape((patch_size, patch_size))
            attention_resized = skimage.transform.resize(attention, (img_height, img_width), mode='reflect', anti_aliasing=True)
            # 叠加注意力权重到图像上，使用 Alpha 通道来突出显示
            ax.imshow(attention_resized, alpha=0.5)
            # ax.set_title(f'{token} ({attention.mean() * 20:.2f})')
            ax.set_title(f'{token}')
            ax.axis('off')
        
        # 创建 epoch 目录
        epoch_dir = os.path.join(save_dir, str(epoch))
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        # 根据 mode 和样本编号生成保存路径
        # sample_name = f"sample_{i}"  # 使用样本编号作为文件名
        if mode == "test":
            save_path = os.path.join(epoch_dir, f'{name}.png')
        elif mode == "ood":
            save_path = os.path.join(epoch_dir, f'ood_{name}.png')

        plt.savefig(save_path)  # 保存热力图为图片
        plt.close(fig)  # 关闭图形防止内存泄漏


def visualize_attention_maps(image, attention_map, tokens, patch_size=14):
    

    img = np.array(image)
    img_height, img_width, _ = img.shape  

    # fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), gridspec_kw={'width_ratios': [10, 1]})


    attention_token_to_image = attention_map.mean(axis=1).reshape((patch_size, patch_size))  # 取每个文本token的平均值
    attention_resized = skimage.transform.resize(attention_token_to_image, (img_height, img_width), mode='reflect', anti_aliasing=True)


    axes[0].imshow(img)
    sns.heatmap(attention_resized, cmap='Oranges', alpha=0.5, ax=axes[0], cbar=True)
    # axes[0].set_title("Text to Image Attention")
    axes[0].axis('off')


    attention_image_to_token = attention_map.mean(axis=0)[:len(tokens)].reshape(len(tokens), 1)   # 取每个图像patch对文本token的平均值
    # print(attention_image_to_token.shape)

    sns.heatmap(attention_image_to_token, cmap='Oranges', annot=True, fmt='.2f', ax=axes[1], yticklabels=tokens, cbar=True)
    # axes[1].set_title("Image to Text Attention")
    axes[1].tick_params(axis='y', rotation=0)



    plt.tight_layout()
    plt.savefig("/root/autodl-tmp/DMSD-CL/attention_map_text_and_image.png")
    plt.show()

def main():
    best_valid_acc = 0
    best_test_acc = 0
    best_ood_acc=0
    opt = parse_option()
    print(opt)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    random.seed(opt.seed)
    # build data loader
    train_loader, valid_loader, test_loader, ood_loader = set_loader(opt)

    # build model and criterion
    model, ce_criterion, cl_criterion,cos_loss,corr_loss= set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train_loss, train_acc,vs,ts,vl,tl= train(train_loader, model, ce_criterion, cl_criterion,optimizer, epoch, opt,cos_loss,corr_loss)
        time2 = time.time()

        #         #向量在三维空间中的分布s
        # # 将张量降维到3D空间
        # colors = sns.color_palette("deep", 4)  # 获取4种柔和的颜色
        # pca = PCA(n_components=2)

        # vs = vs.detach().cpu().numpy()  # 从 GPU 移动到 CPU 并转换为 NumPy 数组
        # ts = ts.detach().cpu().numpy()
        # vl = vl.detach().cpu().numpy()
        # tl = tl.detach().cpu().numpy()
        
        # vs_con_2d = pca.fit_transform(vs)  # 视觉模态
        # ts_con_2d = pca.fit_transform(ts)  # 文本模态
        # vl_con_2d = pca.fit_transform(vl) 
        # tl_con_2d = pca.fit_transform(tl)

        
        # # 创建绘图
        # fig = plt.figure(figsize=(10, 8))
        # # ax = fig.add_subplot(111, projection='3d')
        # ax = plt.gca()
        
        # # 绘制3D散点图
        # ax.scatter(vs_con_2d[:, 0], vs_con_2d[:, 1], c=[colors[0]], label='Visual Inter-Modality')
        # ax.scatter(ts_con_2d[:, 0], ts_con_2d[:, 1], c=[colors[1]], label='Textual Inter-Modality')
        # ax.scatter(vl_con_2d[:, 0], vl_con_2d[:, 1], c=[colors[2]], label='Visual Intra-Modality')
        # ax.scatter(tl_con_2d[:, 0], tl_con_2d[:, 1], c=[colors[3]], label='Textual Intra-Modality')

        # # 设置图例和标签
        # # ax.set_xlabel('Component 1')
        # # ax.set_ylabel('Component 2')
        # # ax.set_zlabel('Component 3')
        # # ax.set_title(f'3D Projection of Tensors (Epoch {epoch})')
        # ax.legend()

        # # 定义保存路径
        # distance_save_path = os.path.join(distance_dir, f'2d_visualization.png')
        
        # # 保存图像
        # plt.savefig(distance_save_path)
        # # plt.close()  # 关闭图形防止内存泄漏
        # print(f'3D visualization for epoch {epoch} saved at: {distance_save_path}')

        # tsne_visualization(vs, ts, vl, tl, epoch, distance_dir)



        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('train_acc', train_acc, global_step=epoch)
        print(f'Train epoch {epoch}, total time {time2 - time1}, train_loss:{train_loss}, train_accuracy:{train_acc}')
        
        val_loss, val_acc, precision, recall, F_score, w_pre,w_rec,w_f1,attention_map_t,attention_map_v,y_true, y_pred,vs,ts,cs,tl,vl,cl,img,tokens,name= eval(valid_loader, model, ce_criterion, opt,cos_loss,corr_loss,epoch, mode="valid")
        print(f'Train epoch {epoch}, valid_loss:{val_loss}, valid_accuracy:{val_acc} \nprecision:{precision}, recall:{recall},F_score:{F_score}\nw_pre:{w_pre}, w_rec:{w_rec}, w_f1:{w_f1}')
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_valid0"  + ".bin"))
            print("better model")
       # eval for one epoch
        test_loss, test_acc, precision, recall, F_score, w_pre,w_rec,w_f1,attention_map_t_test,attention_map_v,y_true,y_pred,vs,ts,cs,tl,vl,cl,img,tokens,name= eval(test_loader, model, ce_criterion, opt,cos_loss,corr_loss,epoch, mode="test")
        # image_1 = Image.open('/root/autodl-tmp/dataset_image/683363557321388032.jpg')
        # tokens_1 = ["feeding", "my", "abs", "nothing", "but", "the", "best", "quality", "beef"]
        # image_2 = Image.open('/root/autodl-tmp/dataset_image/820410362063454210.jpg')
        # tokens_2 = ["foil", "fish", "packets", "with", "spinach", "and", "tomato"]
        # visualize_attention_on_image(img, attention_map_t_test, tokens, epoch,sample_name=name,mode="test")
        # visualize_attention_on_image_batch(img, attention_map_t_test, tokens, epoch,sample_name = name, mode ="test")

        # visualize_attention_maps(img, attention_map_t, tokens,patch_size=14)
    

        #绘制混淆矩阵图
        # confusion_save_path = os.path.join(confusion_dir, f'Confusion_Matrix_epoch_{epoch}.png')
        # preds = np.argmax(y_pred, axis=-1)  # 获取预测的类别标签
        # cm = confusion_matrix(y_true, preds)  # 计算混淆矩阵
        # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化混淆矩阵

        # plt.figure(figsize=(6, 6))
        # custom_cmap = sns.color_palette("vlag", as_cmap=True)
        # sns.heatmap(cm_normalized, annot=True, cmap=custom_cmap, fmt=".4f", cbar=True)  # 绘制热力图
        # plt.title(f"Confusion Matrix (Epoch {epoch})")
        # plt.ylabel('True Lable')
        # plt.xlabel('Predict Lable')
        # plt.savefig(confusion_save_path)  # 保存混淆矩阵为图片
        # plt.close()


        writer.add_scalar('test_acc', test_acc, global_step=epoch)
        writer.add_scalar('F_score', F_score, global_step=epoch)
        writer.add_scalar('w_f1', w_f1, global_step=epoch)
        print(f'Train epoch {epoch}, test_loss:{test_loss}, test_accuracy:{test_acc} \nprecision:{precision}, recall:{recall},F_score:{F_score}\nw_pre:{w_pre}, w_rec:{w_rec}, w_f1:{w_f1}')
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch
            best_test_precision = precision
            best_test_recall = recall
            best_test_F_score = F_score
            best_test_w_pre = w_pre
            best_test_w_rec = w_rec
            best_test_w_f1 = w_f1
            torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_test0"  + ".bin"))
            print("better model")
        ood_loss, ood_acc, precision, recall, F_score, w_pre,w_rec,w_f1,attention_map_t_ood,attention_map_v,y_true,y_pred,vs,ts,cs,tl,vl,cl,img,tokens,name = eval(ood_loader, model, ce_criterion, opt,cos_loss,corr_loss,epoch, mode="ood")
        # image_4 = Image.open('/root/autodl-tmp/dataset_image/683295117847969793.jpg')
        # tokens_4 = ["the","pa","welcome","center" ,"is","completely", "empty", "today"]
        # # visualize_attention_on_image(image_4, attention_map_t_ood, tokens_4, epoch,mode="ood")
        # visualize_attention_on_image_batch(img, attention_map_t_test, tokens, epoch,sample_name = name, mode ="ood")

        writer.add_scalar('OOD_loss', ood_loss, global_step=epoch)
        writer.add_scalar('OOD_acc', ood_acc, global_step=epoch)
        print(f'Train epoch {epoch}, OOD_loss:{ood_loss}, OOD_accuracy:{ood_acc} \nprecision:{precision}, recall:{recall},F_score:{F_score}\nw_pre:{w_pre}, w_rec:{w_rec}, w_f1:{w_f1}')
        if ood_acc > best_ood_acc:
            best_ood_acc = ood_acc
            best_ood_epoch = epoch
            best_ood_precision = precision
            best_ood_recall = recall
            best_ood_F_score = F_score
            best_ood_w_pre = w_pre
            best_ood_w_rec = w_rec
            best_ood_w_f1 = w_f1
            torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_OOD0"  + ".bin"))
            print("better model")
        
                # 使用 add_scalars 方法将不同指标绘制在同一张图上
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss, 'test': test_loss, 'ood': ood_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc, 'test': test_acc, 'ood': ood_acc}, epoch)

    writer.close()
    torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_last"  + ".bin"))

    with open(os.path.join(opt.save_folder, "best_accuracies.txt"), "w") as f:
        f.write('Best test accuracy: {:.2f}\n'.format(best_test_acc) + 
        ' epoch: {} precesion: {:.2f} recall: {:.2f} F_score: {:.2f} w_pre: {:.2f} w_rec: {:.2f} w_f1: {:.2f}\n'.format(
        best_test_epoch, best_test_precision, best_test_recall, best_test_F_score, best_test_w_pre, best_test_w_rec, best_test_w_f1))
        f.write('Best OOD accuracy: {:.2f}\n'.format(best_ood_acc) +    
        ' epoch: {} precesion: {:.2f} recall: {:.2f} F_score: {:.2f} w_pre: {:.2f} w_rec: {:.2f} w_f1: {:.2f}\n'.format(
        best_ood_epoch, best_ood_precision, best_ood_recall, best_ood_F_score, best_ood_w_pre, best_ood_w_rec, best_ood_w_f1))
    

    print('best accuracy: {:.2f}'.format(best_test_acc))
    print('best accuracy: {:.2f}'.format(best_ood_acc))
      

if __name__ == "__main__":
    main()



