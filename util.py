from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


#nltk.download('wordnet')
# 创建两个相同图像的裁剪
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

#计算和储存平均值和当前值
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#标签平滑处理
def label_smoothing(target,label_smooth=0.1):
    target = (1.0-label_smooth)*target + label_smooth/2
    return target

#计算准确率
def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = 1
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        k=1
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res=correct_k.mul_(100.0 / batch_size)
        # res = []
        # for k in topk:
        #     correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #     res.append(correct_k.mul_(100.0 / batch_size))
        return res

#计算宏观F1分数   
def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(true, preds, average='macro')
    return p_macro, r_macro, f_macro

#计算加权F1分数
def weighted_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(true, preds, average='binary')
    return p_macro, r_macro, f_macro

# # 绘制混淆矩阵的函数
# def plot_confusion_matrix(y_true, y_pred, file_path=None):
#     preds = np.argmax(y_pred, axis=-1)  # 获取预测的类别标签
#     cm = confusion_matrix(y_true, preds)  # 计算混淆矩阵
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化混淆矩阵

#     plt.figure(figsize=(6, 6))
#     sns.heatmap(cm_normalized, annot=True, cmap='YlGnBu', fmt=".4f", cbar=True)  # 绘制热力图
#     plt.title(title)
#     plt.ylabel('True Lable')
#     plt.xlabel('Predict Lable')
#     if file_path:
#         plt.savefig(file_path)  # 保存图像到指定路径
#     else:
#         plt.show()  # 显示图像
#     plt.close()

# 调整学习率
def adjust_learning_rate(args, optimizer, epoch):
    
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        if args.cosine:
            eta_min = lr * (args.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (
                    1 + math.cos(math.pi * epoch / args.epochs)) / 2
        else:
            steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
            if steps > 0:
                lr = lr * (args.lr_decay_rate ** steps)
        param_group['lr'] = lr

#预热学习率
def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

#设置优化器
def set_optimizer(opt, model):
    # model = model.module
    param_optimizer = list(model.named_parameters())
    param_bert = list(model.roberta.named_parameters())
    param_vit = list(model.vit.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd, pd in param_bert)], 'lr': opt.global_pre,
         'weight_decay': 1e-4},
        {'params': [p for n, p in param_optimizer if not (any(nd in n for nd, pd in param_bert) or any(nd in n for nd, pd in param_vit))]},
        {'params': model.vit.parameters(), 'lr': opt.global_pre, 'weight_decay': 1e-4}
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=opt.global_learning_rate, weight_decay=opt.global_weight_decay)
    return optimizer

#保存模型
def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

# 读取image-text 
image_text_dir = '/root/autodl-tmp/DMSD-CL/data/clean_data/clean_image_text.txt'
def get_image_text():
    image_text = {}
    with open(image_text_dir) as f:
        for line in f.readlines():
            sp = line.strip().split()
            if sp[0] not in image_text.keys():
                image_text[sp[0]] = " ".join(sp[1:])
    return image_text

# 读取text, label
train_dir = '/root/autodl-tmp/DMSD-CL/data/clean_data/train.txt'
train_glm_dir = '/root/autodl-tmp/DMSD-CL/glm/image_train_new.csv'
valid_dir = '/root/autodl-tmp/DMSD-CL/data/clean_data/valid.txt'
valid_glm_dir = '/root/autodl-tmp/DMSD-CL/glm/image_valid_new.csv'
test_dir = '/root/autodl-tmp/DMSD-CL/data/clean_data/test.txt'
test_glm_dir = '/root/autodl-tmp/DMSD-CL/glm/image_test_new.csv'

#读取和处理训练数据
def get_train():
    with open("/root/autodl-tmp/DMSD-CL/data/train_id.txt","r") as f:
        train_ids = eval(f.read())
    all_text={}
    all_label={}
    df = pd.read_csv(train_glm_dir)
    all_glm={}
    with open(train_dir) as f:
        for line in f.readlines():
            sp = line.strip().split()
            sp_1 = sp[1:-1]
            sp_1[0] = sp_1[0][1:]
            sp_1[-1] = sp_1[-1][:-2]
            sp_label = int(sp[-1][:1])
            if sp[0][2:-2] not in all_text.keys() and sp[0][2:-2] in train_ids:
                all_text[sp[0][2:-2]] = " ".join(sp_1)
                all_label[sp[0][2:-2]] = sp_label
    
    # 遍历DataFrame中的每一行
    for index, row in df.iterrows():
        # 从image_path中提取ID（假设ID是文件名的整数部分）
        image_path = row['image_path']
        image_id = os.path.basename(image_path).split('.')[0]  # 例如 '715681844763566080'
        
        # 获取描述文本
        description = row['describe']
        
        # 将ID和对应的描述保存到字典中
        all_glm[image_id] = description


    return all_text,all_label,all_glm

def get_valid():
    all_text = {}
    all_label = {}    
    with open(valid_dir) as f:
        for line in f.readlines():
            sp = line.strip().split()
            sp_1 = sp[1:-2]
            sp_1[0] = sp_1[0][1:]
            sp_1[-1] = sp_1[-1][:-2]
            sp_label = int(sp[-1][:1])
            if sp[0][2:-2] not in all_text.keys():
                all_text[sp[0][2:-2]] = " ".join(sp_1)
                all_label[sp[0][2:-2]] = sp_label

    df = pd.read_csv(valid_glm_dir)
    all_glm={}
    # 遍历DataFrame中的每一行
    for index, row in df.iterrows():
        # 从image_path中提取ID（假设ID是文件名的整数部分）
        image_path = row['image_path']
        image_id = os.path.basename(image_path).split('.')[0]  # 例如 '715681844763566080'
        
        # 获取描述文本
        description = row['describe']
        
        # 将ID和对应的描述保存到字典中
        all_glm[image_id] = description
    
    return all_text, all_label, all_glm




def get_test():
    all_text = {}
    all_label = {}
    with open(test_dir) as f:
        for line in f.readlines():
            sp = line.strip().split()
            sp_1 = sp[1:-2]
            sp_1[0] = sp_1[0][1:]
            sp_1[-1] = sp_1[-1][:-2]
            sp_label = int(sp[-1][:1])
            if sp[0][2:-2] not in all_text.keys():
                all_text[sp[0][2:-2]] = " ".join(sp_1)
                all_label[sp[0][2:-2]] = sp_label

    df = pd.read_csv(test_glm_dir)
    all_glm={}
    # 遍历DataFrame中的每一行
    for index, row in df.iterrows():
        # 从image_path中提取ID（假设ID是文件名的整数部分）
        image_path = row['image_path']
        image_id = os.path.basename(image_path).split('.')[0]  # 例如 '715681844763566080'
        
        # 获取描述文本
        description = row['describe']
        
        # 将ID和对应的描述保存到字典中
        all_glm[image_id] = description

    return all_text, all_label, all_glm

def get_opposite():
    opposite_text = {}
    with open(r"/root/autodl-tmp/DMSD-CL/data/clean_data/opposite.txt") as f:
        for line in f.readlines():
            line = line.strip()
            id = line.split('[')[1].split(',')[0]
            text = line.split(',',1)[1].rsplit(',',1)[0]
            opposite_text[id] = text
    return opposite_text

#out-of-distribution
def get_ood():
    ood_text = {}
    ood_label = {}
    with open(r"/root/autodl-tmp/DMSD-CL/data/clean_data/ood_all.txt") as f:
        for line in f.readlines():
            line = line.strip()
            id = line.split('[')[1].split(',')[0]
            text = line.split(',',1)[1].rsplit(',',1)[0]
            label = line.rsplit(',',1)[1].split(']')[0]
            ood_text['o'+id] = text
            ood_label['o'+id] = int(label)
    return ood_text, ood_label

# 文本增强 - 替换同义词
def eda_SR(originalSentence, n):
  """
  Paper Methodology -> Randomly choose n words from the sentence that are not stop words. 
                       Replace each of these words with one of its synonyms chosen at random.
  originalSentence -> The sentence on which EDA is to be applied
  n -> The number of words to be chosen for random synonym replacement
  """
  stops = set(stopwords.words('english'))
  splitSentence = list(originalSentence.split(" "))
  splitSentenceCopy = splitSentence.copy()
  # Since We Make Changes to The Original Sentence List The Indexes Change and Hence an initial copy proves useful to get values
  ls_nonStopWordIndexes = []
  for i in range(len(splitSentence)):
    if splitSentence[i].lower() not in stops:
      ls_nonStopWordIndexes.append(i)
  if (n > len(ls_nonStopWordIndexes)):
    return originalSentence
  for i in range(n):
    indexChosen = random.choice(ls_nonStopWordIndexes)
    ls_nonStopWordIndexes.remove(indexChosen)
    synonyms = []
    originalWord = splitSentenceCopy[indexChosen]
    for synset in wordnet.synsets(originalWord):
      for lemma in synset.lemmas():
        if lemma.name() != originalWord:
          synonyms.append(lemma.name())
    if (synonyms == []):
      continue
    splitSentence[indexChosen] = random.choice(synonyms).replace('_', ' ')
  return " ".join(splitSentence)

#随机选择一个单词找到它的同义词插入到句子里面不同位置n次
def eda_RI(originalSentence, n):
  """
  Paper Methodology -> Find a random synonym of a random word in the sentence that is not a stop word. 
                       Insert that synonym into a random position in the sentence. Do this n times
  originalSentence -> The sentence on which EDA is to be applied
  n -> The number of times the process has to be repeated
  """
  stops = set(stopwords.words('english'))
  splitSentence = list(originalSentence.split(" "))
  splitSentenceCopy = splitSentence.copy() 
  # Since We Make Changes to The Original Sentence List The Indexes Change and Hence an initial copy proves useful to get values
  ls_nonStopWordIndexes = []
  for i in range(len(splitSentence)):
    if splitSentence[i].lower() not in stops:
      ls_nonStopWordIndexes.append(i)
  if (n > len(ls_nonStopWordIndexes)):
    return originalSentence
  WordCount = len(splitSentence)
  for i in range(n):
    indexChosen = random.choice(ls_nonStopWordIndexes)
    ls_nonStopWordIndexes.remove(indexChosen)
    synonyms = []
    originalWord = splitSentenceCopy[indexChosen]
    for synset in wordnet.synsets(originalWord):
      for lemma in synset.lemmas():
        if lemma.name() != originalWord:
          synonyms.append(lemma.name())
    if (synonyms == []):
      continue
    splitSentence.insert(random.randint(0,WordCount-1), random.choice(synonyms).replace('_', ' '))
  return " ".join(splitSentence)

# 文本增强 - 交换单词位置
def eda_RS(originalSentence, n):
  """
  Paper Methodology -> Find a random synonym of a random word in the sentence that is not a stop word. 
                       Insert that synonym into a random position in the sentence. Do this n times
  originalSentence -> The sentence on which EDA is to be applied
  n -> The number of times the process has to be repeated
  """
  splitSentence = list(originalSentence.split(" "))
  WordCount = len(splitSentence)
  for i in range(n):
    firstIndex = random.randint(0,WordCount-1)
    secondIndex = random.randint(0,WordCount-1)
    while (secondIndex == firstIndex and WordCount != 1):
      secondIndex = random.randint(0,WordCount-1)
    splitSentence[firstIndex], splitSentence[secondIndex] = splitSentence[secondIndex], splitSentence[firstIndex]
  return " ".join(splitSentence)

#文本增强 - 随机删除单词
def eda_RD(originalSentence, p):
  """
  Paper Methodology -> Randomly remove each word in the sentence with probability p.
  originalSentence -> The sentence on which EDA is to be applied
  p -> Probability of a Word Being Removed
  """
  og = originalSentence
  if (p == 1):
      raise Exception("Always an Empty String Will Be Returned") 
  if (p > 1 or p < 0):
    raise Exception("Improper Probability Value")
  splitSentence = list(originalSentence.split(" "))
  lsIndexesRemoved = []
  WordCount = len(splitSentence)
  for i in range(WordCount):
    randomDraw = random.random()
    if randomDraw <= p:
      lsIndexesRemoved.append(i)
  lsRetainingWords = []
  for i in range(len(splitSentence)):
    if i not in lsIndexesRemoved:
      lsRetainingWords.append(splitSentence[i])
  if (lsRetainingWords == []):
    return og
  return " ".join(lsRetainingWords)

# 选择文本增强的方法
def text_aug(seq):
    p=random.randint(1,10)
    if p<=8:
        seq=eda_SR(seq,3)
    p=random.randint(1,10)
    if p<=4:
        seq=eda_RI(seq,3)
    p=random.randint(1,10)
    if p<=4:
        seq=eda_RS(seq,3)
    p=random.randint(1,10)
    if p<=2:
        seq=eda_RD(seq,0.2)
    return seq

if __name__ == '__main__':
  import argparse
  import os

    
    
    
