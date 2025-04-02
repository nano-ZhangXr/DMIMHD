import sys
import os
import time
import csv
import random
import logging
import numpy as np
from tqdm import tqdm
from pprint import pformat
from functools import partial
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel

from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import Adam, AdamW

from DynamicModality.ImageExtractor import ImgEncoderGlobal, ImgEncoderLocal
from DynamicModality.TextExtractor import TxtEncoder
from DynamicModality.DynamicModule import InteractionModule

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs


############################################################### Logging ###############################################################

def setting_logger(log_dir):

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # add file handler
    f_handler = logging.FileHandler(os.path.join(log_dir, f"{str(time.time())}-output.log"), mode='w')
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                             datefmt='%m/%d/%Y %H:%M:%S'))
    logger.addHandler(f_handler)

    return logger


################################################################ Utils ################################################################

def send_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


def make_exp_dirs(exp_name):
    day_logs_root = 'logs/' + time.strftime("%Y-%m%d", time.localtime())
    os.makedirs(day_logs_root, exist_ok=True)
    exp_log_path = os.path.join(day_logs_root, exp_name)

    os.makedirs(exp_log_path, exist_ok=True)

    return exp_log_path


def _norm(s):
    return ' '.join(s.strip().split())


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


################################################################ Data #################################################################

# PreProcessing Text
def _process_text(text, tkr, mlen=None):
    if mlen:
        tked = tkr(text, padding=True, truncation=True, max_length=mlen, return_tensors='pt')
    else:

        tked = tkr(text, padding=True, return_tensors='pt')
    tked_ids = tked['input_ids']
    tked_msk = tked['attention_mask']
    stc_lens = tked_msk.sum(dim=1)

    return tked_ids, tked_msk, stc_lens


# PreProcessing Image
def _process_image(image_path):
    image_transform = transforms.Compose([
        transforms.Resize((612, 612)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.568, 0.538, 0.503],
            std=[0.307, 0.300, 0.310]
        )
    ])
    image = Image.open(image_path)
    image = image_transform(image)

    return image


def make_dataloader(split, cfg):
    datalist = []
    if split == 'train':
        file = './dataset/train/train.csv'
        image_dir = './dataset/train/images'
    elif split == 'dev':
        file = './dataset/dev/dev.csv'
        image_dir = './dataset/dev/images'
    else:
        file = './dataset/test/test.csv'
        image_dir = './dataset/test/images'

    with open(file, 'r', encoding='utf-8') as rv:

        data = csv.reader(rv)
        for idx, line in enumerate(data):
            if idx > 0:
                Title, Caption, Sentiment, Emotion, Desire, Inference = line
                data_dic = {"Title": Title, "Caption": Caption, "Sentiment": Sentiment, "Emotion": Emotion,
                            "Desire": Desire, "Inference": Inference}
                datalist.append(data_dic)

    dataset = MyDataset(datalist, cfg, image_dir)

    return dataset


class MyDataset(Dataset):

    def __init__(self, datalist, cfg, image_dir):

        self.data = datalist
        self.cfg = cfg
        self.image_dir = image_dir

    def __len__(self):
        if self.cfg.debug:
            return 513
        else:
            return len(self.data)

    def __getitem__(self, idx):
        ori = self.data[idx]
        
        # Label
        sent_dic = {'negative': 0, 'neutral': 1, 'positive': 2}
        emo_dic = {'fear': 0, 'happiness': 1, 'anger': 2, 'sad': 3, 'neutral': 4, 'disgust': 5}
        des_dic = {'social-contact': 0, 'none': 1, 'vengeance': 2, 'family': 3, 'curiosity': 4, 'romance': 5,
                   'tranquility': 6}
        
        # Text
        context_str = f"{ori['Title']} [SEP] {ori['Caption']}"
        Sentiment = ori['Sentiment']
        Emotion = ori['Emotion']
        Desire = ori['Desire']
        sent_label = torch.LongTensor([sent_dic[Sentiment]])
        emo_label = torch.LongTensor([emo_dic[Emotion]])
        des_label = torch.LongTensor([des_dic[Desire]])
        
        # Image
        image_name = f'{idx + 1}.jpg'
        image_path = os.path.join(self.image_dir, image_name)
        
        res = {
            "context": context_str,
            "sent_label": sent_label,
            "emo_label": emo_label,
            "des_label": des_label,
            "image_dir": image_path,
        }
        
        return res
    
    @staticmethod
    def collate(batch, tkr, max_len):

        batch_context = ['__start__' + i['context'] for i in batch]

        # Texts Process
        new_batch = {}
        input_ids, attention_msk, stc_lens = _process_text(batch_context, tkr, max_len)
        new_batch['input_ids'] = input_ids
        new_batch['attention_msk'] = attention_msk
        new_batch['stc_lens'] = stc_lens
        new_batch['label_des'] = torch.cat([i['des_label'] for i in batch])
        new_batch['label_emo'] = torch.cat([i['emo_label'] for i in batch])
        new_batch['label_sent'] = torch.cat([i['sent_label'] for i in batch])
        
        # Images Process
        image_paths = [os.path.join(i['image_dir']) for i in batch]
        images = [_process_image(image_path) for image_path in image_paths]
        new_batch['image'] = torch.stack(images)

        return new_batch


################################################################ Model ################################################################
class MyModel(nn.Module):
    def __init__(self, cfg, tkr):
        super(MyModel, self).__init__()

        self.cfg = cfg
        self.tkr = tkr

        # Pre-trained BERT
        self.bert = BertModel.from_pretrained(cfg.model.model_name_or_path)
        self.bert.resize_token_embeddings(len(self.tkr))
        self.tsize = self.bert.config.hidden_size

        # Pre-trained ResNet18
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.resnet18 = nn.Sequential(*(list(self.resnet.children())[:-2]))

        # Visual
        self.img_encoder_g = ImgEncoderGlobal(embed_size=cfg.selecting.embed_size)  # Global
        self.img_encoder_l = ImgEncoderLocal(img_dim=800, embed_size=cfg.selecting.embed_size) # Regional

        # Textual
        self.text_encoder = TxtEncoder(self.cfg)
        
        # Dynamic Capsule Network
        self.interaction = InteractionModule(self.cfg.selecting)

        # Loss Function
        self.celoss = nn.CrossEntropyLoss()
        # Dense Layers
        self.proj_des = nn.Sequential(nn.Linear(1536, 128),
                                      nn.LeakyReLU(negative_slope=self.cfg.train.negative_slope, inplace=True),
                                      nn.Dropout(p=self.cfg.train.dropout_des),
                                      nn.Linear(128, 64),
                                      nn.LeakyReLU(negative_slope=self.cfg.train.negative_slope, inplace=True),
                                      nn.Dropout(p=self.cfg.train.dropout_des),
                                      nn.Linear(64, 7))
        self.proj_sent = nn.Sequential(nn.Linear(1536, 128),
                                       nn.LeakyReLU(negative_slope=self.cfg.train.negative_slope, inplace=True),
                                       nn.Dropout(p=self.cfg.train.dropout_sent),
                                       nn.Linear(128, 64),
                                       nn.LeakyReLU(negative_slope=self.cfg.train.negative_slope, inplace=True),
                                       nn.Dropout(p=self.cfg.train.dropout_sent),
                                       nn.Linear(64, 3))
        self.proj_emo = nn.Sequential(nn.Linear(1536, 128),
                                      nn.LeakyReLU(negative_slope=self.cfg.train.negative_slope, inplace=True),
                                      nn.Dropout(p=self.cfg.train.dropout_emo),
                                      nn.Linear(128, 64),
                                      nn.LeakyReLU(negative_slope=self.cfg.train.negative_slope, inplace=True),
                                      nn.Dropout(p=self.cfg.train.dropout_emo),
                                      nn.Linear(64, 6))
  

        # KaiMing's initialization with adjustment for Leaky ReLU
        for layer in [self.proj_des, self.proj_sent, self.proj_emo]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    init.kaiming_uniform_(module.weight, a=self.cfg.train.negative_slope, nonlinearity='leaky_relu')
                    if module.bias is not None:
                        init.zeros_(module.bias)


    def forward(self, input_ids, attention_msk, stc_lens, label_des, label_emo, label_sent,
                mode='train', image=None, **kwargs):
        # CUDA
        device = input_ids.device
        batch_size = input_ids.shape[0]
        input_ids = input_ids.cuda()
        attention_msk = attention_msk.cuda()
        image = image.cuda()
        stc_lens = stc_lens.cuda()
        label_des = label_des.cuda()
        label_emo = label_emo.cuda()
        label_sent = label_sent.cuda()
        
        
        # Text Processing
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_msk)
        text_feat = bert_out.last_hidden_state[:, 0, :]
        stc, wrd = self.text_encoder(input_ids=input_ids, attention_mask=attention_msk)


        # Image Processing
        image_feat = self.resnet(image)
        resnet_out = self.resnet18(image).view(batch_size, 256, -1)
        rgn = self.img_encoder_l(resnet_out)
        img = self.img_encoder_g(image)


        # Dynamic Capsule Network
        Aggr = self.interaction(rgn, img, wrd, stc, stc_lens)


        # Fusion Representation
        fusion = torch.cat((image_feat, Aggr, text_feat), dim=-1)

############################################################# Dense Layer #############################################################
        proj_feat_des = self.proj_des(fusion)  # [batch_size, 7]
        proj_feat_sent = self.proj_sent(fusion)  # [batch_size, 3]
        proj_feat_emo = self.proj_emo(fusion)  # [batch_size, 6]

     
        if mode == 'train':
            loss_des = self.celoss(proj_feat_des, label_des)
            loss_sent = self.celoss(proj_feat_sent, label_sent)
            loss_emo = self.celoss(proj_feat_emo, label_emo)

            loss = loss_des + loss_emo + loss_sent
            return loss

        elif mode in ['eval', 'test']:
            cls_res_des = torch.argmax(proj_feat_des, dim=-1)
            cls_res_sent = torch.argmax(proj_feat_sent, dim=-1)
            cls_res_emo = torch.argmax(proj_feat_emo, dim=-1)
            return label_des, cls_res_des, label_emo, cls_res_emo, label_sent, cls_res_sent

        else:
            raise ValueError('Mode should be among [train, eval, test].')


################################################################ Eval ################################################################
def get_metrics(y_true, y_pred):
    # Calculate p, r, f1, accuracy
    p = precision_score(np.concatenate(y_true), np.concatenate(y_pred), average='macro')
    r = recall_score(np.concatenate(y_true), np.concatenate(y_pred), average='macro')
    f1 = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='macro')
    acc = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))

    return p, r, f1, acc


def log_metrics(ep, mode, metrics, name):
    if accelerator.is_main_process:
        # Logging metrics
        des_res_log = f"{name}: p:{metrics[0] * 100:.3f} r:{metrics[1] * 100:.3f} f1:{metrics[2] * 100:.3f}"
        log.info(f'Epoch {ep} {mode} {des_res_log}')


def eval_net(ep, model, loader, mode='eval'):
    model.eval()
    y_true_des_all, y_pred_des_all = [], []
    y_true_sent_all, y_pred_sent_all = [], []
    y_true_emo_all, y_pred_emo_all = [], []

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            y_true_des, y_pred_des, y_true_emo, y_pred_emo, y_true_sent, y_pred_sent = model(**batch, mode=mode)

            y_true_des = accelerator.gather_for_metrics(y_true_des)
            y_pred_des = accelerator.gather_for_metrics(y_pred_des)
            y_true_sent = accelerator.gather_for_metrics(y_true_sent)
            y_pred_sent = accelerator.gather_for_metrics(y_pred_sent)
            y_true_emo = accelerator.gather_for_metrics(y_true_emo)
            y_pred_emo = accelerator.gather_for_metrics(y_pred_emo)


            y_true_des_all.append(y_true_des.detach().cpu().numpy())
            y_pred_des_all.append(y_pred_des.detach().cpu().numpy())
            y_true_sent_all.append(y_true_sent.detach().cpu().numpy())
            y_pred_sent_all.append(y_pred_sent.detach().cpu().numpy())
            y_true_emo_all.append(y_true_emo.detach().cpu().numpy())
            y_pred_emo_all.append(y_pred_emo.detach().cpu().numpy())


    t = np.concatenate(y_true_des_all)
    p = np.concatenate(y_pred_des_all)
    print("true:", t.shape)
    print("pred:", p.shape)


    des_metrics = get_metrics(y_true_des_all, y_pred_des_all)
    sent_metrics = get_metrics(y_true_sent_all, y_pred_sent_all)
    emo_metrics = get_metrics(y_true_emo_all, y_pred_emo_all)


    log_metrics(ep, mode, des_metrics, 'desire')
    log_metrics(ep, mode, sent_metrics, 'sentiment')
    log_metrics(ep, mode, emo_metrics, 'emotion')
    

    # For best average f1-Score
    average_f1 = des_metrics[2] + sent_metrics[2] + emo_metrics[2]
    return average_f1

############################################################## Training ##############################################################
def run_train(cfg, tkr, train_dataset, eval_dataset, log_path):
    ############### Dataset ###############
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=16, shuffle=True,
                                  collate_fn=partial(train_dataset.collate, tkr=tkr, max_len=cfg.dataset.max_input_len))
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.eval.batch_size, num_workers=16, shuffle=False,
                                 collate_fn=partial(eval_dataset.collate, tkr=tkr, max_len=cfg.dataset.max_input_len))
    ################ Model ################
    model = MyModel(cfg, tkr)

    total_params = sum(p.numel() for p in model.parameters())
    log.info((f'{total_params:,} total parameters.'))
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'{total_trainable_params:,} training parameters.')

    ############### Optimizer & lr_scheduler ###############
    # Sign different lr for pt- and from-scratch layers
    from_scratch_layers = nn.ModuleList([model.proj_des, model.proj_emo, model.proj_sent])
    other_params = filter(lambda p: id(p) not in list(map(id, from_scratch_layers.parameters())), model.parameters())

    optimizer_grouped_parameters = [
        {"params": other_params,
         "weight_decay": cfg.train.weight_decay,
         'lr': cfg.train.pt_lr},

        {"params": from_scratch_layers.parameters(),
         "weight_decay": cfg.train.weight_decay,
         'lr': cfg.train.lr},
    ]
    
    
    ############################## Optimizer ##############################
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=cfg.train.weight_decay)


    if cfg.train.max_steps > 0:
        t_total = cfg.train.max_steps
        cfg.train.max_epoch = cfg.train.max_steps // (
                len(train_dataloader) // cfg.train.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // cfg.train.gradient_accumulation_steps * cfg.train.max_epoch

        
    ############################# LR Scheduler #############################
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.train.warmup_steps,
        num_training_steps=t_total)

  
    if accelerator:
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for part in [model, optimizer, train_dataloader]:
            send_to_device(part, device)

    # Train!
    if accelerator.is_main_process:
        log.info("******** Running training ********")
        log.info("  Num examples = %d", len(train_dataset))
        log.info("  Num Epochs = %d", cfg.train.max_epoch)
        log.info("  Gradient Accumulation steps = %d", cfg.train.gradient_accumulation_steps)
        log.info("  Total optimization steps = %d", t_total)

    print_every = max(int(len(train_dataloader) / 10), 2)
    eval_every = 1

    max_epoch = cfg.train.max_epoch
    best_score = 0
    best_epoch = 0

    for epoch in range(max_epoch):
        
        model.train()
        if accelerator.is_main_process:
            log.info(f"{'-' * 21} Current Epoch:  {epoch} {'-' * 21}")

        time_now = time.time()
        show_loss, show_loss1, show_loss2 = 0, 0, 0
        for idx, batch in enumerate(train_dataloader):

            optimizer.zero_grad()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batch = send_to_device(batch, device)

            # batch = send_to_device(batch, device)
            loss = model(**batch)

            loss_mean = loss
            if accelerator:
                accelerator.backward(loss_mean)
            else:
                loss_mean.backward()
            optimizer.step()

            cur_lr = optimizer.param_groups[-1]['lr']
            show_loss += loss_mean.detach().cpu().numpy()

            # print statistics
            if accelerator:
                print_condition = (idx % print_every == print_every - 1) and (accelerator.is_main_process)
            else:
                print_condition = (idx % print_every == print_every - 1)

            if print_condition:
                cost_time = time.time() - time_now
                time_now = time.time()
                log_train = f'lr: {cur_lr:.6f} | step: {idx + 1}/{len(train_dataloader) + 1} | time cost {cost_time:.2f}s | '
                log_loss = f'loss: {(show_loss / print_every):.4f}'
                log.info(log_train + log_loss)
                show_loss = 0

            lr_scheduler.step()

        if accelerator.is_main_process:
            log.info(f'Current lr: {cur_lr}')
        if (epoch % eval_every) == (eval_every - 1) and epoch >= 0:
            if accelerator.is_main_process:
                log.info('Evaluating Net...')
            res = eval_net(epoch, model, eval_dataloader, mode='eval')
            
            if best_score <= res:
                best_score = res
                best_epoch = epoch

            # if accelerator.is_main_process:
            #     log.info(f"Cur epoch: {epoch} | Eval_Score: {res * 100:.3f}")

            # save model
            if accelerator.is_main_process:
                log.info('Saving Model...')
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(),
                             os.path.join(log_path, 'epoch' + str(epoch).zfill(3) + '.pth'))

    del model
    time.sleep(3)

    torch.cuda.empty_cache()
    time.sleep(3)

    return best_epoch


def run_test(best_epoch, test_dataset, log_path, cfg, tkr):
    model = MyModel(cfg, tkr)

    test_dataloader = DataLoader(test_dataset, batch_size=cfg.eval.batch_size, num_workers=16, shuffle=False,
                                 collate_fn=partial(test_dataset.collate, tkr=tkr, max_len=cfg.dataset.max_input_len))

    test_loader, model = accelerator.prepare(test_dataloader, model)

    model_pth_path = os.path.join(log_path, 'epoch' + str(best_epoch).zfill(3) + '.pth')
    model = accelerator.unwrap_model(model)
    model.load_state_dict(torch.load(model_pth_path, map_location='cpu'))
    # model = load_model(model, log_path, best_epoch, accelerator)
    test_score = eval_net(best_epoch, model, test_loader, mode='test')
    if accelerator.is_main_process:
        log.info(f"Best Epoch: {best_epoch} | Test Score: {test_score}")



@hydra.main(config_path="Config", config_name="basic_cfg", version_base='1.1')
def main(cfg: DictConfig):
    # On Server
    #     accelerator = Accelerator()
    global accelerator

    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp])
    device = accelerator.device

    # w/o accelerator
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # accelerator=None

    # On Local Mac
    # device = torch.device("mps")
    # accelerator = None

    setup_seed(int(cfg.train.seed))

    log_path = make_exp_dirs(cfg.name)
    if accelerator.is_main_process:
        global log
        log = setting_logger(log_path)

    tkr = BertTokenizer.from_pretrained(cfg.model.model_name_or_path)

    train_dataset = make_dataloader('train', cfg)
    eval_dataset = make_dataloader('dev', cfg)
    test_dataset = make_dataloader('test', cfg)

    # log model file into log
    # log.info(inspect.getsource(model_file))

    # log.info(model)

    str_cfg = OmegaConf.to_yaml(cfg)

    if accelerator.is_main_process:
        log.info(f'Found device: {device}')

        log.info(f"Config: {str_cfg}")

        log.info(f"train data: {len(train_dataset)}")
        log.info(f"dev data: {len(eval_dataset)}")
        log.info(f"test data: {len(test_dataset)}")

    # writr_gt(test_dataloader, log_path, tkr=tkr)

    if cfg.debug:
        cfg.train.max_epoch = 1

    if accelerator.is_main_process:
        log.info(f"Training...")
    best_epoch = run_train(cfg=cfg, tkr=tkr,
                           train_dataset=train_dataset, eval_dataset=eval_dataset,
                           log_path=log_path)

    if accelerator.is_main_process:
        log.info(f"Testing...")
    run_test(best_epoch, test_dataset, log_path, cfg, tkr)

    OmegaConf.save(cfg, os.path.join(log_path, "config.yaml"))

    # clear saved models
    dir_name = os.path.join(log_path, f'epoch*')
    cmd = 'rm -rf ' + dir_name
    # print(cmd)
    # os.system(cmd)
    

if __name__ == '__main__':
    main()