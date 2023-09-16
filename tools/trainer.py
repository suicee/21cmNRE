
import torch
import torch.nn as nn
import torch.optim as optim
from tools.DataLoader import SummaryDataset
from torch.utils.data import DataLoader,random_split
from torch.optim.lr_scheduler import StepLR
import logging
import numpy as np
import pickle


def train_classifier(model,data,params,train_params={'epochs':100,'batch_size':20,'learning_rate':1e-1,'validation_fraction':10,}):
    epochs=train_params['epochs']
    batch_size=train_params['batch_size']
    lr=train_params['learning_rate']
    val_fra=train_params['validation_fraction']

    epoch_eval=True

    DS=SummaryDataset(data,params,norm=False)


    n_val = int(len(DS) * val_fra/100)
    n_train = len(DS) - n_val
    train, val = random_split(DS, [n_train, n_val])

    train_data=DataLoader(train,batch_size=batch_size,shuffle=True)
    val_data=DataLoader(val,batch_size=batch_size,shuffle=False)
    # logger.info(model)

    lossF=nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.8)

    device = torch.device('cuda')
    model.to(device=device)


    best_eval=np.inf
    best_epoch=0
    for epoch in range(epochs): 

        trained=0
        epoch_loss=0
        correct_sample=0
        model.train()
        # logger.info(f'epoch{epoch}-----training')

        for batches in train_data:

            datas=(batches['data'])
            params=batches['param']

            #generate labels
            labels=torch.cat((torch.ones((datas.shape[0],1)),torch.zeros(datas.shape[0],1)),dim=0)

            datas=torch.cat(2*[datas],0)

            re_orderred_params=torch.flip(params,dims=(0,))
            params=torch.cat((params,re_orderred_params),dim=0)

            datas = datas.to(device=device, dtype=torch.float32)
            params = params.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            model.zero_grad()
            preds= torch.sigmoid(model(datas,params))

            correct_sample+=torch.sum(torch.tensor(preds>0.5,dtype=torch.float32,device=device)==labels).item()

            loss = lossF(preds,labels)
            loss.backward()
            epoch_loss+=loss.item()
            optimizer.step()
            trained+=batch_size

            # if(trained%1000==0):
            #     logger.info(f"{trained}//{n_train}")
        
        if epoch_eval:
            model.eval()
            eval_loss=0
            correct_eval_sample=0
            for val_batches in val_data:

                datas=val_batches['data']
                params=val_batches['param']

                labels=torch.cat((torch.ones((datas.shape[0],1)),torch.zeros(datas.shape[0],1)),dim=0)

                datas=torch.cat(2*[datas],0)

                re_orderred_params=torch.flip(params,dims=(0,))
                params=torch.cat((params,re_orderred_params),dim=0)

                datas = datas.to(device=device, dtype=torch.float32)
                params = params.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    preds= torch.sigmoid(model(datas,params))
                correct_eval_sample+=torch.sum(torch.tensor(preds>0.5,dtype=torch.float32,device=device)==labels).item()
                loss = lossF(preds,labels)    
                eval_loss+=loss.item()

            if eval_loss<best_eval:
                
                # torch.save(model.state_dict(),f'models/best_model.pt')
                best_eval=eval_loss
                best_epoch=epoch
                best_model=pickle.loads(pickle.dumps(model))
            # logger.info(f"************current best:epoch{best_epoch} best_eval={best_eval/len(val_data)}******************")
        
        scheduler.step()
        
    return best_model
        