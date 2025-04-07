import numpy as np
import CGR_3D
import copy
import time
import pandas as pd
import torch.utils.data as Data
import torch
from torch import nn 
from model import cnn
import argparse


def cgr():

    return CGR_3D.cgr_3d(args.fasta_file, args.k)


def load_data(host_file, rank):
    # process label
    host_list = []
    f_in = open(host_file)
    f_out = open("%s_label.txt" % rank, "w")
    count = 0
    for line in f_in:
        host = line.strip("\n")
        if host not in host_list:
            host_list.append(host)
            f_out.write(str(count) + "\t" + host + "\n")
            count += 1
    f_in.close()
    f_out.close()

    f_in = open(host_file)
    label_list = []
    for line in f_in:
        host = line.strip("\n")
        label_list.append(host_list.index(host))
    f_in.close()
    y = np.array(label_list)
    x = cgr()
    lst = list(zip(x, y))


    
    train_data, val_data, test_data = Data.random_split(lst, [round(0.8 * len(lst)), round(0.1 * len(lst)), len(lst) - round(0.8 * len(lst)) - round(0.1 * len(lst))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,  
                                       shuffle=True,  
                                       num_workers=2)  
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,  
                                     shuffle=True,  
                                     num_workers=2)  
    
    test_dataloader = Data.DataLoader(dataset= test_data,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader  

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    
   
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()
    


    for epoch in range(num_epochs):
        print("Epoch{}/{}".format(epoch,num_epochs-1))
        print("-"*10)

        
        train_loss = 0.0 
        train_corrects = 0 
        val_loss = 0.0 
        val_corrects = 0 
        train_num = 0 
        val_num = 0

       
        for step, (b_x, b_y) in enumerate(train_dataloader):
            
            b_x = b_x.to(device)
            b_y = b_y.to(device) 
            model.train() 

            output = model(b_x)
            pre_lab = torch.argmax(output,dim=1)
            loss = criterion(output, b_y.long()) 

            optimizer.zero_grad() 
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item() * b_x.size(0) 
            train_corrects += torch.sum(pre_lab == b_y.data) 
            train_num += b_x.size(0) 

        
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)  
            b_y = b_y.to(device)  
            model.eval()  
            output = model(b_x)  
            pre_lab = torch.argmax(output, dim=1)  
            loss = criterion(output, b_y.long())  
            
            val_loss += loss.item() * b_x.size(0)  
            val_corrects += torch.sum(pre_lab == b_y.data)  
            val_num += b_x.size(0)  

        
        train_loss_all.append(train_loss/train_num) 
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        
        
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        
        time_use = time.time() - since
        print("The time consumed by training and validation: {:.0f}m{:.0f}s".format(time_use//60,time_use%60))

    

    torch.save(best_model_wts, args.savefolder)

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                        "train_loss_all": train_loss_all,
                                        "val_loss_all": val_loss_all,
                                        "train_acc_all": train_acc_all,
                                        "val_acc_all": val_acc_all})
    return train_process


def test_model_process(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model = model.to(device) 

    
    test_corrects = 0  
    test_num = 0

    
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)  
            test_data_y = test_data_y.to(device)  
            model.eval()  
            output = model(test_data_x)
            
            pre_lab = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0) 

    test_acc = test_corrects.double().item()/test_num 
    print('Accuracy：', test_acc)


def get_args():

    """Get args"""
    parser = argparse.ArgumentParser(
        description="PHPCGR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--fasta_file', default='./input/Phapes.fasta' )
    parser.add_argument('--host_file', default='./input/Hosts.txt')

    parser.add_argument('--savefolder', default='./output/best_model.pth', type=str,help='Save the optimal parameters')



    parser.add_argument('--k', type=int, help='the length of kmer')


    args = parser.parse_args()
    return args

def CGR_3D_main(args):
    torch.manual_seed(95)
    fasta_file = args.fasta_file
    
    host_file = args.host_file

    CGR_CNN = cnn()
    train_dataloader, val_dataloader, test_dataloader = load_data(host_file, 'specie')

    train_model_process(CGR_CNN, train_dataloader, val_dataloader, 40)

    CGR_CNN.load_state_dict(torch.load(args.savefolder))
    test_model_process(CGR_CNN, test_dataloader)

    pass

if __name__ == "__main__":
    args = get_args()
    CGR_3D_main(args)

