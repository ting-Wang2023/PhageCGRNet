# PHPCGR
PHPCGR is a novel method for predicting the classification of phage hosts, which integrates k-mer frequency information with Chaos Game Representation positional information and employs a Convolutional Neural Network model for classification prediction.
# Dataset
We used the deephost and cherry datasets, which are available for download in references [1] and [2] 
# Usage
python phpcgr.py --phage_file phage_file_name.fasta --host_file host_file_name.txt --savefolder best_model.pth --k 7
# ParametersDataset
--phage_file sequences file of phages (input file, fasta file format only)  
--host_file  file of hosts' category (input file, txt file format only)  
--savefolder save the best model, default='./output/ '  
--k  length of k-mer  

# Example
Unzip file example_phage.7z to get file example_phage.fasta  
python phpcgr.py --phage_file example_phage.fasta --host_file expample_host.txt --savefolder best_model.pth --k 7
# Citation
Ting Wang, Zu-Guo Yu, Jinyan Li, PHPCGR: A Novel Method for Phage Host Classification Prediction based on Chaos Game Representation and Convolutional Neural Network
#References
[1]Ruohan W, Xianglilan Z, Jianping W, et al. DeepHost: phage host prediction with convolutional neural network. Briefings in Bioinformatics 2022; 23(1):bbab385.  
[2]Shang J, Sun Y. CHERRY: a Computational metHod for accuratE pRediction of virus–pRokarYotic interactions using a graph encoder–decoder model. Briefings in Bioinformatics 2022, 23(5): bbac182.
