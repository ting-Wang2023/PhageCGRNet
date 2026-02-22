# PhageCGRNet
PhageCGRNet is a novel method for predicting the classification of phage hosts, which integrates k-mer frequency information with Chaos Game Representation positional information and employs a Convolutional Neural Network model for classification prediction.
# Dataset
We used the deephost and cherry datasets, which are available for download in references [1] and [2] 
# Environment Setup
1. Install conda
2. Change directory to the path of this project
bash
cd {your_PhageCGRNet_project_path}
3. Run following commands in the terminal
bash
conda create -n PhageCGRNet python=3.9.18
conda activate PhageCGRNet
pip install -r requirements.txt

# Usage
python phagecgrnet.py --phage_file phage_file_name.fasta --host_file host_file_name.txt --savefolder best_model.pth --k 7
# ParametersDataset
--phage_file sequences file of phages (input file, fasta file format only)  
--host_file  file of hosts' category (input file, txt file format only)  
--savefolder save the best model, default='./output/ '  
--k  length of k-mer  

# Example
Unzip file example_phage.7z to get file example_phage.fasta  
python phagecgrnet.py --phage_file example_phage.fasta --host_file expample_host.txt --savefolder best_model.pth --k 7
# Citation
Ting Wang, Zu-Guo Yu, Jinyan Li, Xuan Lin, PhageCGRNet: Integrating Chaos Game Representation of Genomes with Convolutional Neural Network for Accurate Phage Host Classification Prediction
# References
[1]Ruohan W, Xianglilan Z, Jianping W, et al. DeepHost: phage host prediction with convolutional neural network. Briefings in Bioinformatics 2022; 23(1):bbab385.  
[2]Shang J, Sun Y. CHERRY: a Computational metHod for accuratE pRediction of virus–pRokarYotic interactions using a graph encoder–decoder model. Briefings in Bioinformatics 2022, 23(5): bbac182.
