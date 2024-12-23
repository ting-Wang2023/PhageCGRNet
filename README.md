# PHPCGR
PHPCGR is a novel method for predicting the classification of phage hosts, which integrates k-mer frequency information with Chaos Game Representation positional information and employs a Convolutional Neural Network model for classification prediction.
# Usage
python phpcgr.py --phage_file phage_file_name.fasta --host_file host_file_name.txt --savefolder best_model.pth --k 7
# Parameters
--phage_file sequences file of phages (input file, fasta file format only)
--host_file  file of hosts' category (input file, txt file format only)
--savefolder save the best model, default='./output/ '
--k  length of k-mer

# Example
python phpcgr.py --phage_file example_phage.fasta --host_file expample_host.txt --savefolder best_model.pth --k 7
# Citation
Ting Wang, Zu-Guo Yu, Jinyan Li, PHPCGR: A Novel Method for Phage Host Classification Prediction based on Chaos Game Representation and Convolutional Neural Network
