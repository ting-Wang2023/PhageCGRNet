
import torch
import sys
from collections import defaultdict
import numpy as np
from functools import reduce
from Bio import SeqIO
from collections import Counter


# defining cgr graph
# CGR_CENTER = (0.5, 0.5)
CGR_X_MAX = 1
CGR_Y_MAX = 1
CGR_X_MIN = 0
CGR_Y_MIN = 0
CGR_A = (CGR_X_MIN, CGR_Y_MIN)
CGR_G = (CGR_X_MAX, CGR_Y_MIN)
CGR_T = (CGR_X_MAX, CGR_Y_MAX)
CGR_C = (CGR_X_MIN, CGR_Y_MAX)
CGR_CENTER = ((CGR_X_MAX - CGR_Y_MIN) / 2, (CGR_Y_MAX - CGR_Y_MIN) / 2)

# Add color code for each element


def empty_dict():
	"""
	None type return vessel for defaultdict
	:return:
	"""
	return None


CGR_DICT = defaultdict(
	empty_dict,
	[
		('A', CGR_A),  # Adenine
		('T', CGR_T),  # Thymine
		('G', CGR_G),  # Guanine
		('C', CGR_C),  # Cytosine
		('U', CGR_T),  # Uracil demethylated form of thymine
		('a', CGR_A),  # Adenine
		('t', CGR_T),  # Thymine
		('g', CGR_G),  # Guanine
		('c', CGR_C),  # Cytosine
		('u', CGR_T)  # Uracil/Thymine
		])


def fasta_reader(fasta):
	
	
	flist = SeqIO.parse(fasta, "fasta")
	for i in flist:
		yield i.description, i.seq
    

def get_all_kmer(seq,k):
    length = len(seq)
    return[seq[i:i+k]for i in range(length - k+1)]

def CGR_Frequency(seq,k,z,F):
    r = reduce(lambda x,y: [i+j for i in x for j in y], [['A','T','C','G']] * k)
    ck =Counter(get_all_kmer(seq, k))
    numk=(len(seq)-k+1)
    for key in ck.keys():
        p=ck[key]
        a=ck[key]/numk
          
    pk1=[]
    pk1.append(r)
    pk1.append(ck)
            
    lst = range(0,4**k);
            
    for i in lst:
        if pk1[0][i] not in pk1[1]:
            pk1[1][pk1[0][i]] = 0
              
    p={}
    p = pk1[1]      
    pks={}
    pks = dict(sorted(p.items(), key=lambda kv: (kv[0])))
    a = -1
    for value in pks.values():
        
            a=a+1
            
            F[z,a]=value
   
    
    return F

def mk_cgr(seq):

	
	cgr = []
	cgr_marker = CGR_CENTER[:
		]   
	for s in seq:
        
		cgr_corner = CGR_DICT[s]
		if cgr_corner:
			cgr_marker = (
				(cgr_corner[0] + cgr_marker[0]) / 2,
				(cgr_corner[1] + cgr_marker[1]) / 2
			)
			cgr.append([s, cgr_marker])
            
		else:
			sys.stderr.write("Bad Nucleotide: " + s + " \n")
        
	return cgr
    

def cgr_position(seq,cgr,k,z,A,B): 
    n=0
    pk={}
    pks={}
    r = reduce(lambda x,y: [i+j for i in x for j in y], [['A','T','C','G']] * k)
    for i in range(0,4**k):
        pks[r[i]]=[0,0]
    for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            n=n+1
            if kmer not in pk.keys():
                    pk[kmer] = []
            pk[kmer].append(cgr[n+k-2][1])
            
    for key,value in pk.items():
        p=np.sum([value[i][0] for i in range(len(value))])/len(value)
        q=np.sum([value[i][1] for i in range(len(value))])/len(value)
        pks[key]=[p,q]         
    pks=dict(sorted(pks.items(), key=lambda kv: (kv[0])))
    l=-1
   
    for value in pks.values():
        
            l=l+1
            A[z,l]=value[0] #CGR_x
            
            B[z,l]=value[1] #CGR_y
    
    return A,B


def cgr_3d(finput,k):
    
    lst = []
    seqs = [fa.seq for fa in SeqIO.parse(finput,"fasta")]
    
    A=np.zeros((len(seqs),4**k))
    B=np.zeros((len(seqs),4**k))
    F=np.zeros((len(seqs),4**k))
    
    z=0
    for seq in seqs:
        cgr=mk_cgr(seq)
        A,B=cgr_position(seq,cgr,k,z,A,B)
        F=CGR_Frequency(seq,k,z,F)
        z+=1
        
        
    #Normalisation
    A=(A-np.min(A))/(np.max(A)-np.min(A))
    B=(B-np.min(B))/(np.max(B)-np.min(B))
    F=(F-np.min(F))/(np.max(F)-np.min(F))

    for i in range(len(seqs)):
        S=[]
        S.append(A[i,:].reshape(2**k,2**k))
        S.append(B[i,:].reshape(2**k,2**k))
        S.append(F[i,:].reshape(2**k,2**k))
        lst.append(torch.tensor(S).float())
    
    return lst
