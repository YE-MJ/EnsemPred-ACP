import sys
import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel, logging
import torch
from save.model_test.catboost_model import catboost_ensemble
from save.model_test.GB_model import gb_ensemble
from save.model_test.RF_model import RF_ensemble
from save.model_test.xgboost_model import xgboost_ensemble
from save.model_test.catboost_model_embedding import catboost_ensemble_enbedding
from save.model_test.SVM_model_embedding import SVM_ensemble_enbedding
from Prott5.test.LSTM_ensemble import ensemble_prott5_LSTM
from ESM2.test.LSTM_ensemble import ensemble_esm_LSTM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def AAC(name,seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    header = ['SampleName'] + [f'AAC_{aa}' for aa in AA]
    
    count = Counter(seq)
    code = [name] + [count.get(aa, 0) / len(seq) for aa in AA]
    encodings_df = pd.DataFrame([code], columns=header)
    
    return encodings_df

def DPC(name, seq, normalized=False):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    diPeptides = ['DPC_' + aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['SampleName'] + diPeptides
    tmpCode = [0] * 400
    AADict = {AA[i]: i for i in range(len(AA))}
    
    for j in range(len(seq) - 1):
        index = AADict[seq[j]] * 20 + AADict[seq[j + 1]]
        tmpCode[index] += 1
    
    if normalized and sum(tmpCode) != 0:
        tmpCode = [i / sum(tmpCode) for i in tmpCode]
    
    code = [name] + tmpCode
    encodings_df = pd.DataFrame([code], columns=header)
    
    return encodings_df

def CKSAAP(name, seq, gap=3, normalized=False):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    
    aaPairs = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    
    header = ['SampleName']
    for g in range(gap + 1):
        header += [f'CKSAAP_{pair}.gap{g}' for pair in aaPairs]
    
    code = [name]

    for g in range(gap + 1):
        myDict = {pair: 0 for pair in aaPairs}
        total_pairs = 0  

        for index1 in range(len(seq) - g - 1):  
            index2 = index1 + g + 1
            if seq[index1] in AA and seq[index2] in AA:
                pair = seq[index1] + seq[index2]
                myDict[pair] += 1
                total_pairs += 1

        if total_pairs > 0:
            if normalized:
                code += [myDict[pair] / total_pairs for pair in aaPairs]  
            else:
                code += [myDict[pair] for pair in aaPairs] 
        else:
            code += [0 for _ in aaPairs]

    encodings_df = pd.DataFrame([code], columns=header)
    
    return encodings_df

def embedding_T5(name, seq,T5_tokenizer, T5_model):
    embeddings = []
    inputs = T5_tokenizer(seq, return_tensors='pt')
    # 모델에 입력하여 임베딩 추출
    with torch.no_grad():
        outputs = T5_model(**inputs)

    # 임베딩 벡터 추출
    embedding = outputs.last_hidden_state

    # 시퀀스 레벨 임베딩을 위해 평균을 취합니다.
    sequence_embedding = torch.mean(embedding, dim=1).squeeze().numpy()

    # 라벨과 임베딩을 리스트에 저장
    embeddings.append([name] + sequence_embedding.tolist())
    
    df_T5 = pd.DataFrame(embeddings)
    df_T5.columns = ['SampleName'] + [f'embedding_{i}' for i in range(1, df_T5.shape[1])]
    
    return df_T5

def embedding_esm2(name, seq, ems2_tokenizer, esm2_model):
    embeddings = []
    inputs = ems2_tokenizer(seq, return_tensors='pt')
    
    with torch.no_grad():
        outputs = esm2_model(**inputs)

    # 임베딩 벡터 추출
    embedding = outputs.last_hidden_state

    # 시퀀스 레벨 임베딩을 위해 평균을 취합니다.
    sequence_embedding = torch.mean(embedding, dim=1).squeeze().numpy()

    # 라벨과 임베딩을 리스트에 저장
    embeddings.append([name] + sequence_embedding.tolist())
    
    df_ems2 = pd.DataFrame(embeddings)
    df_ems2.columns = ['SampleName'] + [f'embedding_{i}' for i in range(1, df_ems2.shape[1])]
    
    return df_ems2

def Result(File, T5_tokenizer, T5_model, ems2_tokenizer, esm2_model, labels):
    Output = {}
    F1  = {}
    AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    seq_list = [] 
    accession_list = [] 
    
    AAC_df_tmp = []
    DPC_df_tmp = []
    CKSAAP_df_tmp = []
    T5_df_tmp = []
    esm2_df_tmp = []
    
    for accession, seq in File.items():
        seq = "".join(seq)
        
        seq_list.append(seq)
        accession_list.append(accession)
        
        for val in seq:
            if val not in AA:
                error_message = f"Sequence contains non-natural amino acids, please provide sequence contains only natural amino acids\nID: {accession}\nSequence: {seq}"
                sys.stderr.write(error_message)
                sys.exit(1)

        AAC_tmp = AAC(accession, seq)
        AAC_df_tmp.append(AAC_tmp)
        
        DPC_tmp = DPC(accession, seq)
        DPC_df_tmp.append(DPC_tmp)
        
        CKSAAP_tmp = CKSAAP(accession, seq)
        CKSAAP_df_tmp.append(CKSAAP_tmp)
        
        T5_tmp = embedding_T5(accession, seq, T5_tokenizer, T5_model)
        T5_df_tmp.append(T5_tmp)
        
        esm2_tmp = embedding_esm2(accession, seq, ems2_tokenizer, esm2_model)
        esm2_df_tmp.append(esm2_tmp)
        
    AAC_df, AAC_name = pd.concat(AAC_df_tmp, axis=0, ignore_index=True), 'AAC'
    DPC_df, DPC_name  = pd.concat(DPC_df_tmp, axis=0, ignore_index=True), 'DPC'
    CKSAAP_df, CKSAAP_name  = pd.concat(CKSAAP_df_tmp, axis=0, ignore_index=True), 'CKSAAP'
    
    CKSAAP_AAC_df, CKSAAP_AAC_name  = pd.merge(CKSAAP_df, AAC_df, on='SampleName'), 'CKSAAP_AAC'
    CKSAAP_DPC_df, CKSAAP_DPC_name  = pd.merge(CKSAAP_df, DPC_df, on='SampleName'), 'CKSAAP_DPC'
    
    DPC_AAC_df, DPC_AAC_name  = pd.merge(DPC_df, AAC_df, on='SampleName'), 'DPC_AAC'
    DPC_CKSAAP_AAC_df, DPC_CKSAAP_AAC_name = pd.merge(DPC_df, CKSAAP_AAC_df, on='SampleName'), 'DPC_CKSAAP_AAC'
    
    T5_df, T5_name = pd.concat(T5_df_tmp, axis=0, ignore_index=True), 'prott5'
    esm2_df, ems2_name = pd.concat(esm2_df_tmp, axis=0, ignore_index=True), 'esm2'
    
    catboost_proba = catboost_ensemble(AAC_df, DPC_df, CKSAAP_df, CKSAAP_AAC_df, CKSAAP_DPC_df, DPC_AAC_df, DPC_CKSAAP_AAC_df,
                      AAC_name, DPC_name, CKSAAP_name, CKSAAP_AAC_name, CKSAAP_DPC_name, DPC_AAC_name, DPC_CKSAAP_AAC_name)
    
    gb_proba = gb_ensemble(AAC_df, DPC_df, CKSAAP_df, CKSAAP_AAC_df, CKSAAP_DPC_df, DPC_AAC_df, DPC_CKSAAP_AAC_df,
                      AAC_name, DPC_name, CKSAAP_name, CKSAAP_AAC_name, CKSAAP_DPC_name, DPC_AAC_name, DPC_CKSAAP_AAC_name)
    
    RF_proba = RF_ensemble(AAC_df, DPC_df, CKSAAP_df, CKSAAP_AAC_df, CKSAAP_DPC_df, DPC_AAC_df, DPC_CKSAAP_AAC_df,
                      AAC_name, DPC_name, CKSAAP_name, CKSAAP_AAC_name, CKSAAP_DPC_name, DPC_AAC_name, DPC_CKSAAP_AAC_name)
    
    xgboost_proba = xgboost_ensemble(AAC_df, DPC_df, CKSAAP_df, CKSAAP_AAC_df, CKSAAP_DPC_df, DPC_AAC_df, DPC_CKSAAP_AAC_df,
                      AAC_name, DPC_name, CKSAAP_name, CKSAAP_AAC_name, CKSAAP_DPC_name, DPC_AAC_name, DPC_CKSAAP_AAC_name)
    
    catboost_embedding_proba = catboost_ensemble_enbedding(T5_df, esm2_df, T5_name, ems2_name)
    SVM_embedding_proba = SVM_ensemble_enbedding(T5_df, esm2_df, T5_name, ems2_name)
    
    LSTM_prott5_proba = ensemble_prott5_LSTM(T5_df, seq_list)
    LSTM_ems2_proba = ensemble_esm_LSTM(esm2_df, seq_list)
    
    # 앙상블 확률의 평균 계산
    ensemble_proba = np.mean([catboost_proba, gb_proba, RF_proba, xgboost_proba, catboost_embedding_proba, SVM_embedding_proba, LSTM_prott5_proba, LSTM_ems2_proba], axis=0)

    # 0.5를 기준으로 이진 분류 수행
    predictions = [1 if proba > 0.5 else 0 for proba in ensemble_proba]
    
    # 성능 지표 계산
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    # AUC-ROC 값 계산
    auc_roc = roc_auc_score(labels, ensemble_proba)

    # 성능 지표 반환
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc
    }
    
    return metrics  # 성능 지표 반환

def load_fasta_file(filepath):
    File = {}
    labels = []  # Label 리스트를 추가합니다.
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    ID = None
    Count = 0
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            Count += 1
            ID = "{0}_{1}".format((line.replace(">", "")).replace("_", "-"), Count)
            File[ID] = []
            
            # line에 'positive'가 있는지 확인하여 label을 추가합니다.
            if "positive" in line.lower():
                labels.append(1)
            else:
                labels.append(0)
        elif ID:
            File[ID].append(line)
    
    return File, labels  # File과 labels를 함께 반환합니다.


def main():
    filepath = './data/independent_ACP_data.txt'
    
    T5_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    T5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    
    ems2_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    esm2_model = AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
    
    File, labels = load_fasta_file(filepath)
    
    # 결과와 성능 지표를 계산
    metrics = Result(File, T5_tokenizer, T5_model, ems2_tokenizer, esm2_model, labels)
    
    # 성능 지표 출력
    print("Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")

if __name__ == '__main__':
    main()
