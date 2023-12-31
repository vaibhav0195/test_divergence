from divergence import skl_efficient
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import os
from npeet import entropy_estimators as ee
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import math

def getPqFromNpyData(npyFilePath):
    sentOutPair = np.load(npyFilePath, allow_pickle=True)
    p = []
    q = []
    for sentIp, gtSentiment, predictedSentiment, actualSent in sentOutPair:
        if float(gtSentiment) == 0:
            p.append(sentIp)
        else:
            q.append(sentIp)
    return p, q


def getcumsumVariance(p,q,norm='l2', whiten=False,
                         pca_max_data=-1,
                         explained_variance=0.95,seed=25):
    data1 = np.vstack([q, p])
    if norm in ['l2', 'l1']:
        data1 = normalize(data1, norm=norm, axis=1)
    pca = PCA(n_components=None, whiten=whiten, random_state=seed + 1)
    pca.fit(data1)
    s = np.cumsum(pca.explained_variance_ratio_)
    varData = np.var(data1)
    varData = np.sum(np.var(data1, axis=1))
    return varData

if __name__ == '__main__':
    dirNames = ["smartMaskingGenderDataset_new", "smartMaskingPoliticalDataset", "smart_masking_redit_suicide",
                "smartMaskingValidationMedal"]
    for dirName in dirNames:

        outDir = "/home/vaibhav/ML/bartexps/{}".format(dirName)
        dataToPlot = []
        sentenceTransformerModel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # for numNeighbou in [0, 10, 50, 100]:
        # dataToPlot = []
        # for numNeighbou in [0, 10, 50, 100,200]:
        for maskPerc in os.listdir(outDir):
            if '.' in maskPerc:
                continue
            numWordsToMask = float(maskPerc)
            npyDataFilePath = outDir + "/{}/sentOut.npy".format(maskPerc)
            p_text, q_text = getPqFromNpyData(npyDataFilePath)
            p_feat = sentenceTransformerModel.encode(p_text)
            q_feat = sentenceTransformerModel.encode(p_text)
            # print(type(q_feat)) kldiv
            num_clusters = min([p_feat.shape[0],q_feat.shape[0]])
            variance = getcumsumVariance(p_feat,q_feat,norm='l2')
            # divergence = ee.kldiv(p_feat, q_feat, k=int(num_clusters/10),base=math.e)
            # divergence = skl_efficient(p_feat, q_feat, k=int(5))
            # divergence_q = skl_efficient(q_feat, p_feat, k=int(num_clusters/10))
            # divergence = np.max([divergence_p,divergence_q])
            print("maskperc {} divergence {}".format(numWordsToMask,variance))
            dataToPlot.append([numWordsToMask,variance])
        dataToPlot = np.asarray(dataToPlot)
        os.makedirs("new_divergence_var".format(dirName), exist_ok=True)
        np.save("new_divergence_var/{}_new_divergence.npy".format(dirName, dirName),
                dataToPlot)
    pass
