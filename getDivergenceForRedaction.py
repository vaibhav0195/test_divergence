from divergence import skl_efficient
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import os
from npeet import entropy_estimators as ee

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

if __name__ == '__main__':
    dirName = "smartMaskingValidationMedal"
    outDir = "/home/vaibhav/ML/bartexps/{}".format(dirName)
    dataToPlot = []
    sentenceTransformerModel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # for numNeighbou in [0, 10, 50, 100]:
    # dataToPlot = []
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
        # divergence = ee.kldiv(p_feat, q_feat, k=int(num_clusters/10))
        divergence = skl_efficient(p_feat, q_feat, k=int(num_clusters/10))
        dataToPlot.append([numWordsToMask,divergence])
    dataToPlot = np.asarray(dataToPlot)
    os.makedirs("new_divergence".format(dirName), exist_ok=True)
    np.save("new_divergence/{}_new_divergence.npy".format(dirName, dirName),
            dataToPlot)
    pass
