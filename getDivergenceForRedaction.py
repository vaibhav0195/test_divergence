from divergence import skl_efficient
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import os
from npeet import entropy_estimators as ee
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import math
import mauve

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


def normaliseData(p,q,norm='l2', whiten=False,
                         pca_max_data=-1,
                         explained_variance=0.9,seed=25):
    data1 = np.vstack([q, p])
    if norm in ['l2', 'l1']:
        data1 = normalize(data1, norm=norm, axis=1)
    # varData = np.sum(np.var(data1,axis=0))
    # if varData > 0.5:
    pca = PCA(n_components=None, whiten=whiten, random_state=seed + 1)
    pca.fit(data1)
    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= explained_variance)  # last index to consider
    data1 = pca.transform(data1)[:, :idx + 1]
    p_data = data1[q.shape[0]: , :]
    q_data = data1[:q.shape[0], :]
    return p_data,q_data

def getRandomWithReplacement(inp_feat):
    inp_ret = []
    numSamplesretinp = [i for i in range(inp_feat.shape[0])]
    for idx_inp in numSamplesretinp:
        inp_ret.append(inp_feat[idx_inp])
    inp_ret = np.asarray(inp_ret)
    return inp_ret

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
            p_feat_orignal = sentenceTransformerModel.encode(p_text)
            q_feat_orignal = sentenceTransformerModel.encode(q_text)
            # print(type(q_feat)) kldiv
            num_clusters = min([p_feat_orignal.shape[0],q_feat_orignal.shape[0]])
            divergences = []
            mauveDivergences = []
            for bootstap_idx in range(10):
                p_feat_sampled = getRandomWithReplacement(p_feat_orignal)
                q_feat_sampled = getRandomWithReplacement(q_feat_orignal)
                p_feat,q_feat = normaliseData(p_feat_sampled,q_feat_sampled,norm='l2')
                # divergence = ee.kldiv(p_feat, q_feat, k=int(num_clusters/10),base=math.e)
                # np.random.choice(colors, n)
                divergence = skl_efficient(p_feat, q_feat, k=int(5))
                out = mauve.compute_mauve(p_features=p_feat_sampled, q_features=q_feat_sampled,
                                          verbose=False, mauve_scaling_factor=1.0)
                divergenceMauve = np.max([-np.log(out.divergence_curve[1, 1]),-np.log(out.divergence_curve[-2, 0])])
                divergences.append(divergence)
                mauveDivergences.append(divergenceMauve)
            divergences = np.asarray(divergences)
            mauveDivergences = np.asarray(mauveDivergences)

            dataToPlot.append([numWordsToMask,np.mean(divergences),np.std(divergences),
                               np.mean(mauveDivergences),np.std(mauveDivergences)])
        dataToPlot = np.asarray(dataToPlot)
        os.makedirs("new_divergence_pca_mauve".format(dirName), exist_ok=True)
        np.save("new_divergence_pca_mauve/{}_new_divergence.npy".format(dirName, dirName),
                dataToPlot)
    pass
