from mauveexps import compute_divergence,compute_divergence_new
# import mauve
import numpy as np
import mauve

from sklearn.metrics import auc as compute_area_under_curve
import matplotlib.pyplot as plt
from universal_estimater import estimate
from numpy.linalg import inv

def gaussian_divergence(mu1: float, mu2: float, sig1: float, sig2: float):
    """Analytical result for KL-divergence of two Gaussians.

    D( N(mu1, sig1) | N(mu2,sig2) )
    Ref: http://allisons.org/ll/MML/KL/Normal/
    """

    mudiff = pow(mu1 - mu2, 2)
    sigdiff = sig1 * sig1 - sig2 * sig2
    lograt = np.log(sig2 / sig1)
    secondTerm = (mudiff + sigdiff) / (2 * sig2 * sig2)
    div = lograt + (mudiff + sig1**2) / (2 * sig2 * sig2) -0.5
    return div

def gaussian_divergence_ndims(mu1, mu2, sig1, sig2):
    """Analytical result for KL-divergence of two Gaussians.

    D( N(mu1, sig1) | N(mu2,sig2) )
    Ref: http://allisons.org/ll/MML/KL/Normal/
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    sig1 = np.asarray(sig1)
    sig2 = np.asarray(sig2)


    sig1Det = np.linalg.det(sig1)
    sig2Det = np.linalg.det(sig2)
    sig2Inv = inv(sig2)
    d = sig1.shape[0]
    trace = np.trace(np.matmul(sig2Inv,sig1))
    subtract = np.subtract(mu2,mu1)
    third_term = np.matmul(np.transpose(subtract),sig2Inv)
    third_term = np.matmul(third_term,subtract)
    first_term = np.log(sig2Det/sig1Det)
    divergenceRet = first_term-d+trace+third_term
    return divergenceRet *0.5

def divergenceActual(p,q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def getRandomWithReplacement(inp_feat):
    inp_ret = []
    inpIdxs = [i for i in range(inp_feat.shape[0])]
    numSamplesretinp = np.random.choice(inpIdxs, inp_feat.shape[0])
    for idx_inp in numSamplesretinp:
        inp_ret.append(inp_feat[idx_inp])
    inp_ret = np.asarray(inp_ret)
    return inp_ret

def getmauvescore(divergence_curve_new):
    x, y = divergence_curve_new.T
    idxs1 = np.argsort(x)
    idxs2 = np.argsort(y)
    mauve_score = 0.5 * (
            compute_area_under_curve(x[idxs1], y[idxs1]) +
            compute_area_under_curve(y[idxs2], x[idxs2])
    )
    return mauve_score

if __name__ == '__main__':

    dimensions_inp = 100
    X = []
    Y_mauve = []
    Y_mauve_new = []

    for sample_size in [50,100,200,500]:

        print(sample_size)
        # num_clusters = min([p_feat_orignal.shape[0],q_feat_orignal.shape[0]])
        mauve_ori  = []
        mauve_updated  = []
        # actualDivs_sampled = []
        for bootstap_idx in range(5):
            # p_feat_sampled = getRandomWithReplacement(p_feat_orignal)
            # q_feat_sampled = getRandomWithReplacement(q_feat_orignal)
            p_feat_sampled = np.random.multivariate_normal([0], [[1]], sample_size)
            q_feat_sampled = np.random.multivariate_normal([5], [[4]], sample_size)
            std_between_samples = abs(float(np.std(p_feat_sampled))- float(np.std(q_feat_sampled)))
            actualDive    = gaussian_divergence_ndims([float(np.mean(p_feat_sampled))], [float(np.mean(q_feat_sampled))],
                                             [[float(np.std(p_feat_sampled))]], [[float(np.std(q_feat_sampled))]])
            num_clusters = min([q_feat_sampled.shape[0], p_feat_sampled.shape[0]])
            # divergence = estimate(p_feat_sampled, q_feat_sampled,n_jobs=4)
            # divergence = skl_efficient(p_feat_sampled, q_feat_sampled, k=13)
            out = mauve.compute_mauve(p_features=p_feat_sampled, q_features=q_feat_sampled,
                                      verbose=False, mauve_scaling_factor=1.0,divergence_curve_discretization_size=2)
            # divergence_curve = out.divergence_curve
            divergence_curve_new = compute_divergence_new(p_feat_sampled,q_feat_sampled)
            mauve_score = np.max([divergence_curve_new[0][1], divergence_curve_new[1][0]])
            # mauve_score = getmauvescore(divergence_curve_new)
            mauve_ori.append(out.mauve)
            mauve_updated.append(mauve_score)

        mauve_ori = np.asarray(mauve_ori)
        mauve_updated = np.asarray(mauve_updated)
        Y_mauve.append([np.mean(mauve_ori),np.std(mauve_ori)])
        Y_mauve_new.append([np.mean(mauve_updated),np.std(mauve_updated)])
        X.append(sample_size)
    print(Y_mauve_new)
    plt.xlabel("Sample size")
    plt.ylabel("Divergence")
    Y_mauve = np.asarray(Y_mauve)
    Y_mauve_new = np.asarray(Y_mauve_new)
    # plt.errorbar(X,Y_mauve[:,0],Y_mauve[:,1],label="original_mauve")
    plt.errorbar(X,Y_mauve_new[:,0],Y_mauve_new[:,1],label="our mauve")
    plt.legend()
    plt.show()

    pass
