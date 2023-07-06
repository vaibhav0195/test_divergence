import random
import numpy as np
import mauveexps as mauve
from universal_estimater import estimate
from numpy.linalg import inv
import os

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


    _,sig1Det = np.linalg.slogdet(sig1)
    _,sig2Det = np.linalg.slogdet(sig2)
    # sig1Det = np.exp(sig1Det)+ 0.5**42
    # sig2Det = np.exp(sig2Det)+ 0.5**42
    sig2Inv = inv(sig2)
    d = sig1.shape[1]
    trace = np.trace(np.matmul(sig2Inv,sig1))
    subtract = mu2-mu1
    third_term = np.matmul(subtract.T,sig2Inv)
    third_term = np.matmul(third_term,subtract)
    first_term = sig2Det-sig1Det
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

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def make_semidef_mat(dimensions_inp,low=0,high=1):
    # A = np.random.rand(dimensions_inp, dimensions_inp)
    A = np.diag([getrandomNumber(low,high) for _ in range(dimensions_inp)])
    # B = np.dot(A, A.T)
    if check_symmetric(A):
        return A
    else:
        return make_semidef_mat(dimensions_inp,low,high)

def getrandomNumber(low,high):
    return round(random.uniform(low, high),2)

if __name__ == '__main__':

    dimensions_inp = 2
    X = []
    Y_div_new = []
    Y_mauve = []
    Y_div_gt = []
    mean_p = [getrandomNumber(0, 1) for i in range(dimensions_inp)]
    std_p = make_semidef_mat(dimensions_inp, 0, 1)
    mean_q = [getrandomNumber(0, 1) for i in range(dimensions_inp)]
    std_q = make_semidef_mat(dimensions_inp, 0, 1)

    for sample_size in [1000,2000,3000,5000]:

        print(sample_size)

        divergences_gauss  = []
        mauve_Divergences  = []
        actualDivs_sampled = []
        for bootstap_idx in range(5):
            p_feat_sampled = np.random.multivariate_normal(mean_p,std_p , sample_size)
            q_feat_sampled = np.random.multivariate_normal(mean_q, std_q, sample_size)
            mean_sampled_p = np.mean(p_feat_sampled,axis=0)
            mean_sampled_q = np.mean(q_feat_sampled,axis=0)

            std_sampled_p = np.cov(p_feat_sampled.T)
            std_sampled_q = np.cov(q_feat_sampled.T)
            if len(std_sampled_p.shape) == 0:
                std_sampled_p = std_sampled_p.reshape(1,1)
                std_sampled_q = std_sampled_q.reshape(1,1)
            actualDive    = gaussian_divergence_ndims(mean_sampled_p,mean_sampled_q,std_sampled_p,std_sampled_q)
            num_clusters = min([q_feat_sampled.shape[0], p_feat_sampled.shape[0]])
            divergence = estimate(p_feat_sampled, q_feat_sampled)
            out = mauve.compute_mauve(p_features=p_feat_sampled, q_features=q_feat_sampled,
                                      verbose=False, mauve_scaling_factor=1.0)
            divergence_curve = out.divergence_curve
            divergenceMauve = np.max([-np.log(divergence_curve[1, 1]), -np.log(divergence_curve[-2, 0])])
            divergences_gauss.append(divergence)
            mauve_Divergences.append(divergenceMauve)
            actualDivs_sampled.append(actualDive)

        actualDivs_sampled_array = np.asarray(actualDivs_sampled)
        divergences = np.asarray(divergences_gauss)
        mauveDivergences = np.asarray(mauve_Divergences)
        mauveDivMean = np.mean(mauveDivergences)
        mauveDivstd = np.std(mauveDivergences)
        divergences_mean = np.mean(divergences)
        divergence_std = np.std(divergences)

        X.append(sample_size)
        Y_div_gt.append([sample_size,np.mean(actualDivs_sampled_array),np.std(actualDivs_sampled_array)])
        Y_div_new.append([sample_size,divergences_mean,divergence_std])
        Y_mauve.append([sample_size,mauveDivMean,mauveDivstd])
    Y_gt_arr = np.asarray(Y_div_gt)
    Y_div_arr = np.asarray(Y_div_new)
    Y_mv_arr = np.asarray(Y_mauve)

    os.makedirs("divergence_exps/{}".format(dimensions_inp), exist_ok=True)
    np.save("divergence_exps/{}/mauve.npy".format(dimensions_inp),
            Y_mv_arr)
    np.save("divergence_exps/{}/div.npy".format(dimensions_inp),
            Y_div_arr)
    np.save("divergence_exps/{}/gt.npy".format(dimensions_inp),
            Y_gt_arr)
    # plt.errorbar(X, Y_div_arr[:, 0], label="div_new")
    # plt.errorbar(X, Y_gt_arr[:, 0], label="gt_div")
    # plt.errorbar(X, Y_mv_arr[:,0],label = "div_mauve")


    # plt.legend()
    # plt.show()
    pass
