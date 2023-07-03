from mauveexps import compute_divergence,compute_divergence_new
import numpy as np
import mauve
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

if __name__ == '__main__':

    dimensions_inp = 100
    plt.xlabel("Sample size")
    plt.ylabel("Divergence")
    X = []
    Y_div_new = []
    Y_mauve = []
    Y_div_gt = []
    for sample_size in [50,100,200,500]:

        print(sample_size)
        # num_clusters = min([p_feat_orignal.shape[0],q_feat_orignal.shape[0]])
        divergences_gauss  = []
        mauve_Divergences  = []
        actualDivs_sampled = []
        for bootstap_idx in range(5):
            # p_feat_sampled = getRandomWithReplacement(p_feat_orignal)
            # q_feat_sampled = getRandomWithReplacement(q_feat_orignal)
            mean_p = [0]
            std_p = [[1]]
            mean_q = [5]
            std_q = [[4]]
            p_feat_sampled = np.random.multivariate_normal(mean_p,std_p , sample_size)
            q_feat_sampled = np.random.multivariate_normal(mean_q, std_q, sample_size)
            # std_between_samples = abs(float(np.std(p_feat_sampled))- float(np.std(q_feat_sampled)))
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
            # divergence = skl_efficient(p_feat_sampled, q_feat_sampled, k=13)
            # out = mauve.compute_mauve(p_features=p_feat_sampled, q_features=q_feat_sampled,num_buckets=5,
            #                           verbose=False, mauve_scaling_factor=1.0)
            # divergence_curve = out.divergence_curve
            # # print(divergence_curve)
            # divergenceMauve = np.max([-np.log(divergence_curve[0, 1]), -np.log(divergence_curve[1, 0])])
            # divergence_curve = compute_divergence_new(p_feat_sampled,q_feat_sampled )
            # divegenceMauve = np.max([divergence_curve[0][1], divergence_curve[1][0]])
            # try:
            #     divergenceMauve = np.max([-np.log(divergence_curve[0, 1]), -np.log(divergence_curve[1, 0])])
            # except Exception as e:
            #     plt.plot(divergence_curve[:,0],divergence_curve[:,1])
            #     plt.show()
            print(divergence)
            divergences_gauss.append(divergence)
            mauve_Divergences.append(0)
            actualDivs_sampled.append(actualDive)

        actualDivs_sampled_array = np.asarray(actualDivs_sampled)
        divergences = np.asarray(divergences_gauss)
        mauveDivergences = np.asarray(mauve_Divergences)
        mauveDivMean = np.mean(mauveDivergences)
        mauveDivstd = np.std(mauveDivergences)
        divergences_mean = np.mean(divergences)
        divergence_std = np.std(divergences)

        X.append(sample_size)
        Y_div_gt.append([np.mean(actualDivs_sampled_array),np.std(actualDivs_sampled_array)])
        Y_div_new.append([divergences_mean,divergence_std])
        Y_mauve.append([mauveDivMean,mauveDivstd])
        # plt.plot(sample_size, divergences_mean,divergence_std, label=str("divergence new"))
        # plt.plot(sample_size, actualDive, label=str("original divergence"))
        # plt.plot(sample_size, mauveDivMean,mauveDivstd, label=str("divergence mauve"))
    Y_gt_arr = np.asarray(Y_div_gt)
    Y_div_arr = np.asarray(Y_div_new)
    Y_mv_arr = np.asarray(Y_mauve)
    # print(Y_div_arr[:,0], Y_div_arr[:,1])
    # print(Y_mv_arr[:,0], Y_mv_arr[:,1])
    print(Y_div_arr)
    plt.errorbar(X, Y_div_arr[:, 0], Y_div_arr[:, 1], label="div_new")
    plt.errorbar(X, Y_gt_arr[:, 0], Y_gt_arr[:, 1], label="gt_div")
    plt.errorbar(X, Y_mv_arr[:,0], Y_mv_arr[:,1],label = "div_mauve")


    plt.legend()
    plt.show()
    pass
