# from mauveexps import compute_divergence,compute_divergence_new
# import mauve
import numpy as np
import mauveexps as mauve
import matplotlib.pyplot as plt
from sklearn.metrics import auc as compute_area_under_curve
from numpy.linalg import inv
import mauve
import random
from universal_estimater import estimate

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

def get_divergence_value(p_feat_sampled,q_feat_sampled,num_buckets):
    out = mauve.compute_mauve(p_features=p_feat_sampled, q_features=q_feat_sampled, num_buckets=num_buckets,
                              verbose=False, mauve_scaling_factor=1.0)
    divergence_curve = out.divergence_curve
    # # print(divergence_curve)
    divergenceMauve = np.max([-np.log(divergence_curve[1, 1]), -np.log(divergence_curve[-2, 0])])
    return divergenceMauve

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def make_semidef_mat(dimensions_inp,low=0,high=1):
    # A = np.random.rand(dimensions_inp, dimensions_inp)
    A = np.diag([getrandomNumber(low,high) for _ in range(dimensions_inp)])
    # B = np.dot(A, A.T)
    if check_symmetric(A) and np.linalg.det(A) !=0:
        return A
    else:
        return make_semidef_mat(dimensions_inp,low,high)

def getrandomNumber(low,high):
    return round(random.uniform(low, high),2)


def check_array(retArray):
    numRows = retArray.shape[0]
    trueList = []
    for idx in range(numRows):
        query = retArray[idx,:]
        for jdx in range(numRows):
            keyArray = retArray[jdx,:]
            if idx != jdx:
                if all(query == keyArray):
                    trueList.append(all(query == keyArray))
    return trueList

def getMultimodalSamples(dictOfMeanAndVariance,numberSamples):
    """

    :param dictOfMeanAndVariance: each key have the value of mean and variance matrix. number of keys tells number
    peaks
    :return: array of samples from the multimodal distribution.
    """
    retList = []
    numGaussians = len(dictOfMeanAndVariance)-1
    for i in range(int(numberSamples/numGaussians)):
        for jdx in range(numGaussians):
            numSamples = len(list(dictOfMeanAndVariance[jdx]))
            choosenValue = random.randint(0,numSamples-1)
            mean = dictOfMeanAndVariance[jdx][choosenValue]['mean']
            cov = dictOfMeanAndVariance[jdx][choosenValue]['cov']
            normalValue = np.random.multivariate_normal(mean, cov, 1)
            retList.append(normalValue.squeeze())
    retListArray = np.asarray(retList)
    # check_array(retListArray)

    return retListArray

if __name__ == '__main__':

    dimensions_inp = 100
    X = []
    Y_mauve = []
    Y_gt = []
    Y_div_new = []
    mean_p = [getrandomNumber(0, 1) for i in range(dimensions_inp)]
    std_p = make_semidef_mat(dimensions_inp, 0, 1)
    mean_q = [getrandomNumber(0, 2) for i in range(dimensions_inp)]
    std_q = make_semidef_mat(dimensions_inp, 0, 1)
    sample_size = 6000
    maxNumCluster = 600

    mean_p2 = [getrandomNumber(0, 4) for i in range(dimensions_inp)]
    std_p2 = make_semidef_mat(dimensions_inp, 0, 1)
    mean_q2 = [getrandomNumber(0, 6) for i in range(dimensions_inp)]
    std_q2 = make_semidef_mat(dimensions_inp, 0, 1)

    numClusters = [2] + list(range(10,3*(maxNumCluster+1),50))

    dictOfMeanAndVariance = [{0:{"mean":mean_p,"cov":std_p},1:{"mean":mean_q,"cov":std_q}},
                             {0:{"mean":mean_p2,"cov":std_p2},1:{"mean":mean_q2,"cov":std_q2}}]

    for numCluster in numClusters:

        mauve_ori  = []
        div_new  = []
        actualDivs_sampled = []

        for bootstap_idx in range(5):

            # p_feat_sampled = np.random.multivariate_normal(mean_p, std_p, sample_size)
            # q_feat_sampled = np.random.multivariate_normal(mean_p, std_p, sample_size)
            p_feat_sampled = getMultimodalSamples(dictOfMeanAndVariance, sample_size)
            q_feat_sampled = getMultimodalSamples(dictOfMeanAndVariance, sample_size)
            # p_feat_sampled = np.random.normal(1, 1, (sample_size,dimensions_inp))
            # q_feat_sampled = np.random.normal(5, 2, (sample_size,dimensions_inp))
            mean_sampled_p = np.mean(p_feat_sampled, axis=0)
            mean_sampled_q = np.mean(q_feat_sampled, axis=0)

            std_sampled_p = np.cov(p_feat_sampled.T)
            std_sampled_q = np.cov(q_feat_sampled.T)
            if len(std_sampled_p.shape) == 0:
                std_sampled_p = std_sampled_p.reshape(1, 1)
                std_sampled_q = std_sampled_q.reshape(1, 1)

            actualDive = gaussian_divergence_ndims(mean_sampled_p, mean_sampled_q, std_sampled_p, std_sampled_q)
            # actualDive = 0
            divergence = estimate(p_feat_sampled, q_feat_sampled,n_jobs=4)
            divergenve_out = get_divergence_value(p_feat_sampled,q_feat_sampled,numCluster)
            mauve_ori.append(divergenve_out)
            actualDivs_sampled.append(actualDive)
            div_new.append(divergence)
        X.append(numCluster)
        Y_gt.append([np.mean(actualDivs_sampled),np.std(actualDivs_sampled)])
        Y_mauve.append([np.mean(mauve_ori),np.std(mauve_ori)])
        Y_div_new.append([np.mean(div_new),np.std(div_new)])

    Y_mauve = np.asarray(Y_mauve)
    Y_gt = np.asarray(Y_gt)
    Y_div_new = np.asarray(Y_div_new)
    np.save("mauve_exps_{}.npy".format(dimensions_inp),
            Y_mauve)
    np.save("div_exps_{}.npy".format(dimensions_inp),
            Y_div_new)
    np.save("gt_exps_{}.npy".format(dimensions_inp),
            Y_gt)
    # # Y_mauve_new2 = np.asarray(Y_mauve_new2)
    # # plt.xlabel("Number clusters")
    # # plt.ylabel("Divergence")
    # # # plt.errorbar(X, Y_gt[:, 0], Y_gt[:, 1], label="gt")
    # # plt.errorbar(X, Y_mauve[:, 0], Y_mauve[:, 1], label="mauve_divergence")
    # # # plt.errorbar(X, Y_mauve_new[:, 0], Y_mauve_new[:, 1], label="50%")
    # # # plt.errorbar(X, Y_mauve_new2[:, 0], Y_mauve_new2[:, 1], label="10%")
    # #
    # plt.rcParams.update({'font.size': 15})
    # fig, ax = plt.subplots()
    # # plt.title('Example of Two Y labels')
    #
    # # using the twinx() for creating another
    # # axes object for secondary y-Axis
    # ax2 = ax.twinx()
    # # ax.plot(dataArraySort[:,0],dataArraySort[:,1], color="g",label="Divergence")
    # # ax.errorbar(X, Y_div_arr[:, 0], Y_div_arr[:, 1], label="div_new")
    # ax.plot(X, Y_mauve[:, 0], label="mauve",color="g")
    #
    # # ax.plot(X, Y_mv_arr[:,0],label = "div_mauve")
    # # ax.errorbar(X, Y_mv_arr[:,0], Y_mv_arr[:,1],label = "div_mauve")
    # ax2.plot(X, Y_gt[:, 0], label="gt_div",color="b")
    # # ax2.errorbar(X, Y_gt_arr[:, 0], Y_gt_arr[:, 1], label="gt_div")
    # # # ax2.plot(attackAccuracy[:,0],attackAccuracy[:,1],color="r",label="Accuracy")
    # #
    # # # giving labels to the axises
    # ax.set_xlabel('number of cluster')
    # ax.set_ylabel('Divergence mauve')
    # #
    # # # secondary y-axis label
    # ax2.set_ylabel('Divergence GT')
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines + lines2, labels + labels2, loc=0)
    # # plt.legend()
    # plt.show()

    pass
