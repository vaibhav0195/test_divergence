import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    dim="100"
    gtPath = "gt_{}.npy".format(dim)
    divPath = "div_{}.npy".format(dim)
    mavPath = "mauve_{}.npy".format(dim)
    dataArray = np.load(gtPath,allow_pickle=True)
    divArray = np.load(divPath,allow_pickle=True)
    mauvArray = np.load(mavPath,allow_pickle=True)
    fig, ax1 = plt.subplots(figsize=(9, 6))
    # ax2 = ax1.twinx()

    ax1.plot()
    ax1.errorbar(divArray[:,0],divArray[:,1],divArray[:,2],label="div",color='r')
    ax1.errorbar(mauvArray[:,0],mauvArray[:,1],divArray[:,2],label="mauv",color='g')
    ax1.errorbar(dataArray[:,0],dataArray[:,1],divArray[:,2],label="gt",color='b')
    ax1.legend()
    # ax2.legend()
    ax1.set_xlabel('Num Samples')
    ax1.set_ylabel('Divergence/Mauve Divergence')

    # secondary y-axis label
    # ax2.set_ylabel('Gt divegence')
    # plt.plot(divArray[:,0],divArray[:,1],label="div")
    # plt.plot(mauvArray[:,0],mauvArray[:,1],label="mauv")
    # plt.plot(dataArray[:,0],dataArray[:,1],label="gt")
    # plt.legend()
    plt.savefig("{}.eps".format(dim), format="eps", bbox_inches='tight')
    plt.show()
