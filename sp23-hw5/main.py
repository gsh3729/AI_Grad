import time
import random
import numpy as np
import matplotlib.pyplot as plt
from hw5.bayes_net import BayesNet
from hw5.utils import Node
import matplotlib.pyplot as plt
import time


def CSRW_BN():
    C = Node("C", [], [0.5])
    S = Node("S", [C], [0.1,0.5])
    R = Node("R", [C], [0.8,0.2])
    W = Node("W", [S,R], [0.99,0.9,0.9,0])
    return BayesNet([C,S,R,W])

def random_BN(numNodes, maxParents):
    nodes = []
    for n in range(numNodes):
        numParents = random.randint(0, min(maxParents,len(nodes)))
        parents = random.sample(nodes, k=numParents)
        # print("parents: ", parents)
        CPT = np.random.random(size=2**numParents)
        nodes.append(Node(str(n), parents, CPT))
    return BayesNet(nodes)


def main():
    bn = random_BN(20,5)
    n = random.sample(bn.nodes, k=6)
    evidence = dict(zip(n[:5], np.random.randint(0,2,10)))
    node = n[-1]
    samples, weights = bn.gen_samples(50000, evidence, True)
    prob_dist = bn.estimate_dist(node, samples, weights)
    print("Prob distribution: ", prob_dist)

    lw = []
    for i in range(100,50000,100) :
        prob_dist = bn.estimate_dist(node, samples[0:i+1], weights)
        lw.append(prob_dist[0])
    
    samples, weights = bn.gen_samples(50000, evidence, True)
    prob_dist = bn.estimate_dist(node, samples, weights)
    print("Prob distribution: ", prob_dist)

    gs = []
    for i in range(100,50000,100) :
        prob_dist = bn.estimate_dist(node, samples[0:i+1], weights)
        gs.append(prob_dist[0])

    plt.plot(lw, label = "Likelihood weighting")
    plt.plot(gs, label = "Gibbs")
    plt.title("Prob versus Number of Samples")
    plt.legend()
    plt.show()

    lwt = []
    gst = []
    for i in range(100,10000,100):
        start = time.time()
        bn.gen_samples(i, evidence, True)
        end = time.time()
        t = end-start  
        lwt.append(t)
        start = time.time()
        bn.gen_samples(i, evidence, False)
        end = time.time()
        t = end-start  
        gst.append(t)

    plt.plot(lwt, label = "Likelihood weighting")
    plt.plot(gst, label = "Gibbs")
    plt.title("Time versus Number of Samples")
    plt.legend()
    plt.show()




    # Testing
    # bn = CSRW_BN()
    # evidence = {bn.nodes[1] : 1, bn.nodes[3] : 1 }
    # samples, weights = bn.gen_samples(10000, evidence, False)
    # prob_dist = bn.estimate_dist(bn.nodes[0], samples, weights)
    # print("Prob distribution: ", prob_dist)

if __name__ == "__main__":
    main()