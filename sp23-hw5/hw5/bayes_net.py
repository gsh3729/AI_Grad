import numpy as np
from hw5.utils import Node


class BayesNet:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.topological_sort()
        self.set_children()
    
    def topological_sort(self):
        new = []
        while self.nodes:
            for n in self.nodes:
                if set(n.parents) <= set(new):
                    new.append(n)
                    self.nodes.remove(n)
        self.nodes = new

    def set_children(self):
        for n in self.nodes:
            for p in n.parents:
                p.children.add(n)


    """
    4.1 Generate sample and weight from Bayes net by likelihood weighting
    """        
    def weighted_sample(self, evidence: dict={}):
        """
        Args:
            evidence (Dict): {Node:0/1} mappings for evidence nodes.
        Returns:
            Dict: {Node:0/1} mappings for all nodes.
            Float: Sample weight. 
        """
        sample = {}
        weight = 1

        for node in self.nodes: 
            if node in evidence:
                parents_evidences = []
                for parent in node.parents:
                    parents_evidences.append(sample[parent])
                if evidence[node] == 0:
                    weight *= node.get_probs(parents_evidences)[0]
                else:
                    weight *= node.get_probs(parents_evidences)[1]
                sample[node] = evidence[node]
            else:
                parents_evidences = []
                for p in node.parents:
                    parents_evidences.append(sample[p])
                sample[node] = node.sample(parents_evidences)
           
        return sample, weight

    """
    4.2 Generate sample from Bayes net by Gibbs sampling
    """  
    def gibbs_sample(self, node: Node, sample: dict):
        """
        Args:
            node (Node): Node to resample.
            sample (Dict): {node:0/1} mappings for all nodes.
        Returns:
            Dict: {Node:0/1} mappings for all nodes.
        """ 
        new = sample.copy()
        evidence = []
        prob = np.ones((2, ))
        for i in node.parents:
            evidence.append(sample[i])
        prob = np.multiply(prob, node.get_probs(evidence))
        for i in node.children:
            evidence.clear()
            for p in i.parents:
                if (p == node):
                    evidence.append(1)
                    continue
                evidence.append(sample[p])
            p1 = i.get_probs(evidence)
            evidence.clear()
            for p in i.parents:
                if (p == node):
                    evidence.append(1)
                    continue
                evidence.append(sample[p])
            p2 = i.get_probs(evidence)  
            child_prob = np.ones((2, ))
            if (sample[i] == 0):
                child_prob[0] = p1[0]
                child_prob[1] = p2[0]
            else:
                child_prob[0] = p1[1]
                child_prob[1] = p2[1]
            prob = np.multiply(prob, child_prob)
        prob_factor = 1 / sum(prob)
        prob = [prob_factor * p for p in prob]
        new[node] = np.random.choice([0,1], p=prob)
        return new

    """
    4.3 Generate a list of samples given evidence and estimate the distribution.
    """  
    def gen_samples(self, numSamples: int, evidence: dict={}, LW: bool=True):
        """
        Args:
            numSamples (int).
            evidence (Dict): {Node:0/1} mappings for evidence nodes.
            LW (bool): Use likelihood weighting if True, Gibbs sampling if False.
        Returns:
            List[Dict]: List of {node:0/1} mappings for all nodes.
            List[float]: List of corresponding weights.
        """       

        samples = []
        non_evidence_nodes = []
        if not LW:
            weights = [1] * numSamples
            sample={}
            for k in self.nodes:
                sample[k] = 0
            samples.append(sample)
            for i in range(len(self.nodes)):
                if self.nodes[i] not in evidence:
                    non_evidence_nodes.append(i)

        else:
            weights = []

        for i in range(numSamples):
            if LW:
                s, w = self.weighted_sample(evidence)
                samples.append(s)
                weights.append(w)
            else:
                s = self.gibbs_sample(self.nodes[non_evidence_nodes[i%len(non_evidence_nodes)]], samples[-1]) 
                samples.append(s)

        return samples, weights

    def estimate_dist(self, node: Node, samples: list[dict], weights: list[float]):
        """
        Args:
            node (Node): Node whose distribution we will estimate.
            samples (List[Dict]): List of {node:0/1} mappings for all nodes.
            weights (List[float]: List of corresponding weights.
        Returns:
            Tuple(float, float): Estimated distribution of node variable.
        """           
        # TODO: 4.3
        num_zeros = 0
        for sample in samples:
            if (sample[node] == 0):
                num_zeros += 1
        p_0 = num_zeros / len(samples)

        return (p_0, 1-p_0)
