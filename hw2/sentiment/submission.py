#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    feat = {}
    for w in x.split(' '):
        if w not in feat:
            feat[w] = 1
        else:
            feat[w] += 1
    return feat
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes: 
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. 
    - The identity function may be used as the featureExtractor function during testing.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    def predictor(x):
        feat = featureExtractor(x)
        score = dotProduct(weights, feat)
        y = 1 if score >= 0 else -1
        return y

    for epoch in range(numEpochs):
        for train_example in trainExamples:
            x, y = train_example
            feat = featureExtractor(x)
            if dotProduct(weights, feat) * y < 1:
                grads = {}
                for f, v in feat.items():
                    grads[f] = - v * y
                increment(weights, -eta, grads)
            
        print('Epoch {}: training error = {:.2f}%, validation error = {:.2f}%'.format(
             epoch, evaluatePredictor(trainExamples, predictor) * 100, 
             evaluatePredictor(validationExamples, predictor) * 100))

    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = {}
        keys = random.sample(weights.keys(), random.randint(1, len(weights)))
        for k in keys:
            phi[k] = random.uniform(0, 1)
        score = dotProduct(weights, phi)
        y = 1 if score >= 0 else -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3e: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        counts = {}
        x = x.replace(' ', '')
        for i in range(len(x) - n + 1):
            ngram = x[i:i + n]
            if ngram in counts:
                counts[ngram] += 1
            else:
                counts[ngram] = 1
        return counts
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3f:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))

# {1:48.54%, 2:40.49%, 3:31.60%, 4:28.42%, 5:27.38%, 6:27.35%, 7:27.10%, 8:29.29%, 9:31.04%, 10:33.88%}

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)
    def get_squared_euclidean_dist(d1, d2):
        if len(d1) < len(d2):
            return get_squared_euclidean_dist(d2, d1)
        else:
            return sum((d1.get(f, 0) - v) ** 2 for f, v in d2.items())

    # Randomly initialize centroids:
    centroids = random.sample(examples, K)
    assignments = [None for _ in range(len(examples))]
    # Step through epochs:
    for epoch in range(maxEpochs):
        # Assign each example point to the nearest centroid:
        new_assginments = [None for _ in range(len(examples))]
        new_centroids = [{} for _ in range(K)]
        cluster_sizes = [0 for _ in range(K)]
        for i, example in enumerate(examples):
            min_dist = float('inf')
            for j, centroid in enumerate(centroids):
                dist = get_squared_euclidean_dist(centroid, example)
                if dist < min_dist:
                    min_dist = dist
                    nearest_c_idx = j
            new_assginments[i] = nearest_c_idx
            increment(new_centroids[nearest_c_idx], 1, example)
            cluster_sizes[nearest_c_idx] += 1
        
        if new_assginments == assignments:
            # Stable centroids found
            break
        else:
            assignments = new_assginments
            # Update centroids:
            for i, c in enumerate(new_centroids):
                for k in c:
                    c[k] /= cluster_sizes[i]
            centroids = new_centroids
    
    # Compute loss:
    loss = 0
    for example, c_idx in zip(examples, assignments):
        loss += get_squared_euclidean_dist(example, centroids[c_idx])
    print(f'Loss = {loss:.3f}')

    return centroids, assignments, loss    
    # END_YOUR_CODE
