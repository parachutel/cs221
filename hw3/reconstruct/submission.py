from typing import Callable, List, Set

import shell
import util
import wordsegUtil


############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # Number of processed characters
        return 0
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return len(self.query) == state
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        succ_and_cost = []
        for seg_len in range(1, len(self.query) - state + 1):
            action = self.query[state:state + seg_len]
            new_state = state + seg_len
            cost = self.unigramCost(action)
            succ_and_cost.append((action, new_state, cost))
        return succ_and_cost
        # END_YOUR_CODE


def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    # print(ucs.actions)
    return ' '.join(ucs.actions)
    # END_YOUR_CODE


############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords: List[str], bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # state = (current index, prev fill)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.queryWords)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        curr_idx, prev_fill = state[0], state[1]
        curr_str = self.queryWords[curr_idx]
        possible_fills = self.possibleFills(curr_str)
        if len(possible_fills) == 0:
            possible_fills = [curr_str]
        succ_and_cost = []
        for possible_fill in possible_fills:
            new_state = (curr_idx + 1, possible_fill)
            cost = self.bigramCost(prev_fill, possible_fill)
            succ_and_cost.append((possible_fill, new_state, cost))
        return succ_and_cost
        # END_YOUR_CODE


def insertVowels(queryWords: List[str], bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords) == 0:
        return ''
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    # print(ucs.actions)
    return ' '.join(ucs.actions)
    # END_YOUR_CODE


############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query: str, bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # (curr fill, n processed chars)
        return (wordsegUtil.SENTENCE_BEGIN, 0)
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state[1] == len(self.query)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)
        curr_fill, n_proccessed = state[0], state[1]
        succ_and_cost = []
        for next_seg_len in range(1, len(self.query) - n_proccessed + 1):
            next_unfilled_str = self.query[n_proccessed:n_proccessed + next_seg_len]
            for next_possible_fill in self.possibleFills(next_unfilled_str):
                new_state = (next_possible_fill, n_proccessed + next_seg_len)
                cost = self.bigramCost(curr_fill, next_possible_fill)
                succ_and_cost.append((next_possible_fill, new_state, cost))
        return succ_and_cost
        # END_YOUR_CODE


def segmentAndInsert(query: str, bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    # print(ucs.actions)
    return ' '.join(ucs.actions)
    # END_YOUR_CODE


############################################################

if __name__ == '__main__':
    shell.main()
