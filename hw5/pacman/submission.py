from util import manhattanDistance
from game import Directions
import random, util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
    """
    The evaluation function takes in the current GameState (defined in pacman.py)
    and a proposed action and returns a rough estimate of the resulting successor
    GameState's value.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Don't forget to limit the search depth using self.depth. Also, avoid modifying
      self.depth directly (e.g., when implementing depth-limited search) since it
      is a member variable that should stay fixed throughout runtime.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    def _nextIndex(i):
        return (i + 1) % gameState.getNumAgents()

    def minimax(s, d, i):
        if s.isWin() or s.isLose() or len(s.getLegalActions(self.index)) == 0:
            # IsEnd
            return s.getScore(), None
        if d == 0:
            return self.evaluationFunction(s), None

        legalActions = s.getLegalActions(i)
        succs = [s.generateSuccessor(i, a) for a in legalActions]
        if i == 0:
            values = [minimax(succ, d, _nextIndex(i))[0] for succ in succs]
            bestScore = max(values)
            bestIndex = random.choice([
                i for i in range(len(values)) if values[i] == bestScore])
            return bestScore, legalActions[bestIndex]
        else: # i > 0
            d -=  0 if _nextIndex(i) > 0 else 1
            values = [minimax(succ, d, _nextIndex(i))[0] for succ in succs]
            return min(values), None

    res = minimax(gameState, self.depth, 0)
    # print('!DEBUG:', res)
    return res[1]
    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
    You may reference the pseudocode for Alpha-Beta pruning here:
    en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 36 lines of code, but don't worry if you deviate from this)
    def _nextIndex(i):
        return (i + 1) % gameState.getNumAgents()
    
    def alphabeta(s, d, i, alpha, beta):
        if s.isWin() or s.isLose() or len(s.getLegalActions(self.index)) == 0:
            # IsEnd
            return s.getScore(), None
        if d == 0:
            return self.evaluationFunction(s), None

        legalActions = s.getLegalActions(i)
        succs = [s.generateSuccessor(i, a) for a in legalActions]
        if i == 0: # maximizingPlayer
            value = -float('inf')
            values = []
            for succ in succs:
                value = max(value, alphabeta(succ, d, _nextIndex(i), alpha, beta)[0])
                values.append(value)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            bestIndex = random.choice([
                i for i in range(len(values)) if values[i] == value])
            return value, legalActions[bestIndex]
        else: # minimizingPlayer
            d -=  0 if _nextIndex(i) > 0 else 1
            value = float('inf')
            for succ in succs:
                value = min(value, alphabeta(succ, d, _nextIndex(i), alpha, beta)[0])
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value, None

    res = alphabeta(gameState, self.depth, 0, -float('inf'), float('inf'))
    # print('!DEBUG:', res)
    return res[1]
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    def _nextIndex(i):
        return (i + 1) % gameState.getNumAgents()

    def expectimax(s, d, i):
        if s.isWin() or s.isLose() or len(s.getLegalActions(self.index)) == 0:
            # IsEnd
            return s.getScore(), None
        if d == 0:
            return self.evaluationFunction(s), None

        legalActions = s.getLegalActions(i)
        succs = [s.generateSuccessor(i, a) for a in legalActions]
        if i == 0:
            values = [expectimax(succ, d, _nextIndex(i))[0] for succ in succs]
            bestScore = max(values)
            bestIndex = random.choice([
                i for i in range(len(values)) if values[i] == bestScore])
            return bestScore, legalActions[bestIndex]
        else: # i > 0
            d -=  0 if _nextIndex(i) > 0 else 1
            values = [expectimax(succ, d, _nextIndex(i))[0] for succ in succs]
            return sum(values) / len(legalActions), None

    res = expectimax(gameState, self.depth, 0)
    # print('!DEBUG:', res)
    return res[1]
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
    """
        Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
    """

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    # Features
    agent_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()
    scared_times = [gs.scaredTimer for gs in currentGameState.getGhostStates()]
    scared_bools = [st > 0 for st in scared_times]
    n_foods = currentGameState.getNumFood()
    foods = currentGameState.getFood().asList()

    weights = {'food_count': 5,
               'food_dist': 0.7,
               'capsules': 20,
               'eat_ghost': 15,
               'escape_ghost': 10}

    # Food count
    food_count_score = 0 if n_foods == 1 \
        else -n_foods * weights['food_count']

    # Food distance
    food_dist_score = 0
    if n_foods > 0:
        min_food_dist = min([util.manhattanDistance(agent_position, food)
            for food in foods])
        food_dist_score = -min_food_dist * weights['food_dist']


    # Capsule distance
    capsules_score = 0
    if len(currentGameState.getCapsules()) > 0:
        min_capsule_dist = min([util.manhattanDistance(agent_position, c_pos) 
            for c_pos in currentGameState.getCapsules()])
        if min_capsule_dist <= 10:
            capsules_score = weights['capsules'] if min_capsule_dist <= 1 \
                else 1 / min_capsule_dist * weights['capsules']


    # Eat ghost
    eat_ghost_score = 0
    if any(scared_bools):
        min_scared_ghost_factor = min([
            util.manhattanDistance(agent_position, g_pos) / scared_times[i]
            for i, g_pos in enumerate(ghost_positions) if scared_bools[i]])
        eat_ghost_score = 1 / min_scared_ghost_factor * weights['eat_ghost']

    # Escape ghost
    escape_ghost_score = 0
    if not all(scared_bools):
        min_ghost_dist = min([
            util.manhattanDistance(agent_position, g_pos) 
            for i, g_pos in enumerate(ghost_positions) if not scared_bools[i]])
        escape_ghost_score = -weights['escape_ghost'] * 3 if min_ghost_dist == 0 \
            else -1 / min_ghost_dist * weights['escape_ghost']


    return currentGameState.getScore() \
            + food_count_score \
            + food_dist_score \
            + capsules_score \
            + eat_ghost_score \
            + escape_ghost_score

    # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
