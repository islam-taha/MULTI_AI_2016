# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [
        self.evaluationFunction(
            gameState,
            action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[
        index] == bestScore]
    # Pick randomly among the best
    chosenIndex = random.choice(bestIndices)

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [
        ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
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

  def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
    self.index = 0  # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
  """
    Returns the minimax action from the current gameState using self.depth
    and self.evaluationFunction.

    Here are some method calls that might be useful when implementing minimax.

    gameState.getLegalActions(agentIndex):
      Returns a list of legal actions for an agent
      agentIndex=0 means Pacman, ghosts are >= 1

    gameState.generateSuccessor(agentIndex, action):
      Returns the successor game state after an agent takes an action

    gameState.getNumAgents():
      Returns the total number of agents in the game
  """

  def getAction(self, gameState):
    # start at depth 1, with pacman agent (indexed as 0)
    return self.minMaxSearch(gameState, 1, 0)

  def minMaxSearch(self, gameState, currentDepth, agentIndex):
    # check if the game ended.
    if currentDepth > self.depth or gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)  # who won

    # get legal moves based on the current agent.
    # legal moves for pacman differ from the legal moves of ghosts.
    legalMoves = [action for action in gameState.getLegalActions(
        agentIndex) if action != Directions.STOP]

    # update depth and give the turn to another agent.
    # we have 3 agents indexed as [0, 1, 2] the 0th agent is the pacman
    # player.
    nextDepth = currentDepth + \
        (1 if (agentIndex + 1) >= gameState.getNumAgents() else 0)
    nextIndex = (agentIndex + 1) % gameState.getNumAgents()

    # go to your successors.
    results = [
        self.minMaxSearch(
            gameState.generateSuccessor(
                agentIndex,
                action),
            nextDepth,
            nextIndex) for action in legalMoves]

    # if we are at the parent node we need to return the required actions
    # to be performed.
    if agentIndex == 0 and currentDepth == 1:
      # chose one of the best sloutions.
      bestMove = max(results)
      bestIndices = [
          index for index in range(
              len(results)) if results[index] == bestMove]
      chosenIndex = random.choice(bestIndices)
      return legalMoves[chosenIndex]

    # if we are at any level but the first one we check
    # if the current player is tha pacman w perform the MAX nodes to chose the optimal solution.
    # otherwise we perform the MIN nodes to get the optimal solution.
    if agentIndex == 0:  # pacman turn.
      bestMove = max(results)
    else:
      bestMove = min(results)

    # return chosen best move. :D
    return bestMove


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    self.INF = 1000000000000000000000000

    def max_score(gameState, alpha, beta, depth, ghosts):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState)

      totalLegalActions = gameState.getLegalActions(0)
      value = -self.INF

      for move in totalLegalActions:
        value = max(value, min_score(gameState.generateSuccessor(
            0, move), alpha, beta, depth, 1, ghosts))
        if value > beta:
          return value
        alpha = max(alpha, value)

      return value

    def min_score(gameState, alpha, beta, depth, agentNumber, ghosts):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState)

      value = self.INF
      totalLegalActions = gameState.getLegalActions(agentNumber)
      if agentNumber == ghosts:
        for move in totalLegalActions:
          value = min(value, max_score(gameState.generateSuccessor(
              agentNumber, move), alpha, beta, depth + 1, ghosts))
          if value < alpha:
            return value
          beta = min(beta, value)
      else:
        for move in totalLegalActions:
          value = min(value, min_score(gameState.generateSuccessor(
              agentNumber, move), alpha, beta, depth, agentNumber + 1, ghosts))
          if value < alpha:
            return value
          beta = min(beta, value)
      return value

    pq = util.PriorityQueue()
    bestAction = "Stop"
    alpha = -self.INF
    score = -self.INF
    beta = self.INF
    for move in gameState.getLegalActions(0):
      cur_score = score
      score = min_score(gameState.generateSuccessor(
          0, move), alpha, beta, 0, 1, gameState.getNumAgents() - 1)
      if score > cur_score:
        bestAction = move
      if score > beta:
        return bestAction
      alpha = max(alpha, score)
      pq.push(move, score)

    while not pq.isEmpty():
      bestAction = pq.pop()
      
    return bestAction

    util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
