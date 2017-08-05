import cv2
import numpy as np

from enduro.agent import Agent
from enduro.action import Action


def phi_position(grid, act):
    # Feature representing position and act
    
    grid = grid.tolist()
    if grid.index(2)>5:
        if act == 0:
            return 1
        elif act == 2:
            return 5
        else:
            return 1
    elif grid.index(2)<4:
        if act == 0:
            return 1
        elif act == 1:
            return 5
        else:
            return 1
    else:
        if act == 0:
            return 10
        elif act == 3:
            return 1
        else:
            return 2

def phi_action(act):
    # Feature representing act, turn off this feature for a better reward LOL

    return 0
    if act == 1 or act == 2:
        return 2
    elif act == 0:
        return 5
    else:
        return 1

def phi_opponent(grid_agent, grid_opponent, act):
    # Feature representing opponents
    
    opponents = np.where(grid_opponent==1)[1]
    agent = np.where(grid_agent==2)[0][0]
    
    if any(x>=0 and x<3 for x in opponents-agent):
        if act == 2:
            return 20
        else:
            return 0
    elif any(x<=0 and x>-3 for x in opponents-agent):
        if act == 1:
            return 20
        else:
            return 0      
    else:
        if act == 3:
            return 0
        else:
            return 5


class FunctionApproximationAgent(Agent):
    
    def __init__(self):
        super(FunctionApproximationAgent, self).__init__()
        # The horizon defines how far the agent can see
        self.horizon_row = 5

        self.grid_cols = 10
        # The state is defined as a tuple of the agent's x position and the
        # x position of the closest opponent which is lower than the horizon,
        # if any is present. There are four actions and so the Q(s, a) table
        # has size of 10 * (10 + 1) * 4 = 440.
        self.Q = np.ones((self.grid_cols, self.grid_cols + 1, 4))

        # Add initial bias toward moving forward. This is not necessary,
        # however it speeds up learning significantly, since the game does
        # not provide negative reward if no cars have been passed by.
        self.Q[:, :, 0] += 1.

        # Helper dictionaries that allow us to move from actions to
        # Q table indices and vice versa
        self.idx2act = {i: a for i, a in enumerate(self.getActionsSet())}
        self.act2idx = {a: i for i, a in enumerate(self.getActionsSet())}

        # Learning rate
        self.alpha = 0.00001
        # Discounting factor
        self.gamma = 0.9
        # Exploration rate
        self.epsilon = 0.01

        # Log the obtained reward during learning
        self.last_episode = 1
        self.episode_log = np.zeros(6510) - 1.
        self.log = []
        
        self.theta = np.random.uniform(low=0.1, high=0.5, size=2)
        self.theta = np.array([0.1,0.1,0.1])
        self.last_action = 0
        self.total_reward = 0

    def initialise(self, road, cars, speed, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        print 'Total reward = '+str(self.total_reward)+"; Theta = "+str(self.theta)
        self.total_reward = 0
        self.next_state2 = grid[:5]

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        self.state2 = self.next_state2

        # If exploring
        if np.random.uniform(0., 1.) < self.epsilon:
            # Select a random action using softmax
            idx = np.random.choice(4)#, p=probs)
            self.action = self.idx2act[idx]
            self.last_action = idx
        else:
            # Select the greedy action
            idx = self.argmax_Qsa(self.state2)
            #print idx
            self.action = self.idx2act[idx]
            self.last_action = idx

        self.reward = self.move(self.action)
        self.total_reward += self.reward

    def sense(self, road, cars, speed, grid):
        """ Constructs the next state from sensory signals.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """
        self.next_state2 = grid[:5]

        # Visualise the environment grid
        #cv2.imshow("Enduro", self._image)

    def learn(self):
        """ Performs the learning procedure. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        _max_Qsa = self.max_Qsa(self.next_state2)
        self.phi = [phi_position(self.state2[0], self.last_action), phi_action(self.last_action),
                   phi_opponent(self.state2[0], self.state2[1:], self.last_action)]
        
        Q_sa = np.dot(self.theta.T, self.phi)
        
        self.theta = self.theta + np.dot(self.alpha*(self.reward + self.gamma * _max_Qsa - Q_sa), self.phi)
        
        pass

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        cv2.imshow("Enduro", self._image)
        #cv2.waitKey(1)

    def QsaAll(self, grid):
        self.allQsa = []
        for each in [0,1,2,3]:
            self.phi = [phi_position(grid[0], each), phi_action(each), 
                        phi_opponent(grid[0], grid[1:], each)]
            _ = np.dot(self.theta.T, self.phi)
            self.allQsa.append(_)
        return self.allQsa
    
    def max_Qsa(self, grid):
        # Calculates max Qsa
        return np.max(self.QsaAll(grid))
    
    def argmax_Qsa(self, grid):
        # Calculates argmax Qsa
        return np.argmax(self.QsaAll(grid))
    
    '''
    # Obsolete function
    
    def maxQsa(self, state):
        return np.max(self.Q[state[0], state[1], :])

    def argmaxQsa(self, state):
        return np.argmax(self.Q[state[0], state[1], :])
    '''



if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=2000, draw=True)
    print 'Total reward: ' + str(a.total_reward)