import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
            #Q = { ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], L): 10,
            #([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], R): -50}
            #Q matrix is dictonary with (state, action) as key and Q as value 
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions  # passed in constructor, in our case, 3 actions, 0=F, 1=L, 2=R

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        action = 0
        q = 0

        if random.random() < self.epsilon: #take a random action
            action = random.choice(self.actions) #chose random element from actions list
            q = self.getQ(state, action) #q for (randomly chosen action) action-state pair
        else: #take a greedy action
            q_list = [self.getQ(state, a) for a in self.actions] #get q's for each action
            max_q_indicies = [i for i, x in enumerate(q) if x == max(q_list)] #https://stackoverflow.com/questions/6294179/how-to-find-all-occurrences-of-an-element-in-a-list
            if len(max_q_indicies) > 1: #if we have more than one instance of a maximum Q (two or more action-state pairs with same Q)
                chosen_action_index = random.choice(max_q_indicies) #chose random index from indicies associated with max q
                action = self.actions[chosen_action_index]
            else: #only one action-state pair with max Q
                action = self.actions[max_q_indicies[0]]
            
            q = max(q_list) #return highest q value

        if return_q:
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the Bellman update
            equation
        '''

        q1 = self.q.get((state1, action1), None)

        if q1 is None: #didn't find state-action pair in dict
            self.q[(state1, action1)] = reward #set q to reward
        else: #implement Bellman update equation
            max_q2 = max([self.getQ(state2, a) for a in self.actions]) #max Q from ALL state_2-action pairs
            #Q(s1, a1) = Q(s1, a1) + alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
            self.q[(state1, action1)] = q1 + self.alpha*(reward + self.gamma*max_q2 - q1)
