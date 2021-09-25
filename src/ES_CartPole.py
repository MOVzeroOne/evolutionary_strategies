import torch 
import torch.nn as nn 
import torch.optim as optim 
import gym 
import copy 
import time 
from tqdm import tqdm
import matplotlib.pyplot as plt 

class network(nn.Module):
    def __init__(self,input_size,output_size,hidden_size):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,output_size))
    
    def forward(self,x):
        return nn.Softmax(dim=1)(self.layers(x))


class ES():
    def __init__(self,main_network,population_size=100,noise_std=0.5,lr=0.5) -> None:
        """
        main network is het network to be optimized 
        population size is how many mutated copies will be initilized each itertion
        noise std is the standard diviation of the noise used for exploration
        lr is the learning rate, how much we update the main network with the information recieved
        """
        
        self.main_network = main_network
        self.population_size = population_size
        self.noise_std = noise_std 
        self.lr = lr 
        self.mean_rewards = []

    def get_model(self):
        """
        returns a deep copy of the main model
        """
        return copy.deepcopy(self.main_network)

    def set_model(self,model):
        """
        sets the main model to be optimized to the supplied model
        """
        self.main_network = model

    def visualize(self):
        plt.cla()
        plt.plot(torch.arange(len(self.mean_rewards)),torch.tensor(self.mean_rewards),label="mean rewards")
        plt.legend()
        plt.pause(0.01)

    def run(self,performance_measurer,steps=200):
        """
        performance_measurer is here a function that takes an agent and calculates the performance / cummulative reward of the agent
        steps is the amount of steps the algorithm runs and optimizes the main model
        
        performance measurer should return a scalar reward not an array
        """
        plt.ion()
        for _ in tqdm(range(steps),ascii=True,desc="main loop"):
            #init childern 
            population = []
            rewards = torch.zeros(self.population_size)

            for _ in range(self.population_size):
                agent = copy.deepcopy(self.main_network)
                self.add_pertubations(agent)
                population.append(agent)
            
            #evalutate
            for index,agent in enumerate(population):
                rewards[index] = performance_measurer(agent)
            
            scaled_rewards = torch.nan_to_num((rewards - rewards.mean())/rewards.std())
            
            #update main network
            for index,agent in enumerate(population):
                self.update_main_network(agent,scaled_rewards[index])
            
            self.mean_rewards.append(rewards.mean())
            self.visualize()
            

    def update_main_network(self,agent,scaled_reward):
        """
        updates the trainable parameters of the main network given a perturbed network and it's scaled reward
        """
        
        for param,param_main in zip(agent.parameters(),self.main_network.parameters()):
            if(param_main.requires_grad):
                param.requires_grad = False
                param_main.requires_grad = False 

                param_main += self.lr/(self.population_size*self.noise_std) * scaled_reward * (param-param_main)
                
                param.requires_grad = True
                param_main.requires_grad = True 
    
    def add_pertubations(self,agent):
        """
        add pertubations to trainable weights based upon the self.noise_std
        """
        for param in agent.parameters():
            if(param.requires_grad):
                param.requires_grad = False
                param += torch.randn_like(param)*self.noise_std
                param.requires_grad = True

def cartpole(agent):
    with torch.no_grad():
        env = gym.make('CartPole-v0')
        obs = env.reset()
        cummulative_reward = 0
        while(True):
            action = torch.argmax(agent(torch.tensor(obs,dtype=torch.float).view(1,-1)))
            obs, reward, done, _ = env.step(action.numpy())
            cummulative_reward += reward
            if(done):
                break 
    env.close()
    return cummulative_reward
        

if __name__ == "__main__":
    net = network(4,2,50)
    opt = ES(net)
    opt.run(cartpole)

    torch.save(opt.get_model().state_dict(),"cartpole_weights.pt")