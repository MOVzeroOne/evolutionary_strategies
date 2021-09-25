import torch 
import torch.nn as nn 
import gym 


class network(nn.Module):
    def __init__(self,input_size,output_size,hidden_size):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,output_size))
    
    def forward(self,x):
        return nn.Softmax(dim=1)(self.layers(x))

def cartpole(agent):
    with torch.no_grad():
        env = gym.make('CartPole-v0')
        env._max_episode_steps = 1000
        obs = env.reset()
        cummulative_reward = 0
        while(True):
            env.render()
            action = torch.argmax(agent(torch.tensor(obs,dtype=torch.float).view(1,-1)))
            obs, reward, done, _ = env.step(action.numpy())
            cummulative_reward += reward
            if(done):
                break
        
    env.close()

    return cummulative_reward
        

if __name__ == "__main__":
    net = network(4,2,50)
    net.load_state_dict(torch.load("cartpole_weights.pt"))
    print("cummulative_reward: ",cartpole(net))


