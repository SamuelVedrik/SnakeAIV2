from .model import SnakeAI, ExperienceBuffer
import torch
import torch.nn as nn
from torch import optim
from snake.game import TrainingSnake
from snake.utils import Direction
from tqdm import tqdm


def train(model: SnakeAI, buffer: ExperienceBuffer, sample_size, criterion, optimizer):

    state, action, reward, next_state = buffer.sample_from_experience(sample_size)
    Q_curr, Q_next = model.get_Q(state, next_state)
    Q_curr = Q_curr[torch.arange(sample_size), action.long()] # Only take the Q of the action we took.
    Q_next = Q_next.max(dim=1)[0] # Take the best action
    
    loss = criterion(Q_curr, reward.to(model.device) + (model.gamma * Q_next))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.update_step()
    model.update_target_net()
    return loss.item()
    
def process_state(state):
    state_board, state_dir = state
    return torch.FloatTensor(state_board).unsqueeze(0), torch.FloatTensor(state_dir)
    
def training_loop(episodes):
    
    buffer_size=24
    buffer = ExperienceBuffer(buffer_size)
    epsilon = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SnakeAI(device=device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.policy_net.parameters())
    
    # Fill buffer with stuff
    step = 0
    for _ in range(buffer_size):
        game = TrainingSnake()
        curr_state = game.get_curr_state()
        while not game.game_over:
            action = model.get_action(process_state(curr_state), epsilon)
            reward, next_state = game.step(Direction(action.item()))
            buffer.collect((process_state(curr_state), action, reward, process_state(next_state)))
            step += 1
            if step > buffer_size:
                break
    
    step = 127 # Train every 128 steps
    all_rewards = []
    for episode in tqdm(range(episodes)):
        game = TrainingSnake()
        curr_state = game.get_curr_state()
        total_game_reward = 0
        lifespan = 0
        while not game.game_over:
            action = model.get_action(process_state(curr_state), epsilon)
            reward, next_state = game.step(Direction(action.item()))
            buffer.collect((process_state(curr_state), action, reward, process_state(next_state)))
            step = (step + 1) % 128
            total_game_reward += reward
            lifespan += 1
            if step == 0:
                for _ in range(4):
                    train(model, buffer, 16, criterion, optimizer)
        
        all_rewards.append(total_game_reward)
        if episode % 100 == 0: 
            print(f"Current reward: {total_game_reward} | Current Lifespan: {lifespan}")
            model.save_policy_net("./snake_ai.pth")

        epsilon = max(epsilon - 1e-3, 1e-2)

    model.save_policy_net("./snake_ai.pth")
    
        
if __name__ == "__main__":
    training_loop(200)