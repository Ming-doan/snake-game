import torch
import numpy as np
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet
import time

class SnakePlayer:
    def __init__(self, model_path='model/model.pth'):
        self.model = Linear_QNet(11, 256, 3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set to evaluation mode
        self.game = SnakeGameAI()
        
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
            ]

        return np.array(state, dtype=int)

    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

    def play_game(self, speed=1):
        """
        Play a single game using the trained model
        Args:
            speed: Frame delay in milliseconds (lower = faster)
        """
        game_over = False
        while not game_over:
            # Get current state
            state = self.get_state(self.game)
            
            # Get move from model
            final_move = self.get_action(state)
            
            # Perform move
            reward, game_over, score = self.game.play_step(final_move)
            
            # Control game speed
            time.sleep(speed / 1000)
        
        return score

def main():
    # Create player instance
    player = SnakePlayer(model_path="./model/model_600_epochs.pth")
    
    # Play games until user quits
    while True:
        score = player.play_game()
        print(f'Game Over! Score: {score}')
        
        # Ask if user wants to play again
        # play_again = input("Play again? (y/n): ")
        # play_again = "y"
        # if play_again.lower() != 'y':
        #     break
        player.game.reset()

if __name__ == '__main__':
    main()