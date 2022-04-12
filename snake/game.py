from collections import namedtuple
import numpy as np
from .utils import Direction, SnakeBody


class BaseSnake:
    
    def __init__(self, board_x: int = 72, board_y: int = 48):
        self.board_x, self.board_y = board_x, board_y
        self.board = np.zeros((self.board_y, self.board_x))
        self.snake = SnakeBody(snake_body=[(8, 5), (9, 5), (10, 5)], direction=Direction.RIGHT)
        self.food = (-1, 1)
        self.score = 0
        self.game_over = False
        
        self._update_snake_board()
        self._update_food_pos()
        self._update_food_board()
        
        
    def _reset_board(self):
        self.board = self.board * 0
        
    def _update_snake_board(self):
        snake_body_array = np.array(self.snake.snake_body)
        # Set snake body to 1
        self.board[snake_body_array[:, 1], snake_body_array[:, 0]] = 1
    
    def _update_food_pos(self):
        valid_locations = np.transpose(np.nonzero(self.board != 1))
        rand_ind = np.random.choice(valid_locations.shape[0])
        self.food = (valid_locations[rand_ind, 1], valid_locations[rand_ind, 0])
        
    def _update_food_board(self):
        food_x, food_y = self.food
        self.board[food_y, food_x] = 2
    
    def check_game_over(self):
        headx, heady = self.snake.get_head()
        # out of bounds
        x_out_of_bounds = headx < 0 or headx > self.board_x-1 
        y_out_of_bounds = heady < 0 or heady > self.board_y-1
        collide_with_self = self.snake.collide_with_self()
        self.game_over = x_out_of_bounds or y_out_of_bounds or collide_with_self
        
    def step(self, direction: Direction):
        
        self.snake.update(direction)
        head = self.snake.get_head()
        need_new_food = False
        if head == self.food:
            self.score += 1
            need_new_food = True
        else:
            self.snake.shrink()
            
        self.check_game_over()
        if self.game_over:
            return
        
        self._reset_board()
        self._update_snake_board()
        if need_new_food:
            self._update_food_pos()
        self._update_food_board()
    
class TrainingSnake(BaseSnake):
    
    def step(self, direction: Direction):
        
        self.snake.update(direction)
        head = self.snake.get_head()
        need_new_food = False
        if head == self.food:
            self.score += 1
            need_new_food = True
        else:
            self.snake.shrink()
            
        self.check_game_over()
        # No updates for the board
        if self.game_over:
            return self.calculate_reward(need_new_food), self.get_curr_state()
        
        self._reset_board()
        self._update_snake_board()
        if need_new_food:
            self._update_food_pos()
        self._update_food_board()
        
        return self.calculate_reward(need_new_food), self.get_curr_state()

    def calculate_reward(self, has_eaten: bool):
        
        reward = -0.1
        if has_eaten:
            reward += 0.6
        if self.game_over:
            reward -= 10
        return reward
    
    def get_curr_state(self):
        dir_array = np.zeros((4,))
        dir_array[self.snake.direction.value] = 1
        return (self.board.copy(), dir_array)
            
class PlayableSnake(BaseSnake):
    
    def __init__(self, board_x: int, board_y: int):
        super().__init__(board_x, board_y)
        import pygame
        from pygame import Color
    
    def draw_board(self):
        self.game_window.fill(Color("black"))
        for snake_x, snake_y in self.snake.snake_body:
            pygame.draw.rect(self.game_window, Color("green"), pygame.Rect(snake_x*10, snake_y*10, 10, 10))
        pygame.draw.rect(self.game_window, Color("white"), pygame.Rect(self.food[0]*10, self.food[1]*10, 10, 10))
    
    def check_events(self):
        running = True
        direction = self.snake.direction
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
            # Whenever a key is pressed down
            elif event.type == pygame.KEYDOWN:
                # W -> Up; S -> Down; A -> Left; D -> Right
                if event.key == pygame.K_UP or event.key == ord('w'):
                    direction = Direction.UP
                if event.key == pygame.K_DOWN or event.key == ord('s'):
                    direction = Direction.DOWN
                if event.key == pygame.K_LEFT or event.key == ord('a'):
                    direction = Direction.LEFT
                if event.key == pygame.K_RIGHT or event.key == ord('d'):
                    direction = Direction.RIGHT
                # Esc -> Create event to quit the game
                if event.key == pygame.K_ESCAPE:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
        return direction, running
    
        
    def main_loop(self):
        pygame.init()
        pygame.display.set_caption('Snake Eater')
        self.game_window = pygame.display.set_mode((self.board_x*10, self.board_y*10))
        self.fps_controller = pygame.time.Clock()
        
        running = True
        while running:
            direction, running = self.check_events()
            self.step(direction)
            self.draw_board()
            # Refresh game screen
            pygame.display.update()
            # Refresh rate
            self.fps_controller.tick(20)
            
            # check game over state
            if self.game_over:
                running = False
        
        if self.game_over:
            print(f"You died! Final score: {self.score}")
            
def main():
    game = PlayableSnake(72, 48)
    game.main_loop()
    
            
if __name__ == "__main__":

    main()
    # Profiling code
    # import cProfile
    # import pstats
    
    # profiler = cProfile.Profile()
    # profiler.enable()
    # main()
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
        
        
            
        

    