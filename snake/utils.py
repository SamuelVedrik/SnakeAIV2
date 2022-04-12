from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum
import random

class Direction(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    

def is_reverse(direction1: Direction, direction2: Direction):
    return abs(direction1.value - direction2.value) == 2

def get_reverse(direction: Direction):
    if direction == Direction.RIGHT:
        return Direction.LEFT
    elif direction == Direction.LEFT:
        return Direction.RIGHT
    elif direction == Direction.UP:
        return Direction.DOWN
    elif direction == Direction.DOWN:
        return Direction.Down
    else:
        raise ValueError(f"Poorly defined direction {direction}")
        
class SnakeBody:

    
    def __init__(self, snake_body: List[Tuple[int]], direction: Direction):
        # In (X, Y) Going > V 
        # Head is the last item in the list
        self.snake_body = snake_body
        self.direction = direction
    
    def get_legal(self):
        """
        Return the legal moves given the current direction.
        """
        if self.direction == Direction.RIGHT or self.direction == Direction.LEFT:
            return Direction.UP, Direction.DOWN
        elif self.direction == Direction.UP or self.direction == Direction.DOWN:
            return Direction.RIGHT, Direction.LEFT
        raise ValueError(f"Direction {self.direction} is invalid")

    def get_new_direction(self, new_direction: Direction):
        """
        Get the direction of the snake given direction. This is to prevent the snake from turning 180 degrees.
        """
        
        return new_direction if not is_reverse(self.direction, new_direction) else self.direction
    
    def get_new_head(self, new_direction: Direction):
        new_headx, new_heady = self.snake_body[-1]
        if new_direction == Direction.RIGHT:
            new_headx += 1
        elif new_direction == Direction.LEFT:
            new_headx -= 1
        elif new_direction == Direction.UP:
            new_heady -=1
        elif new_direction == Direction.DOWN:
            new_heady += 1
        return new_headx, new_heady

    def update(self, new_direction: Direction):
        new_direction = self.get_new_direction(new_direction)
        new_head = self.get_new_head(new_direction)
        self.snake_body.append(new_head)
        self.direction = new_direction
    
    def shrink(self):
        self.snake_body.pop(0)
        
    def get_head(self):
        return self.snake_body[-1]
    
    def collide_with_self(self):
        return self.get_head() in self.snake_body[:-1]