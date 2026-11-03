"""Snake controlled by analytically constructed neural-network weights."""

from .network import DOWN, LEFT, RIGHT, UP, SnakeNet
from .simulation import DIRECTIONS, SnakeSimulation

__all__ = ['SnakeNet', 'SnakeSimulation', 'UP', 'RIGHT', 'DOWN', 'LEFT', 'DIRECTIONS']
