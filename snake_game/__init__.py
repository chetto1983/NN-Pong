"""Snake controlled by analytically constructed neural-network weights."""

from .network import DOWN, LEFT, RIGHT, UP, SnakeNet, create_network, load_network
from .shortcut_network import ShortcutSnakeNet
from .simulation import DIRECTIONS, SnakeSimulation

__all__ = ['SnakeNet', 'ShortcutSnakeNet', 'SnakeSimulation', 'create_network',
           'load_network', 'UP', 'RIGHT', 'DOWN', 'LEFT', 'DIRECTIONS']
