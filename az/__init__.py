"""AlphaZero-style learning components for Azul (4-player).

Pipeline: encoder -> net -> PUCT MCTS -> self-play -> train -> repeat.

Training runs offline (PyTorch, GPU). Serving loads exported numpy weights
so the web app needs only numpy. This package targets the 4-player game
(9 factories) — the web app's default and the first format we optimise.
"""

NUM_PLAYERS = 4
NUM_FACTORIES = 9  # GameState.NUM_FACTORIES[4-2] == 9
NUM_COLORS = 5
NUM_PATTERN_LINES = 5
