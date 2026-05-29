"""Fixed action space for 4-player Azul and mapping to framework moves.

A framework move is a tuple (move_type, factory_id, TileGrab). Crucially,
given (source, tile colour, destination) the TileGrab is fully determined
by the game state: `number` is whatever is available, `num_to_pattern_line`
is min(available, free slots), and the rest spills to the floor. So every
legal move maps to exactly one (source, colour, destination) triple, and we
can use a flat fixed-size action index for the policy head.

Layout (4-player => 9 factories + centre = 10 sources):

    source  : 0..8 = factory index, 9 = centre
    colour  : 0..4 = Tile value
    dest    : 0..4 = pattern line index, 5 = floor only

    index = (source * NUM_COLORS + colour) * NUM_DESTS + dest
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.normpath(os.path.join(_HERE, "..", "framework")),
    os.path.normpath(os.path.join(_HERE, "..", "..", "framework")),
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from utils import Move  # noqa: E402

from az import NUM_COLORS, NUM_FACTORIES  # noqa: E402

CENTRE_SOURCE = NUM_FACTORIES        # 9
NUM_SOURCES = NUM_FACTORIES + 1      # 10
NUM_DESTS = 6                        # 5 pattern lines + floor
FLOOR_DEST = 5
ACTION_SIZE = NUM_SOURCES * NUM_COLORS * NUM_DESTS  # 300


def move_to_action(move):
    """Map a framework move tuple to its flat action index."""
    mt, fid, tg = move
    source = fid if mt == Move.TAKE_FROM_FACTORY else CENTRE_SOURCE
    colour = int(tg.tile_type)
    dest = tg.pattern_line_dest if tg.num_to_pattern_line > 0 else FLOOR_DEST
    return (source * NUM_COLORS + colour) * NUM_DESTS + dest


def legal_mask(legal_moves):
    """Return (mask, idx_to_move) for a list of legal framework moves.

    mask: float32 array of length ACTION_SIZE, 1.0 where an action is legal.
    idx_to_move: dict action_index -> the framework move to apply.
    """
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    idx_to_move = {}
    for m in legal_moves:
        idx = move_to_action(m)
        mask[idx] = 1.0
        idx_to_move[idx] = m
    return mask, idx_to_move
