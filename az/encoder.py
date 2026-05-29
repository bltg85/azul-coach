"""Encode an Azul game state into a fixed-length feature vector.

The encoding is *canonical*: players are listed starting from the player to
move, so the net always sees "me first, then opponents in turn order". This
lets a single net evaluate any seat.

Feature layout (4 players):

  Per player (in canonical order), 62 features each:
      25  wall grid (binary, row-major)
       5  pattern-line fill fraction (count / capacity)
      25  pattern-line colour (one-hot per line, zero if empty)
       1  floor fill fraction (tiles / 7)
       1  score (/ 100)
       5  number_of[colour] / 5  (tiles of each colour on the wall)
  Shared:
      45  factories: 9 x 5 colour counts (/ 4)
       5  centre colour counts (/ 10)
       1  first-player token still available (1.0 / 0.0)

  Total = 4*62 + 51 = 299
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

from utils import Tile  # noqa: E402

from az import NUM_COLORS, NUM_FACTORIES, NUM_PATTERN_LINES, NUM_PLAYERS  # noqa: E402

PER_PLAYER = 62
SHARED = NUM_FACTORIES * NUM_COLORS + NUM_COLORS + 1  # 45 + 5 + 1 = 51
FEATURE_SIZE = NUM_PLAYERS * PER_PLAYER + SHARED       # 299


def _encode_player(p, out, off):
    # 25 wall grid (grid_state is a numpy 5x5 of 0/1)
    grid = np.asarray(p.grid_state, dtype=np.float32).reshape(-1)
    out[off:off + 25] = grid
    off += 25
    # 5 pattern-line fill fraction
    for i in range(NUM_PATTERN_LINES):
        out[off + i] = p.lines_number[i] / (i + 1)
    off += NUM_PATTERN_LINES
    # 25 pattern-line colour one-hot
    for i in range(NUM_PATTERN_LINES):
        t = p.lines_tile[i]
        if t != -1:
            out[off + i * NUM_COLORS + int(t)] = 1.0
    off += NUM_PATTERN_LINES * NUM_COLORS
    # 1 floor fill fraction
    out[off] = sum(p.floor) / len(p.floor)
    off += 1
    # 1 score
    out[off] = p.score / 100.0
    off += 1
    # 5 number_of per colour
    for t in Tile:
        out[off + int(t)] = p.number_of[t] / 5.0
    off += NUM_COLORS
    return off


def encode(game_state, current_player):
    """Return a float32 feature vector for `game_state` from the perspective
    of `current_player` (canonical: that player listed first)."""
    out = np.zeros(FEATURE_SIZE, dtype=np.float32)
    n = len(game_state.players)
    off = 0
    for k in range(NUM_PLAYERS):
        seat = (current_player + k) % n
        off = _encode_player(game_state.players[seat], out, off)
    # Shared: factories
    for f in range(NUM_FACTORIES):
        fd = game_state.factories[f]
        for t in Tile:
            out[off + int(t)] = fd.tiles[t] / 4.0
        off += NUM_COLORS
    # Shared: centre
    for t in Tile:
        out[off + int(t)] = game_state.centre_pool.tiles[t] / 10.0
    off += NUM_COLORS
    # Shared: first-player token still available
    out[off] = 0.0 if game_state.first_player_taken else 1.0
    off += 1
    assert off == FEATURE_SIZE, (off, FEATURE_SIZE)
    return out
