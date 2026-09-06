"""Encode an Azul game state into a fixed-length feature vector (v2).

Canonical: players listed starting from the player to move ("me first").

v2 adds features for the strategic factors a strong human (Martin) flagged
as missing in v1:
  * first-player token / turn order — who secures going first next round
  * tile availability — how many of each colour remain in the bag, and how
    many tiles are still on the board this round

Feature layout (4 players):

  Per player (canonical order), 62 each:
      25 wall grid | 5 pattern-line fill | 25 pattern-line colour
       1 floor fill | 1 score | 5 number_of[colour]
  Shared:
      45 factories (9 x 5 colour counts /4)
       5 centre colour counts /10
       1 first-player token still available
       4 who goes first NEXT round (canonical one-hot of next_first_player)
       5 bag availability per colour ((bag+used)/20)
       1 tiles still on the board this round (factories+centre, normalised)

  Total = 4*62 + 61 = 309
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.normpath(os.path.join(_HERE, "..", "framework")),
    os.path.normpath(os.path.join(_HERE, "..", "..", "framework")),
):
    if os.path.isdir(_cand):
        if _cand in sys.path:
            sys.path.remove(_cand)
        sys.path.insert(0, _cand)
        break

from utils import Tile  # noqa: E402

from az import NUM_COLORS, NUM_FACTORIES, NUM_PATTERN_LINES, NUM_PLAYERS  # noqa: E402

ENCODER_VERSION = 2
PER_PLAYER = 62
# 45 factories + 5 centre + 1 token-available + 4 next-first + 5 bag + 1 board-tiles
SHARED = NUM_FACTORIES * NUM_COLORS + NUM_COLORS + 1 + NUM_PLAYERS + NUM_COLORS + 1
FEATURE_SIZE = NUM_PLAYERS * PER_PLAYER + SHARED  # 309


def _encode_player(p, out, off):
    grid = np.asarray(p.grid_state, dtype=np.float32).reshape(-1)
    out[off:off + 25] = grid
    off += 25
    for i in range(NUM_PATTERN_LINES):
        out[off + i] = p.lines_number[i] / (i + 1)
    off += NUM_PATTERN_LINES
    for i in range(NUM_PATTERN_LINES):
        t = p.lines_tile[i]
        if t != -1:
            out[off + i * NUM_COLORS + int(t)] = 1.0
    off += NUM_PATTERN_LINES * NUM_COLORS
    out[off] = sum(p.floor) / len(p.floor)
    off += 1
    out[off] = p.score / 100.0
    off += 1
    for t in Tile:
        out[off + int(t)] = p.number_of[t] / 5.0
    off += NUM_COLORS
    return off


def encode(game_state, current_player):
    """Return a float32 v2 feature vector from `current_player`'s perspective."""
    out = np.zeros(FEATURE_SIZE, dtype=np.float32)
    n = len(game_state.players)
    off = 0
    for k in range(NUM_PLAYERS):
        seat = (current_player + k) % n
        off = _encode_player(game_state.players[seat], out, off)
    # factories
    for f in range(NUM_FACTORIES):
        fd = game_state.factories[f]
        for t in Tile:
            out[off + int(t)] = fd.tiles[t] / 4.0
        off += NUM_COLORS
    # centre
    for t in Tile:
        out[off + int(t)] = game_state.centre_pool.tiles[t] / 10.0
    off += NUM_COLORS
    # first-player token still available
    out[off] = 0.0 if game_state.first_player_taken else 1.0
    off += 1
    # who goes first NEXT round (the seat that took the token), canonical one-hot
    nfp = game_state.next_first_player
    if nfp is not None and nfp >= 0:
        out[off + ((nfp - current_player) % n)] = 1.0
    off += NUM_PLAYERS
    # bag availability per colour (draw bag + used bag, which gets reshuffled in)
    for t in Tile:
        c = game_state.bag.count(t) + game_state.bag_used.count(t)
        out[off + int(t)] = c / 20.0
    off += NUM_COLORS
    # tiles still on the board this round (factories + centre)
    board = sum(fd.total for fd in game_state.factories) + game_state.centre_pool.total
    out[off] = board / 40.0
    off += 1
    assert off == FEATURE_SIZE, (off, FEATURE_SIZE)
    return out
