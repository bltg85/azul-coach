"""Thin wrapper around framework GameState that exposes a turn-by-turn API.

The framework's GameRunner is structured as a long-running loop, which is
awkward for tree search. AzulSim moves one ply at a time and exposes:

    legal_moves(), apply(move), clone(), is_terminal, current_player,
    scores()  (includes end-of-game bonuses once terminal).
"""
import copy
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.normpath(os.path.join(_HERE, "..", "framework")),
    os.path.normpath(os.path.join(_HERE, "..", "..", "framework")),
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from model import GameState, PlayerState, TileDisplay  # noqa: E402
from utils import PlayerTrace  # noqa: E402


class _BoundedList(list):
    """List whose append() is a no-op. Lets framework code call
    moves[-1].append(move) without us paying memory for history we never
    read."""
    __slots__ = ()

    def append(self, _x):
        return


_NOOP_LIST = _BoundedList()


class _TraceStub:
    """Drop-in replacement for PlayerTrace during simulation.

    ExecuteMove appends to moves[-1] and writes round_scores[-1]. We need
    those slots to exist but their contents are discarded. To avoid any
    unbounded growth across deep simulations, moves[-1] is a singleton
    no-op list and StartRound is itself a no-op.
    """

    __slots__ = ("moves", "round_scores", "bonuses", "id")

    def __init__(self):
        self.moves = [_NOOP_LIST]
        self.round_scores = [0]
        self.bonuses = 0
        self.id = -1

    def StartRound(self):
        # No-op: we never read this back.
        pass


def _fast_clone_td(td):
    new = TileDisplay.__new__(TileDisplay)
    new.tiles = dict(td.tiles)
    new.total = td.total
    return new


def _fast_clone_player(p):
    new = PlayerState.__new__(PlayerState)
    new.id = p.id
    new.score = p.score
    new.GRID_SIZE = p.GRID_SIZE
    new.lines_number = p.lines_number[:]
    new.lines_tile = p.lines_tile[:]
    # grid_scheme never mutates after construction — share it
    new.grid_scheme = p.grid_scheme
    new.grid_state = p.grid_state.copy()
    new.floor = p.floor[:]
    new.floor_tiles = p.floor_tiles[:]
    new.number_of = dict(p.number_of)
    # Replace heavy PlayerTrace with stub; we never read it back here
    new.player_trace = _TraceStub()
    return new


def _fast_clone_gs(gs):
    new = GameState.__new__(GameState)
    new.players = [_fast_clone_player(p) for p in gs.players]
    new.bag = gs.bag[:]
    new.bag_used = gs.bag_used[:]
    new.factories = [_fast_clone_td(f) for f in gs.factories]
    new.centre_pool = _fast_clone_td(gs.centre_pool)
    new.first_player_taken = gs.first_player_taken
    new.first_player = gs.first_player
    new.next_first_player = gs.next_first_player
    return new


class AzulSim:
    __slots__ = ("gs", "cur", "terminal")

    def __init__(self, game_state, current_player):
        self.gs = game_state
        self.cur = current_player
        self.terminal = False

    def current_player(self):
        return self.cur

    def legal_moves(self):
        if self.terminal:
            return []
        return self.gs.players[self.cur].GetAvailableMoves(self.gs)

    def apply(self, move):
        n = len(self.gs.players)
        self.gs.ExecuteMove(self.cur, move)
        if self.gs.TilesRemaining():
            self.cur = (self.cur + 1) % n
            return
        # End of round
        self.gs.ExecuteEndOfRound()
        if any(p.GetCompletedRows() > 0 for p in self.gs.players):
            for p in self.gs.players:
                p.EndOfGameScore()
            self.terminal = True
            return
        self.gs.SetupNewRound()
        self.cur = self.gs.first_player

    def clone(self):
        new = AzulSim.__new__(AzulSim)
        new.gs = _fast_clone_gs(self.gs)
        new.cur = self.cur
        new.terminal = self.terminal
        return new

    def scores(self):
        return [p.score for p in self.gs.players]
