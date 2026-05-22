"""Level-1 heuristic Azul player.

Evaluates each legal move by simulating its effect on the player's own
state, then scoring with three terms:

  1. Round-end delta  - exact, from PlayerState.ScoreRound() (floor
     penalty plus immediate placement points if a pattern line fills).
  2. Partial progress - soft credit for tiles sitting in partially-filled
     pattern lines, weighted by how close to completion they are and the
     estimated placement value when they eventually score.
  3. End-game potential - row / column / colour-set completion, cubed
     so only nearly-finished lines contribute meaningfully.

The bot ignores opponents entirely. That is fine for level 1; level 2
(MCTS) will model them. We do model the first-player-token penalty
because it shows up as a floor tile in our own scoring.
"""
import copy
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.normpath(os.path.join(_HERE, "..", "framework")),       # vendored
    os.path.normpath(os.path.join(_HERE, "..", "..", "framework")), # sibling
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from model import Player, PlayerState  # noqa: E402
from utils import Move, Tile  # noqa: E402


DEFAULT_WEIGHTS = {
    "partial_progress": 0.7,
    "row_progress": 0.4,
    "col_progress": 0.6,
    "set_progress": 0.5,
    # Bonus per extra tile taken from centre beyond the first. Captures
    # "denying opponents access to stacked tiles" which the bot otherwise
    # cannot see (it ignores opponents). Without this, ties break toward
    # factories because factory moves are evaluated first.
    "centre_take": 0.4,
}

GRID_SIZE = PlayerState.GRID_SIZE


class HeuristicPlayer(Player):
    def __init__(self, _id, weights=None, name="heuristic"):
        super().__init__(_id)
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.name = name

    def SelectMove(self, moves, game_state):
        best_score = float("-inf")
        best_move = moves[0]
        for move in moves:
            s = self._evaluate(move, game_state)
            if s > best_score:
                best_score = s
                best_move = move
        return best_move

    def _evaluate(self, move, game_state):
        plr = game_state.players[self.id]
        sim = copy.deepcopy(plr)

        tg = move[2]
        if tg.num_to_pattern_line > 0:
            sim.AddToPatternLine(
                tg.pattern_line_dest, tg.num_to_pattern_line, tg.tile_type
            )
        if tg.num_to_floor_line > 0:
            sim.AddToFloor([tg.tile_type] * tg.num_to_floor_line)
        if move[0] == Move.TAKE_FROM_CENTRE and not game_state.first_player_taken:
            sim.GiveFirstPlayerToken()

        # Snapshot for partial-progress evaluation BEFORE ScoreRound clears lines.
        pattern_snapshot = [
            (sim.lines_number[i], sim.lines_tile[i]) for i in range(GRID_SIZE)
        ]

        before = sim.score
        sim.ScoreRound()
        round_delta = sim.score - before

        # Partial progress on lines still holding tiles after ScoreRound.
        # A line is "still partial" if it didn't fill this round (ScoreRound
        # would have cleared filled lines, leaving lines_number == 0 for those).
        partial = 0.0
        for i, (n_before_score, tile_before_score) in enumerate(pattern_snapshot):
            line_size = i + 1
            # If the line was full it has now been scored and cleared. Skip.
            if n_before_score == line_size or n_before_score == 0:
                continue
            # Only count if it survived ScoreRound (it should, but be safe).
            if sim.lines_number[i] == 0:
                continue
            fullness = n_before_score / line_size
            tile = tile_before_score
            col = int(sim.grid_scheme[i][tile])
            est_value = _estimate_placement_value(sim.grid_state, i, col)
            partial += est_value * (fullness ** 2)

        endgame = _endgame_potential(sim, self.weights)

        centre_bonus = 0.0
        if move[0] == Move.TAKE_FROM_CENTRE and tg.number > 1:
            centre_bonus = self.weights["centre_take"] * (tg.number - 1)

        return (
            round_delta
            + self.weights["partial_progress"] * partial
            + endgame
            + centre_bonus
        )


def _estimate_placement_value(grid_state, row, col):
    """Rough estimate of points if a tile lands at (row, col) on grid_state.

    Counts contiguous neighbours that already exist; mirrors the scoring
    rule in PlayerState.ScoreRound but for a hypothetical placement.
    """
    above = 0
    for j in range(row - 1, -1, -1):
        if grid_state[j][col] == 1:
            above += 1
        else:
            break
    below = 0
    for j in range(row + 1, GRID_SIZE):
        if grid_state[j][col] == 1:
            below += 1
        else:
            break
    left = 0
    for j in range(col - 1, -1, -1):
        if grid_state[row][j] == 1:
            left += 1
        else:
            break
    right = 0
    for j in range(col + 1, GRID_SIZE):
        if grid_state[row][j] == 1:
            right += 1
        else:
            break

    vertical = above + below
    horizontal = left + right
    if vertical == 0 and horizontal == 0:
        return 1.0
    score = 0.0
    if vertical > 0:
        score += 1 + vertical
    if horizontal > 0:
        score += 1 + horizontal
    return score


def _endgame_potential(plr, weights):
    grid = plr.grid_state
    total = 0.0
    for i in range(GRID_SIZE):
        row_count = sum(grid[i][j] for j in range(GRID_SIZE))
        col_count = sum(grid[j][i] for j in range(GRID_SIZE))
        total += weights["row_progress"] * (row_count / GRID_SIZE) ** 3 * PlayerState.ROW_BONUS
        total += weights["col_progress"] * (col_count / GRID_SIZE) ** 3 * PlayerState.COL_BONUS
    for tile in Tile:
        n = plr.number_of[tile]
        total += weights["set_progress"] * (n / GRID_SIZE) ** 3 * PlayerState.SET_BONUS
    return total
