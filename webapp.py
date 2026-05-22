"""Web UI for playing Azul against 1-3 bots.

Multi-user, session-keyed game state. Each browser session gets its own
game stored in memory. Stale games (>1 hour inactive) are evicted to
keep memory bounded.

Local dev:
    python webapp.py

Production (gunicorn behind Render/Fly/etc.):
    gunicorn -w 1 -t 120 webapp:app
    (1 worker keeps the in-memory GAMES dict consistent; -t 120 lets the
     coach endpoint finish slow MCTS runs without being killed.)

Env vars:
    SECRET_KEY        Flask session signing key (required in prod).
    MAX_COACH_ITER    Upper bound on MCTS coach iterations (default 2000;
                      set lower on shared-CPU hosts).
    MAX_BOT_MCTS_ITER Upper bound on MCTS iter for opponent bots.
    MAX_SESSIONS      Cap on concurrent live games (default 50).
    DISABLE_LOG_SAVE  Set to "1" to skip writing games/*.json (cloud disks
                      are usually ephemeral so the files vanish anyway).
"""
import datetime
import json
import os
import secrets
import sys
import threading
import time
from copy import deepcopy

from flask import Flask, redirect, render_template, request, session, url_for

import _framework_path  # noqa: F401
from model import GameState, PlayerState  # noqa: E402
from utils import Move, MoveToString, Tile  # noqa: E402

from agents.heuristic import HeuristicPlayer  # noqa: E402
from agents.mcts import MCTSPlayer  # noqa: E402
from agents.sim import AzulSim  # noqa: E402


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

MAX_COACH_ITER = int(os.environ.get("MAX_COACH_ITER", "2000"))
MAX_BOT_MCTS_ITER = int(os.environ.get("MAX_BOT_MCTS_ITER", "2000"))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "50"))
SESSION_TTL_S = int(os.environ.get("SESSION_TTL_S", "3600"))
DISABLE_LOG_SAVE = os.environ.get("DISABLE_LOG_SAVE", "0") == "1"

GAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
if not DISABLE_LOG_SAVE:
    os.makedirs(GAMES_DIR, exist_ok=True)


TILE_NAMES = {
    Tile.BLUE: "B", Tile.YELLOW: "Y", Tile.RED: "R",
    Tile.BLACK: "K", Tile.WHITE: "W",
}
WALL_LAYOUT = [
    [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE],
    [Tile.WHITE, Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK],
    [Tile.BLACK, Tile.WHITE, Tile.BLUE, Tile.YELLOW, Tile.RED],
    [Tile.RED, Tile.BLACK, Tile.WHITE, Tile.BLUE, Tile.YELLOW],
    [Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE, Tile.BLUE],
]


# ---------------------------------------------------------------------------
# Session-keyed game registry.
# ---------------------------------------------------------------------------

GAMES = {}              # sid -> game dict
GAMES_LOCK = threading.Lock()


def _empty_game():
    return {
        "sim": None,
        "bots": [],
        "bot_specs": [],
        "user_seat": 0,
        "pending": None,
        "message": "Welcome — start a new game.",
        "coach": None,
        "coach_busy": False,
        "coach_history": [],
        "log": [],
        "seed": None,
        "saved_path": None,
        "started_at": None,
        "history": [],
        "last_seen": time.time(),
        "lock": threading.Lock(),
    }


def _evict_stale_locked():
    """Remove games whose owners haven't poked the server in SESSION_TTL_S.
    Called under GAMES_LOCK."""
    cutoff = time.time() - SESSION_TTL_S
    stale = [sid for sid, g in GAMES.items() if g["last_seen"] < cutoff]
    for sid in stale:
        del GAMES[sid]
    # Hard cap: evict oldest if we still exceed the limit.
    if len(GAMES) > MAX_SESSIONS:
        by_age = sorted(GAMES.items(), key=lambda kv: kv[1]["last_seen"])
        for sid, _ in by_age[: len(GAMES) - MAX_SESSIONS]:
            del GAMES[sid]


def get_game():
    """Return (or create) the current visitor's game state."""
    sid = session.get("sid")
    if not sid:
        sid = secrets.token_urlsafe(16)
        session["sid"] = sid
        session.permanent = True
    with GAMES_LOCK:
        game = GAMES.get(sid)
        if game is None:
            _evict_stale_locked()
            game = _empty_game()
            GAMES[sid] = game
        game["last_seen"] = time.time()
    return sid, game


# ---------------------------------------------------------------------------
# Game helpers (operate on a passed-in game dict).
# ---------------------------------------------------------------------------

def make_bot(spec, pid):
    spec = spec.strip().lower()
    if spec == "heuristic":
        return HeuristicPlayer(pid)
    if spec.startswith("mcts:"):
        iters = int(spec.split(":", 1)[1])
        iters = max(1, min(iters, MAX_BOT_MCTS_ITER))
        return MCTSPlayer(pid, iterations=iters)
    raise ValueError(f"unknown bot spec {spec!r}")


def snapshot_for_undo(game):
    sim = game["sim"]
    if sim is None or sim.terminal:
        return
    game["history"].append({
        "sim": deepcopy(sim),
        "log": list(game["log"]),
        "coach_history": list(game["coach_history"]),
    })


def save_game_log(game, sid):
    if DISABLE_LOG_SAVE:
        return None
    sim = game["sim"]
    if sim is None or not sim.terminal or game["saved_path"]:
        return None
    now = datetime.datetime.now()
    scores = sim.scores()
    winners = [i for i, s in enumerate(scores) if s == max(scores)]
    fname = f"game_{now.strftime('%Y%m%d_%H%M%S')}_seed{game['seed']}_{sid[:8]}.json"
    path = os.path.join(GAMES_DIR, fname)
    payload = {
        "schema_version": 1,
        "session_id_prefix": sid[:8],
        "started_at": game["started_at"],
        "ended_at": now.isoformat(timespec="seconds"),
        "seed": game["seed"],
        "user_seat": game["user_seat"],
        "seats": [
            {"seat": 0, "name": "YOU"},
            *[{"seat": i + 1, "name": spec} for i, spec in enumerate(game["bot_specs"])],
        ],
        "final_scores": scores,
        "winner_seats": winners,
        "user_won": game["user_seat"] in winners and len(winners) == 1,
        "user_tied_for_win": game["user_seat"] in winners and len(winners) > 1,
        "num_moves": len(game["log"]),
        "move_log": game["log"],
        "coach_history": game["coach_history"],
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        game["saved_path"] = path
        return path
    except OSError:
        # Read-only fs on cloud — silently skip
        return None


def advance_bots(game):
    sim = game["sim"]
    if sim is None or sim.terminal:
        return
    bots_by_seat = {b.id: b for b in game["bots"]}
    while not sim.terminal and sim.cur != game["user_seat"]:
        seat = sim.cur
        bot = bots_by_seat[seat]
        moves = sim.legal_moves()
        if not moves:
            break
        gs_copy = deepcopy(sim.gs)
        moves_copy = deepcopy(moves)
        chosen = bot.SelectMove(moves_copy, gs_copy)
        game["log"].append(
            f"P{seat} ({bot.__class__.__name__}): "
            + MoveToString(seat, chosen).replace("\n", " | ")
        )
        sim.apply(chosen)
        game["coach"] = None


def build_view(game):
    sim = game["sim"]
    if sim is None:
        return None
    gs = sim.gs
    factories = []
    for i, fd in enumerate(gs.factories):
        groups = []
        for tile in Tile:
            n = fd.tiles[tile]
            if n > 0:
                groups.append({"tile": TILE_NAMES[tile], "tile_id": int(tile), "count": n})
        factories.append({"id": i, "groups": groups, "empty": fd.total == 0})

    centre_groups = []
    for tile in Tile:
        n = gs.centre_pool.tiles[tile]
        if n > 0:
            centre_groups.append({"tile": TILE_NAMES[tile], "tile_id": int(tile), "count": n})
    centre = {
        "groups": centre_groups,
        "first_player_token": not gs.first_player_taken,
        "empty": gs.centre_pool.total == 0 and gs.first_player_taken,
    }

    players_view = []
    for p in gs.players:
        lines = []
        for i in range(5):
            tile = p.lines_tile[i]
            lines.append({
                "size": i + 1,
                "count": p.lines_number[i],
                "tile": TILE_NAMES[Tile(tile)] if tile != -1 else None,
            })
        wall = []
        for r in range(5):
            row = []
            for c in range(5):
                t = WALL_LAYOUT[r][c]
                row.append({"tile": TILE_NAMES[t], "filled": bool(p.grid_state[r][c])})
            wall.append(row)
        floor = []
        for i, slot in enumerate(p.floor):
            penalty = PlayerState.FLOOR_SCORES[i]
            tile_letter = None
            if slot == 1:
                idx_in_tiles = sum(1 for j in range(i) if p.floor[j] == 1) - (
                    1 if not gs.first_player_taken else 0
                )
                if 0 <= idx_in_tiles < len(p.floor_tiles):
                    tile_letter = TILE_NAMES[Tile(p.floor_tiles[idx_in_tiles])]
                else:
                    tile_letter = "1"
            floor.append({"filled": slot == 1, "tile": tile_letter, "penalty": penalty})
        players_view.append({
            "id": p.id,
            "score": p.score,
            "lines": lines,
            "wall": wall,
            "floor": floor,
            "is_user": p.id == game["user_seat"],
            "is_current": p.id == sim.cur and not sim.terminal,
            "name": "YOU" if p.id == game["user_seat"] else f"Bot {p.id}",
        })

    destinations = []
    valid_dest_map = {}
    pending_view = None
    if game["pending"] is not None and sim.cur == game["user_seat"] and not sim.terminal:
        pending = game["pending"]
        tile_id = pending["tile"]
        for move in sim.legal_moves():
            mt, fid, tg = move
            if pending["source"] == "factory" and mt != Move.TAKE_FROM_FACTORY:
                continue
            if pending["source"] == "centre" and mt != Move.TAKE_FROM_CENTRE:
                continue
            if pending["source"] == "factory" and fid != pending["id"]:
                continue
            if int(tg.tile_type) != tile_id:
                continue
            if tg.num_to_pattern_line > 0:
                destinations.append({
                    "label": f"Pattern line {tg.pattern_line_dest + 1}",
                    "dest": tg.pattern_line_dest,
                    "to_line": tg.num_to_pattern_line,
                    "to_floor": tg.num_to_floor_line,
                })
                valid_dest_map[tg.pattern_line_dest] = {
                    "to_line": tg.num_to_pattern_line,
                    "to_floor": tg.num_to_floor_line,
                }
            else:
                destinations.append({
                    "label": "Floor only", "dest": -1,
                    "to_line": 0, "to_floor": tg.num_to_floor_line,
                })
                valid_dest_map[-1] = {"to_line": 0, "to_floor": tg.num_to_floor_line}
        source_label = (f"Factory {pending['id'] + 1}" if pending["source"] == "factory"
                        else "Centre")
        pending_view = {
            "source": pending["source"], "source_id": pending["id"],
            "source_label": source_label,
            "tile": TILE_NAMES[Tile(tile_id)], "tile_id": tile_id,
        }

    return {
        "factories": factories,
        "centre": centre,
        "players": players_view,
        "pending": pending_view,
        "destinations": destinations,
        "valid_dests": valid_dest_map,
        "message": game["message"],
        "log": game["log"][-12:],
        "current_player": sim.cur,
        "user_seat": game["user_seat"],
        "is_user_turn": sim.cur == game["user_seat"] and not sim.terminal,
        "game_over": sim.terminal,
        "scores": sim.scores() if sim.terminal else None,
        "coach": game["coach"],
        "coach_busy": game["coach_busy"],
        "undo_count": len(game["history"]),
        "max_coach_iter": MAX_COACH_ITER,
    }


# ---------------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    _, game = get_game()
    with game["lock"]:
        view = build_view(game)
    return render_template("index.html", view=view, game=game)


@app.route("/new", methods=["POST"])
def new_game():
    bots_spec = request.form.get("bots", "heuristic,heuristic,heuristic").strip()
    seed_str = request.form.get("seed", "").strip()
    seed = int(seed_str) if seed_str else int(time.time())
    specs = [s.strip() for s in bots_spec.split(",") if s.strip()]
    if not (1 <= len(specs) <= 3):
        return "Need 1-3 bot opponents", 400

    n_players = len(specs) + 1
    _, game = get_game()
    with game["lock"]:
        import random
        random.seed(seed)
        gs = GameState(n_players)
        for p in gs.players:
            p.player_trace.StartRound()
        game["sim"] = AzulSim(gs, gs.first_player)
        game["user_seat"] = 0
        game["bots"] = [make_bot(spec, i + 1) for i, spec in enumerate(specs)]
        game["bot_specs"] = specs
        game["pending"] = None
        game["message"] = f"New game (seed {seed}) — you are player 0. Opponents: {specs}"
        game["log"] = []
        game["coach"] = None
        game["coach_history"] = []
        game["seed"] = seed
        game["saved_path"] = None
        game["history"] = []
        game["started_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        advance_bots(game)
    return redirect(url_for("index"))


@app.route("/select", methods=["POST"])
def select_source():
    source = request.form["source"]
    source_id = int(request.form.get("id", -1))
    tile_id = int(request.form["tile"])
    _, game = get_game()
    with game["lock"]:
        if game["sim"] is None or game["sim"].terminal:
            return redirect(url_for("index"))
        if game["sim"].cur != game["user_seat"]:
            game["message"] = "Not your turn."
            return redirect(url_for("index"))
        game["pending"] = {"source": source, "id": source_id, "tile": tile_id}
        game["message"] = "Pick a destination."
    return redirect(url_for("index"))


@app.route("/cancel", methods=["POST"])
def cancel_selection():
    _, game = get_game()
    with game["lock"]:
        game["pending"] = None
        game["message"] = "Selection cancelled."
    return redirect(url_for("index"))


@app.route("/place", methods=["POST"])
def place():
    dest = int(request.form["dest"])
    sid, game = get_game()
    with game["lock"]:
        sim = game["sim"]
        if sim is None or sim.terminal or sim.cur != game["user_seat"]:
            return redirect(url_for("index"))
        pending = game["pending"]
        if pending is None:
            return redirect(url_for("index"))
        chosen = None
        for move in sim.legal_moves():
            mt, fid, tg = move
            if pending["source"] == "factory" and mt != Move.TAKE_FROM_FACTORY:
                continue
            if pending["source"] == "centre" and mt != Move.TAKE_FROM_CENTRE:
                continue
            if pending["source"] == "factory" and fid != pending["id"]:
                continue
            if int(tg.tile_type) != pending["tile"]:
                continue
            if dest == -1:
                if tg.num_to_pattern_line == 0:
                    chosen = move
                    break
            else:
                if tg.pattern_line_dest == dest and tg.num_to_pattern_line > 0:
                    chosen = move
                    break
        if chosen is None:
            game["message"] = "Invalid placement."
            return redirect(url_for("index"))
        snapshot_for_undo(game)
        game["log"].append(
            f"P{game['user_seat']} (YOU): "
            + MoveToString(game["user_seat"], chosen).replace("\n", " | ")
        )
        sim.apply(chosen)
        game["pending"] = None
        game["coach"] = None
        game["message"] = "Move placed. Bots playing..."
        advance_bots(game)
        if sim.terminal:
            saved = save_game_log(game, sid)
            if saved:
                game["message"] = f"Game over. Log saved to {os.path.basename(saved)}"
            else:
                game["message"] = "Game over."
    return redirect(url_for("index"))


@app.route("/undo", methods=["POST"])
def undo():
    _, game = get_game()
    with game["lock"]:
        if not game["history"]:
            game["message"] = "Nothing to undo."
            return redirect(url_for("index"))
        if game["saved_path"] and os.path.exists(game["saved_path"]):
            try:
                os.remove(game["saved_path"])
            except OSError:
                pass
        snap = game["history"].pop()
        game["sim"] = snap["sim"]
        game["log"] = snap["log"]
        game["coach_history"] = snap["coach_history"]
        game["pending"] = None
        game["coach"] = None
        game["saved_path"] = None
        remaining = len(game["history"])
        game["message"] = (f"Undo. {remaining} earlier undo(s) available."
                           if remaining else "Undo. Back at game start.")
    return redirect(url_for("index"))


@app.route("/coach", methods=["POST"])
def get_coach():
    iters = max(1, min(int(request.form.get("iter", 1000)), MAX_COACH_ITER))
    _, game = get_game()
    with game["lock"]:
        sim = game["sim"]
        if sim is None or sim.terminal or sim.cur != game["user_seat"]:
            game["message"] = "Coach only available on your turn."
            return redirect(url_for("index"))
        moves = sim.legal_moves()
        if not moves:
            return redirect(url_for("index"))
        gs_copy = deepcopy(sim.gs)
        moves_copy = deepcopy(moves)
        user_seat = game["user_seat"]
        game["coach_busy"] = True
    # Heavy MCTS work outside the lock so other routes for THIS session
    # aren't blocked. (Different sessions are independent anyway.)
    t0 = time.time()
    coach_bot = MCTSPlayer(user_seat, iterations=iters)
    coach_bot.SelectMove(moves_copy, gs_copy)
    elapsed = time.time() - t0
    with game["lock"]:
        stats = coach_bot.last_stats
        top = []
        for c in stats["candidates"][:5]:
            mt, fid, tg = c["move"]
            source = (f"Factory {fid + 1}" if mt == Move.TAKE_FROM_FACTORY else "Centre")
            dest = (f"Pattern line {tg.pattern_line_dest + 1}"
                    if tg.num_to_pattern_line > 0 else "Floor only")
            top.append({
                "visits": c["visits"], "value": c["avg_value"],
                "source": source, "tile": TILE_NAMES[Tile(int(tg.tile_type))],
                "count": tg.number, "to_line": tg.num_to_pattern_line,
                "to_floor": tg.num_to_floor_line, "dest": dest,
            })
        game["coach"] = {
            "iterations": stats["iterations"],
            "elapsed_s": elapsed,
            "candidates": top,
        }
        game["coach_busy"] = False
        game["message"] = f"Coach: {iters} iter in {elapsed:.1f}s. Top move highlighted."
    return redirect(url_for("index"))


@app.route("/healthz")
def healthz():
    return {"ok": True, "active_sessions": len(GAMES)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Open http://127.0.0.1:{port} in your browser")
    app.run(debug=False, host="0.0.0.0", port=port)
