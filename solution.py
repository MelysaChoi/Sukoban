#   You may only add standard python imports
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files
from typing import Callable, Union

import os                       # For time functions
import math                     # For infinity

from src import (
    # For search engine implementations
    SearchEngine, SearchNode, SearchStatistics,
    # For Sokoban-specific implementations
    SokobanState,
    sokoban_goal_state,
    UP, DOWN, LEFT, RIGHT,
    # You may further import any constants you may need.
    # See `search_constants.py`
)

# Global timer variable to sync heuristic with search time
GLOBAL_START_TIME = os.times()[0]

# SOKOBAN HEURISTICS
def heur_alternate(state: 'SokobanState') -> float:

    boxes = state.boxes
    storage = state.storage
    obstacles = state.obstacles

    storage_set = set(storage)
    boxes_set = set(boxes)

    storage_rows = set(y for (x, y) in storage)
    storage_cols = set(x for (x, y) in storage)

    # ----------------------------
    # SAFE DEAD SQUARES (cached)
    # ----------------------------
    cache = getattr(heur_alternate, "_cache", None)
    if cache is None:
        cache = {}
        heur_alternate._cache = cache

    key = (state.width, state.height, state.obstacles, state.storage)
    dead_squares = cache.get(key)

    if dead_squares is None:
        from collections import deque

        w, h = state.width, state.height
        obstacles_set = set(obstacles)
        storage_fs = set(storage)

        def blocked(cell):
            x, y = cell
            return (x < 0 or x >= w or y < 0 or y >= h or cell in obstacles_set)

        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # reverse "pull" BFS from storage (ignoring boxes)
        reachable = set()
        q = deque()

        for s in storage_fs:
            if not blocked(s):
                reachable.add(s)
                q.append(s)

        while q:
            cx, cy = q.popleft()
            for dx, dy in dirs:
                prev = (cx + dx, cy + dy)            # where the box could have been
                behind = (cx + 2 * dx, cy + 2 * dy)  # where robot would stand to push
                if blocked(prev) or blocked(behind):
                    continue
                if prev not in reachable:
                    reachable.add(prev)
                    q.append(prev)

        dead_squares = set()
        for x in range(w):
            for y in range(h):
                c = (x, y)
                if not blocked(c) and c not in reachable:
                    dead_squares.add(c)

        cache[key] = dead_squares

    # --- DEADLOCK DETECTION ---
    for (x, y) in boxes:

        if (x, y) in storage_set:
            continue

        # safe prune: box is on a dead square (can never reach storage)
        if (x, y) in dead_squares:
            return float('inf')

        left  = (x - 1, y)
        right = (x + 1, y)
        up    = (x, y - 1)
        down  = (x, y + 1)

        left_blocked  = left in obstacles or left in boxes_set
        right_blocked = right in obstacles or right in boxes_set
        up_blocked    = up in obstacles or up in boxes_set
        down_blocked  = down in obstacles or down in boxes_set

        # Check for corners
        if (left_blocked and up_blocked) or \
           (left_blocked and down_blocked) or \
           (right_blocked and up_blocked) or \
           (right_blocked and down_blocked):
            return float('inf')

        # check for corridors
        if left_blocked and right_blocked:
            if x not in storage_cols:
                return float('inf')

        if up_blocked and down_blocked:
            if y not in storage_rows:
                return float('inf')

        # check if the box is against a wall and no storage unit along that wall
        if (left_blocked or right_blocked) and x not in storage_cols:
            if left_blocked and right_blocked:
                return float('inf')

        if (up_blocked or down_blocked) and y not in storage_rows:
            if up_blocked and down_blocked:
                return float('inf')

        # adjacent boxes
        # horizontal
        if (x + 1, y) in boxes_set:
            if (x + 1, y) not in storage_set:
                up_pair_blocked = ((x, y - 1) in obstacles or (x, y - 1) in boxes_set) and \
                                  ((x + 1, y - 1) in obstacles or (x + 1, y - 1) in boxes_set)

                down_pair_blocked = ((x, y + 1) in obstacles or (x, y + 1) in boxes_set) and \
                                    ((x + 1, y + 1) in obstacles or (x + 1, y + 1) in boxes_set)

                if up_pair_blocked and down_pair_blocked:
                    return float('inf')

        # vertical
        if (x, y + 1) in boxes_set:
            if (x, y + 1) not in storage_set:
                left_pair_blocked = ((x - 1, y) in obstacles or (x - 1, y) in boxes_set) and \
                                    ((x - 1, y + 1) in obstacles or (x - 1, y + 1) in boxes_set)

                right_pair_blocked = ((x + 1, y) in obstacles or (x + 1, y) in boxes_set) and \
                                     ((x + 1, y + 1) in obstacles or (x + 1, y + 1) in boxes_set)

                if left_pair_blocked and right_pair_blocked:
                    return float('inf')

        # 2x2 block deadlock
        square = [
            (x, y),
            (x + 1, y),
            (x, y + 1),
            (x + 1, y + 1)
        ]

        count_blockers = 0
        contains_storage = False

        for pos in square:
            if pos in storage_set:
                contains_storage = True
            if pos in boxes_set or pos in obstacles:
                count_blockers += 1

        if count_blockers >= 3 and not contains_storage:
            return float('inf')

    # --- GREEDY UNIQUE MATCHING ---
    # (tighten: only match unstored boxes)
    unstored_boxes = [b for b in boxes if b not in storage_set]

    # (tighten: don't match to a storage already occupied by a stored box)
    remaining_storage = [s for s in storage if s not in boxes_set]
    if not remaining_storage:
        remaining_storage = list(storage)

    remaining_boxes = list(unstored_boxes)

    total = 0

    while remaining_boxes:

        best_dist = float('inf')
        best_pair = None

        for b in remaining_boxes:
            for s in remaining_storage:
                dist = abs(b[0] - s[0]) + abs(b[1] - s[1])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (b, s)

        if best_pair is None:
            break

        total += best_dist
        remaining_boxes.remove(best_pair[0])
        remaining_storage.remove(best_pair[1])

    # small robot-to-box penalty (global min; GBFS-friendly)
    if unstored_boxes and state.robots:
        min_robot_distance = math.inf
        for (rx, ry) in state.robots:
            for (bx, by) in unstored_boxes:
                d = abs(bx - rx) + abs(by - ry)
                if d < min_robot_distance:
                    min_robot_distance = d
        total += 0.2 * min_robot_distance

    return total

def heur_zero(state: 'SokobanState') -> float:
    return 0

def heur_manhattan_distance(state: 'SokobanState') -> float:
    cost = 0

    for box in state.boxes: #iterate through all boxes
        if box not in state.storage: #check only boxes not in storage
            min_dist = math.inf

            for storage_spot in state.storage:
                dist = abs(box[0] - storage_spot[0]) + abs(box[1]-storage_spot[1])
                min_dist = min(min_dist, dist)

            cost += min_dist
    
    return cost

def fval_function(node: 'SearchNode', weight: float) -> float:
    return node.gval + (weight * node.hval)


def weighted_astar(
        initial_state: 'SokobanState',
        heur_fn: Callable,
        weight: float,
        timebound: int) -> tuple[Union['SokobanState', bool], 'SearchStatistics']:


    global GLOBAL_START_TIME
    GLOBAL_START_TIME = os.times()[0]

    search_engine = SearchEngine(strategy = 'custom', cc='full')

    wrapped_fval = lambda node: fval_function(node,weight)

    search_engine.init_search(initial_state,sokoban_goal_state, heur_fn, wrapped_fval)

    result = search_engine.search(timebound = timebound)

    return result

def iterative_astar( # uses f(n)
        initial_state: 'SokobanState',
        heur_fn: Callable,
        weight: float = 1,
        timebound: int = 5) -> tuple[Union['SokobanState', bool], 'SearchStatistics']:

    global GLOBAL_START_TIME
    GLOBAL_START_TIME = os.times()[0]
    
    start_time = GLOBAL_START_TIME

    best_sol = None
    best_cost = math.inf
    best_stats = None

    # Start with pure gbfs, then refine with weighted A*
    
    # gbfs
    duration = os.times()[0] - start_time
    remain_t = timebound - duration
    
    if remain_t > 0:
        search_engine = SearchEngine('best_first', 'full')
        search_engine.init_search(
            init_state=initial_state,
            goal_fn=sokoban_goal_state,
            heur_fn=heur_fn
        )
        solution, stats = search_engine.search(timebound=remain_t)
        
        best_stats = stats
        
        if solution:
            best_sol = solution
            best_cost = solution.gval
    
    # weighted A*
    weight_set = [10, 5, 2, 1]
    
    if weight in weight_set:
        start = weight_set.index(weight)
        weight_set = weight_set[start:]
    elif weight < 1:
        weight_set = [1]

    for cur_w in weight_set:

        if (os.times()[0] - start_time) > 1.99:
            break

        duration = os.times()[0] - start_time
        remain_t = timebound - duration

        if remain_t <= 0:
            break

        costbound = None
        if best_sol:
            costbound = (math.inf, math.inf, best_cost)

        search_engine = SearchEngine('custom', 'full')
        wrapped_fval = lambda node: fval_function(node, cur_w)
        search_engine.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval)
        
        solution, stats = search_engine.search(timebound=remain_t, costbound=costbound)

        if best_stats is None:
            best_stats = stats
        else:
            best_stats = SearchStatistics(
                best_stats.states_expanded + stats.states_expanded,
                best_stats.states_generated + stats.states_generated,
                best_stats.states_pruned_cycles + stats.states_pruned_cycles,
                best_stats.states_pruned_cost + stats.states_pruned_cost,
                os.times()[0] - start_time
            )

        if solution and solution.gval < best_cost:
            best_sol = solution
            best_cost = solution.gval

        if cur_w == 1:
            break

    if best_sol:
        return best_sol, best_stats
    else:
        return False, best_stats
    
def iterative_gbfs( # uses h(n)
        initial_state: 'SokobanState',
        heur_fn: Callable,
        timebound: int = 5) -> tuple[Union['SokobanState', bool], 'SearchStatistics']:

    global GLOBAL_START_TIME
    GLOBAL_START_TIME = os.times()[0]

    start_time = GLOBAL_START_TIME

    best_sol = None
    best_cost = math.inf
    best_stats = None

    while True:
        if (os.times()[0] - start_time) > 1.99:
            break

        duration = os.times()[0] - start_time
        remaining_t = timebound - duration

        if remaining_t <= 0:
            break

        search_engine = SearchEngine('best_first', 'full')

        search_engine.init_search(init_state = initial_state, goal_fn = sokoban_goal_state, heur_fn=heur_fn)

        costbound = None
        if best_sol:
            costbound = (best_cost, math.inf, math.inf)
        
        solution, stats = search_engine.search(remaining_t, costbound)

        if best_stats == None:
            best_stats = stats
        else:   
            best_stats = SearchStatistics(best_stats.states_expanded + stats.states_expanded, best_stats.states_generated +stats.states_generated,
                                          best_stats.states_pruned_cycles + stats.states_pruned_cycles, best_stats.states_pruned_cost + stats.states_pruned_cost,
                                          os.times()[0] - start_time)
        
        if solution and solution.gval < best_cost:
            best_sol = solution
            best_cost = solution.gval
        else:
            break

    if best_sol:
        return best_sol, best_stats
    else:
        return False, best_stats