from enum import Enum
from grid_universe.gym_env import (
    Observation,
    Action,
    BaseAction,
    step
)
from grid_universe.levels.grid import Level

from typing import Tuple

# Core API
from grid_universe.levels.grid import Level
from grid_universe.state import State
from grid_universe.levels.convert import to_state, from_state

from grid_universe.objectives import (
    exit_objective_fn, default_objective_fn
)
import random
import heapq

class Agent:
    """Grid Universe agent template.

    This class is the single public interface that Coursemology will import and
    interact with when evaluating your submission. You should extend the
    internals (add helper classes / functions in other files if you wish) but
    MUST preserve:

    1. The class name: Agent
    2. The public method: step(self, state: Level | Observation) -> Action

    High‑level lifecycle per environment tick:
        state  --->  step(...)  --->  Action

    The "state" object type depends on the task:
    - Task 1 & 2: a fully structured Level instance.
    - Task 3 & 4: a Observation dictionary whose primary observation is an RGBA image
      plus limited structured metadata in the 'info' sub‑dict. In this case you
      typically perform perception to build (or approximate) an internal
      structured representation before planning.

    Performance constraints:
    - Keep per‑step latency small (single CPU, ~1GB RAM). Avoid O(W*H) scans of
      the full grid every step.
    - Determinism helps reproducibility; seed your own RNG if you add any
      stochastic components.

    You may add __init__ parameters (with defaults) if needed for your own
    development, but the grader will instantiate Agent() with no arguments.
    """

    def __init__(self):
        self.path = None
        self.startingState = None
        self.index = 0
        self.objective_fn = None

        self.portal_pairs = None

        self.valid_use_key_pos_set = None
        self.collectible_pos_set = None

        self.mst_cache = {}
        self.mst_cache_hits = 0
        self.mst_calls = 0

        self.required_key_ghost_speed = None
        self.hasCoins = False
        self.hasSpeed = False

    def step(self, state: Level | Observation) -> Action:
        self.index += 1

        # Plan once, execute sequentially
        if self.path is None:
            self.index = 0

            if isinstance(state, Level):
                self.initialiseLevel(state)
            else:
                self.initialiseObservation(state)

            if not self.startingState:
                print("No starting state!")
                return random.choice(list(Action))
            
            self.path = self.aStarSearch(self.startingState)

        # If no path found, just RNG
        if not self.path:
            print("No path found!!")
            return random.choice(list(Action))

        # Return next action from the path
        return self.path[self.index] if self.index < len(self.path) else Action.WAIT

    def aStarSearch(self, startingState: State):
        frontier = []  # priority queue (min-heap)
        heapq.heappush(frontier, (0, 0, 0, 0, startingState, [])) 
        counter = 0

        x = 0

        visited = set()

        while frontier:
            _, _, _, _, curr_state, actions = heapq.heappop(frontier)

            old_agent_pos = self.get_agent_position(curr_state)
                
            x += 1

            if curr_state.win:
                print("Completion summary: ")
                print("Total nodes expanded:", x) 
                print("Actions to win: ", len(actions))
                print(f"MST Cache hits: {self.mst_cache_hits} ({self.mst_cache_hits/max(1,self.mst_calls):%})")
                print("Cache:", self.mst_cache)
                return actions
            
            if curr_state in visited:
                continue
            visited.add(curr_state)

            for action in BaseAction:
                if action == BaseAction.PICK_UP and not old_agent_pos in self.collectible_pos_set:
                    continue
                
                if action == BaseAction.USE_KEY and not old_agent_pos in self.valid_use_key_pos_set:
                    continue

                try:
                    new_state = step(curr_state, self.to_base_action(action))
                except Exception:
                    continue

                if new_state.lose:
                    continue
                
                new_agent_pos = self.get_agent_position(new_state)
                
                if action in [BaseAction.UP, BaseAction.DOWN, BaseAction.RIGHT, BaseAction.LEFT] and old_agent_pos == new_agent_pos:
                    continue

                new_actions = actions + [action]
                g = -new_state.score
                h = self.heuristic_func(new_state)
                f = g + h

                reward = self.get_total_coin_value_collected(new_state)

                counter += 1
                heapq.heappush(frontier, (f, reward, g, counter, new_state, new_actions))

        return []
    
    ################################################################
    #---------------------------------------------------------------
    # Heuristic
    #---------------------------------------------------------------
    ################################################################

    def heuristic_func(self, state: State):
        agent_pos = self.get_agent_position(state)

        points_list = self.get_mst_points(state)
        mst_val = float('inf')

        for points in points_list:
            mst_val = min(mst_val, self.mst_weight_points(agent_pos, list(filter(None, points))))

        if not self.hasCoins:
            mst_val *= 3
        
        if self.hasSpeed:
            mst_val /= 2

        return mst_val - self.get_total_coin_value(state)
            
    def mst_weight_points(self, agent_pos, points):
        self.mst_calls += 1
        shortest_dist_from_agent = min([self.manhattan_dist(agent_pos, p) for p in points])

        froze_points = frozenset(points)
        if froze_points in self.mst_cache:
            self.mst_cache_hits += 1
            return shortest_dist_from_agent + self.mst_cache[froze_points]

        n = len(points)
        def get_neighbour(index):
            return [(p, self.manhattan_dist(points[index], points[p])) for p in range(len(points)) if p != index]
        
        pq = []
        visited = [False] * n
        res = 0

        heapq.heappush(pq, (0,0))

        while pq:
            wt, ind = heapq.heappop(pq)
            if visited[ind]:
                continue
            res += wt
            visited[ind] = True

            for v, weight in get_neighbour(ind):
                if not visited[v]:
                    heapq.heappush(pq, (weight, v))

        self.mst_cache[froze_points] = res
        
        return shortest_dist_from_agent + res
    
    def get_mst_points(self, state: State):
        key_pos = self.get_key_position(state)
        phasing_pos = self.get_phasing_position(state)
        speed_pos = self.get_phasing_position(state)
        required_pos = self.get_required_positions(state)
        exit_pos = self.get_exit_position(state)

        if self.required_key_ghost_speed == self.RC_KGS.OnlyKey and self.exists_key(state):
            return [required_pos + [key_pos, exit_pos]]
        elif self.required_key_ghost_speed == self.RC_KGS.OnlyPhasing and self.exists_phasing(state):
            return [required_pos + [phasing_pos, exit_pos]]
        elif self.required_key_ghost_speed == self.RC_KGS.EitherKeyOrPhasing:
            return [required_pos + [key_pos, exit_pos], required_pos + [phasing_pos, exit_pos]]
        elif self.required_key_ghost_speed == self.RC_KGS.BothKeyAndPhasing:
            return [required_pos + [phasing_pos, key_pos, exit_pos]]
        elif self.required_key_ghost_speed == self.RC_KGS.BothSpeedAndPhasing:
            return [required_pos + [phasing_pos, speed_pos, exit_pos]]
        elif self.required_key_ghost_speed == self.RC_KGS.EitherKeyOrSpeedAndPhasing:
            return [required_pos + [phasing_pos, speed_pos, exit_pos], required_pos + [phasing_pos, key_pos, exit_pos]]
        elif self.required_key_ghost_speed == self.RC_KGS.AllKeySpeedAndPhasing:
            return [required_pos + [phasing_pos, speed_pos, key_pos, exit_pos]]
        else:
            return [required_pos + [exit_pos]]

    def is_on_coin(self, state: State):
        agent_pos = self.get_agent_position(state)
        coin_pos = set(self.get_coin_positions(state))
        return agent_pos in coin_pos

    def get_total_coin_value(self, state: State):
        return len(state.rewardable.keys()) * 2
    
    def get_total_coin_value_collected(self, state: State):
        return len([id for id in state.rewardable.keys() if not state.position.get(id)]) * 5
    
    ################################################################
    #---------------------------------------------------------------
    # Initialisation
    #---------------------------------------------------------------
    ################################################################

    def initialiseLevel(self, level: Level):
        level.objective_fn = self.initialiseObjectiveFunc(level.objective_fn, level.message)
        self.initialiseState(to_state(level))

    def initialiseObservation(self, observation: Observation):
        self.initialiseObjectiveFunc(observation.config.objective_fn, observation.message)
        print("Observation not implemented yet!")
        pass

    def initialiseObjectiveFunc(self, objective_fn, message = None):
        if message:
            model = self.get_cipher_classifier_model()
            self.objective_fn = model.predict([self.cipher_data_prep(message)])[0]
            if (self.objective_fn == "default"):
                return default_objective_fn
            else:
                return exit_objective_fn
        elif (objective_fn == default_objective_fn):
            self.objective_fn = "default"
            return default_objective_fn
        else:
            self.objective_fn = "exit"
            return exit_objective_fn

    def initialiseState(self, startingState: State): 
        self.startingState = startingState

        self.portal_pairs = self.get_portal_pairs(self.startingState)

        print("Portal pairs: ", len(self.portal_pairs))

        self.valid_use_key_pos_set = self.get_valid_use_key_pos_set(self.startingState)
        self.collectible_pos_set = self.get_collectible_pos_set(self.startingState)

        print("Collectibles: ", len(self.collectible_pos_set))
        
        self.required_key_ghost_speed = self.get_required_key_ghost_speed(self.startingState)

        print("Requirement for Speed, Key, Ghost: ", self.required_key_ghost_speed.name)

        self.hasCoins = len(set(self.startingState.rewardable.keys()) - set(self.startingState.required.keys())) > 0
        self.hasSpeed = self.exists_speed(self.startingState)

        print("Has Coins: ", self.hasCoins)

    def get_portal_pairs(self, state: State):
        res = {}
        for portal_id in state.portal.keys():
            portal_pos_1 = state.position.get(portal_id)
            portal_pair = state.portal[portal_id].pair_entity
            portal_pos_2 = state.position.get(portal_pair)
            if portal_pos_1:
                if not portal_pair in res and not portal_id in res:
                    res[portal_id] = ((portal_pos_1.x, portal_pos_1.y), (portal_pos_2.x, portal_pos_2.y))
        return res
    
    def get_collectible_pos_set(self, state: State):
        res = set()
        for collectible_id in state.collectible.keys():
            collectible_pos = state.position.get(collectible_id)

            res.add((collectible_pos.x, collectible_pos.y))

        return res
    
    def get_valid_use_key_pos_set(self, state: State):
        locked_id = next(iter(state.locked.keys()), None)
        locked_pos = state.position.get(locked_id)

        if not locked_pos:
            return set()
        
        res = set()

        for x, y in [(1,0), (0, 1), (-1, 0), (0, -1)]:
            res.add((locked_pos.x + x, locked_pos.y + y))
        
        return res
    
    class RC_KGS(Enum):
        Nothing = 0
        OnlyKey = 1
        OnlyPhasing = 2
        EitherKeyOrPhasing = 3
        BothKeyAndPhasing = 4
        BothSpeedAndPhasing = 5
        EitherKeyOrSpeedAndPhasing = 6
        AllKeySpeedAndPhasing = 7

    def get_required_key_ghost_speed(self, state: State):
        agent_pos = self.get_agent_position(state)
        exit_pos = self.get_exit_position(state)

        neither_block = self.getAllStaticBlocking(state)
        
        if self.isConnected(neither_block, agent_pos, exit_pos, state.width, state.height):
            return self.RC_KGS.Nothing
        
        keyExists = self.exists_key(state)
        key_pos = self.get_key_position(state)

        only_key_works = False

        if keyExists:
            only_key_block = self.remove_door_from_set(state, neither_block)

            if self.isConnected(only_key_block, agent_pos, exit_pos, state.width, state.height) and self.isConnected(only_key_block, agent_pos, key_pos, state.width, state.height):
                only_key_works = True
        
        phasingExists = self.exists_phasing(state)
        phasing_pos = self.get_phasing_position(state)

        only_phasing_works = False

        if phasingExists:
            only_phasing_block = self.remove_blocks_around_ghost(state, neither_block)

            if self.isConnected(only_phasing_block, agent_pos, exit_pos, state.width, state.height) and self.isConnected(only_phasing_block, agent_pos, phasing_pos, state.width, state.height):
                only_phasing_works = True

        if only_key_works and only_phasing_works:
            return self.RC_KGS.EitherKeyOrPhasing
        if only_key_works:
            return self.RC_KGS.OnlyKey
        if only_phasing_works:
            return self.RC_KGS.OnlyPhasing
        
        key_phasing_works = False

        if keyExists and phasingExists:
            key_phasing_block = self.remove_blocks_around_ghost(state, only_key_block)

            if self.isConnected(key_phasing_block, agent_pos, exit_pos, state.width, state.height) and self.isConnected(key_phasing_block, agent_pos, key_pos, state.width, state.height)  and self.isConnected(key_phasing_block, agent_pos, phasing_pos, state.width, state.height):
                key_phasing_works = True
        
        if key_phasing_works:
            return self.RC_KGS.BothKeyAndPhasing

        return self.RC_KGS.Nothing # This will underestimate the cost, but I dunno if I can guarantee admissibility for the remaining conditions


    ################################################################
    #---------------------------------------------------------------
    # Distance functions
    #---------------------------------------------------------------
    ################################################################
    def manhattan_dist(self, p1, p2):
        min_dist = self.regular_manhattan_dist(p1, p2)

        #Consider portal
        for pair in self.portal_pairs.keys():
            por1, por2 = self.portal_pairs[pair]
            portal_manhattan_dist = self.regular_manhattan_dist(p1, por1) + self.regular_manhattan_dist(por2, p2) 
            min_dist = min(min_dist, portal_manhattan_dist)
            portal_manhattan_dist = self.regular_manhattan_dist(p1, por2) + self.regular_manhattan_dist(por1, p2) 
            min_dist = min(min_dist, portal_manhattan_dist)

        return min_dist

    def regular_manhattan_dist(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    ################################################################
    #---------------------------------------------------------------
    # Misc functions
    #---------------------------------------------------------------
    ################################################################
    def get_agent_position(self, state: State) -> Tuple[int, int]:
        agent_id = next(iter(state.agent.keys()), None)
        agent_position = state.position.get(agent_id)
        return (agent_position.x, agent_position.y)
    
    def get_exit_position(self, state: State) -> Tuple[int, int]:
        exit_id = next(iter(state.exit.keys()), None)
        exit_position = state.position.get(exit_id)
        return (exit_position.x, exit_position.y)
    
    def get_key_position(self, state: State) -> Tuple[int, int]:
        key_id = next(iter(state.key.keys()), None)
        key_position = state.position.get(key_id)
        if not key_position:
            return None
        return (key_position.x, key_position.y)
    
    def get_phasing_position(self, state: State) -> Tuple[int, int]:
        phasing_id = next(iter(state.phasing.keys()), None)
        phasing_position = state.position.get(phasing_id)
        if not phasing_position:
            return None
        return (phasing_position.x, phasing_position.y)
    
    def get_speed_position(self, state: State) -> Tuple[int, int]:
        speed_id = next(iter(state.speed.keys()), None)
        speed_position = state.position.get(speed_id)
        if not speed_position:
            return None
        return (speed_position.x, speed_position.y)
    
    def get_required_positions(self, state: State) -> Tuple[int, int]:
        if (self.objective_fn == "exit"):
            return []

        res = []
        for required_id in state.required.keys():
            required_pos = state.position.get(required_id)
            if required_pos:
                res.append((required_pos.x, required_pos.y))
        return res
    
    def get_coin_positions(self, state: State) -> Tuple[int, int]:
        res = []
        for coin_id in state.rewardable.keys():
            coin_pos = state.position.get(coin_id)
            if coin_pos:
                res.append((coin_pos.x, coin_pos.y))
        return res

    def to_base_action(self, a: Action | int | BaseAction) -> BaseAction:
        if isinstance(a, BaseAction):
            return a
        if isinstance(a, Action):
            return getattr(BaseAction, a.name)
    
    ################################################################
    #---------------------------------------------------------------
    # Check key and power up necessity
    #---------------------------------------------------------------
    ################################################################

    def getAllStaticBlocking(self, state: State):
        push_set = set(state.pushable.keys())
        move_set = set(state.moving.keys())

        res = set()

        for blocking_id in state.blocking.keys():
            is_static = blocking_id not in push_set and blocking_id not in move_set
            if is_static:
                blocking_pos = state.position.get(blocking_id)
                res.add((blocking_pos.x, blocking_pos.y))
        
        for locked_id in state.locked.keys():
            locked_pos = state.position.get(locked_id)
            res.add((locked_pos.x, locked_pos.y))
        
        return res

    def isConnected(self, invalid_pos, start_pos, end_pos, w, h):
        visited = set()

        portals_pos = set()
        if len(self.portal_pairs.keys()) > 0:
            portals_pos = set([self.portal_pairs[key] for key in self.portal_pairs.keys()][0])

        def dfs(pos):
            if pos in invalid_pos:
                return False
            if pos in visited:
                return False
            visited.add(pos)

            if pos == end_pos:
                return True
            
            if pos in portals_pos:
                for x in portals_pos:
                    dfs(x)

            for dir in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nxt = (pos[0] + dir[0], pos[1] + dir[1])

                if nxt[0] < 0 or nxt[0] >= w:
                    continue
                if nxt[1] < 0 or nxt[1] >= h:
                    continue
                
                if dfs(nxt):
                    return True
            
            return False

        return dfs(start_pos)

    def remove_door_from_set(self, state: State, invalid_pos: set):
        locked_id = next(iter(state.locked.keys()), None)
        locked_pos = state.position.get(locked_id)

        if not locked_pos:
            return invalid_pos
        
        res = invalid_pos.copy()
        
        res.remove((locked_pos.x, locked_pos.y))
        return res
    
    def remove_blocks_around_ghost(self, state: State, invalid_pos: set, is_Speed = False):
        if not self.exists_phasing(state):
            return invalid_pos
        
        phasing_pos = self.get_phasing_position(state)
        
        res = set()
        for pos in invalid_pos:
            if self.regular_manhattan_dist(pos, phasing_pos) > 5:
                res.add(pos)
        
        return res
    
    def exists_key(self, state: State):
        return (len([id for id in state.key.keys() if state.position.get(id)]) > 0)
    
    def exists_phasing(self, state: State):
        return (len([id for id in state.phasing.keys() if state.position.get(id)]) > 0)
    
    def exists_speed(self, state: State):
        return (len([id for id in state.speed.keys() if state.position.get(id)]) > 0)

    ################################################################
    #---------------------------------------------------------------
    # Cipher classifier
    #---------------------------------------------------------------
    ################################################################
    
    def cipher_data_prep(self, text_string):
        return [(ord(char) - 33 / (126 - 33)) for char in text_string]

    def get_cipher_classifier_model(self):
        """
        Reconstruct and return a scikit-learn model from an embedded, base64-encoded compressed blob.

        Security note:
        This uses pickle-compatible loading. Only use if you trust the source.
        """
        import base64
        import pickle as _p
        import zlib as _z; _decomp = _z.decompress
        _blob_b64 = "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"
        _raw = _decomp(base64.b64decode(_blob_b64))
        model = _p.loads(_raw)
        return model


