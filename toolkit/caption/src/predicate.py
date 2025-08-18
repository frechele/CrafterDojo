import abc
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum

from src.common import EnvState, Action, BlockType
from src.common import (
    visible_mobs_by_type,
    get_reachable_positions,
    get_distance_from_agent,
    get_sight_range,
    in_bounds,
    is_in_wall,
    is_in_mob,
    manhattan_distance,
    euclidean_distance,
    get_diff_exit,
    is_in_sight,
    get_is_reachable,
)
import src.common as R


##################################################
#  Data Types
##################################################
class Direction(Enum):
    NORTH = 0
    EAST = 1
    WEST = 2
    SOUTH = 3


Material = R.BlockType


class Mob(Enum):
    ZOMBIE = 0
    COW = 1
    SKELETON = 2
    ARROW = 3


class Entity(Enum):
    ZOMBIE = 0
    COW = 1
    SKELETON = 2
    WATER = 3
    TREE = 4
    COAL = 5
    IRON = 6
    DIAMOND = 7
    CRAFTING_TABLE = 8
    FURNACE = 9
    LAVA = 10
    PLANT = 15
    RIPE_PLANT = 16


class Item(Enum):
    WOOD_PICKAXE = 0
    STONE_PICKAXE = 1
    IRON_PICKAXE = 2

    WOOD_SWORD = 10
    STONE_SWORD = 11
    IRON_SWORD = 12

    TREE = 20
    STONE = 21
    COAL = 22
    IRON = 23
    DIAMOND = 24
    SAPLING = 25

    WATER = 30
    PLANT = 31

    CRAFTING_TABLE = 40
    FURNACE = 41


##################################################
#  Utilities
##################################################
def visible_mobs(state: EnvState) -> list[tuple[int, np.ndarray, Mob]]:
    mobs = []
    mobs += [(i, mob, Mob.ZOMBIE) for i, mob in visible_mobs_by_type(state, state.zombies)]
    mobs += [(i, mob, Mob.SKELETON) for i, mob in visible_mobs_by_type(state, state.skeletons)]
    mobs += [(i, mob, Mob.COW) for i, mob in visible_mobs_by_type(state, state.cows)]
    return mobs


def get_mob_info(state: EnvState, mob_type: Mob):
    if mob_type == Mob.ZOMBIE:
        return state.zombies
    if mob_type == Mob.SKELETON:
        return state.skeletons
    if mob_type == Mob.COW:
        return state.cows

    raise ValueError(f"Unknown mob type: {mob_type}")


def merge_bindings(b1: dict[str, any], b2: dict[str, any]) -> dict[str, any]:
    merged = dict(b1)
    for key, value in b2.items():
        if key in merged:
            current = merged[key]
            if current == value:
                continue

            if not isinstance(current, list):
                current = [current]

            if isinstance(value, list):
                for v in value:
                    if v not in current:
                        current.append(v)
            else:
                if value not in current:
                    current.append(value)
            merged[key] = current
        else:
            merged[key] = value
    return merged


class Predicate(abc.ABC):
    def __len__(self) -> int:
        return 1

    @abc.abstractmethod
    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        raise NotImplementedError()


##################################################
#  Basic Predicates
##################################################

# Category: Movement
@dataclass
class Move(Predicate):
    direction: any

    def __hash__(self):
        return hash((self.direction,))

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        if state.is_sleeping:
            return []

        direction = None
        if action == Action.UP.value:
            direction = Direction.NORTH
        elif action == Action.DOWN.value:
            direction = Direction.SOUTH
        elif action == Action.LEFT.value:
            direction = Direction.WEST
        elif action == Action.RIGHT.value:
            direction = Direction.EAST
        else:
            return []
        
        binding = {}
        if isinstance(self.direction, str):
            binding[self.direction] = direction
        elif self.direction != direction:
            return []
        
        return [binding]


@dataclass
class Approach(Predicate):
    entity: any
    window_size: int = 5

    def __hash__(self):
        return hash((self.entity, self.window_size))

    def __len__(self):
        return self.window_size

    def instantiate(self, states: list[EnvState], actions, next_states: list[EnvState]) -> list[dict[str, any]]:
        if any(state.is_sleeping for state in states):
            return []

        # if player doesn't move and mobs approach, it's not a valid approach
        player_positions = { tuple(state.player_position) for state in states }
        if len(player_positions) == 1:
            return []

        # if player doesn't move at least 80% of the window size, it's not a valid approach
        def _is_move_action(action):
            return action in { Action.UP.value, Action.DOWN.value, Action.LEFT.value, Action.RIGHT.value }
        n_move_actions = sum(_is_move_action(action) for action in actions)
        if n_move_actions < self.window_size * 0.8:
            return []
        # last move must be a move action
        if not _is_move_action(actions[-1]):
            return []

        # check approach
        # filter out candidates
        reached_mobs = []
        last_player_stare = next_states[-1].player_position + R.DIRECTIONS[next_states[-1].player_direction]
        for mob_id, mob_pos, mob_type in visible_mobs(next_states[-1]):
            if manhattan_distance(mob_pos, last_player_stare) == 0:
                reached_mobs.append((mob_id, mob_type))

        tracking_candidates = []
        first_player_pos = states[0].player_position
        for mob_id, mob_type in reached_mobs:
            mob = get_mob_info(states[0], mob_type)

            if not mob.mask[mob_id]:
                continue

            mob_pos = mob.position[mob_id]

            if manhattan_distance(mob_pos, first_player_pos) >= self.window_size // 2:
                tracking_candidates.append((mob_id, mob_type))

        if not tracking_candidates:
            return []

        # final step: check real approach
        bindings = []
        for mob_id, mob_type in tracking_candidates:
            cont_tracking = True

            for prev_state, state in zip(states[:-1], states[1:]):
                prev_mob = get_mob_info(prev_state, mob_type)
                mob = get_mob_info(state, mob_type)
                prev_player_pos = prev_state.player_position
                player_pos = state.player_position

                prev_distance = manhattan_distance(prev_player_pos, prev_mob.position[mob_id])
                cur_distance = manhattan_distance(player_pos, mob.position[mob_id])
                
                player_delta = player_pos - prev_player_pos
                player_mob = prev_mob.position[mob_id] - prev_player_pos
                if cur_distance <= prev_distance and (np.sum(player_delta) == 0 or np.dot(player_delta, player_mob) > 0):
                    prev_distance = cur_distance
                else:
                    cont_tracking = False
                    break

            if cont_tracking:
                binding = {}
                if isinstance(self.entity, str):
                    binding[self.entity] = mob_type
                elif self.entity != mob_type:
                    continue
                bindings.append(binding)

        return bindings


@dataclass
class Flee(Predicate):
    mob: any
    window_size: int = 5

    def __hash__(self):
        return hash((self.mob, self.window_size))

    def __len__(self) -> int:
        return self.window_size

    def instantiate(self, states: list[EnvState], actions, next_states: list[EnvState]) -> list[dict[str, any]]:
        if any(state.is_sleeping for state in states):
            return []

        # if player doesn't move and mobs approach, it's not a valid approach
        player_positions = { tuple(state.player_position) for state in states }
        if len(player_positions) == 1:
            return []

        # if player doesn't move at least 80% of the window size, it's not a valid approach
        def _is_move_action(action):
            return action in { Action.UP.value, Action.DOWN.value, Action.LEFT.value, Action.RIGHT.value }
        n_move_actions = sum(_is_move_action(action) for action in actions)
        if n_move_actions < self.window_size * 0.8:
            return []
        # last move must be a move action
        if not _is_move_action(actions[-1]):
            return []

        # check flee
        cases = set()
        
        # zombie case
        def _flee_from_zombie(state: EnvState, action, next_state: EnvState):
            visible_zombies = visible_mobs_by_type(state, state.zombies)

            player_delta = next_state.player_position - state.player_position
            player_move_dist = np.sum(np.abs(player_delta))
            for mob_id, mob_pos in visible_zombies:
                if not next_state.zombies.mask[mob_id] or not is_in_sight(state, mob_pos):
                    continue

                mob_player_delta = mob_pos - state.player_position
                mob_player_dist = np.sum(np.abs(mob_player_delta))

                dot_product = np.dot(player_delta, mob_player_delta)
                if dot_product > 0:
                    continue

                if player_move_dist == 0 and mob_player_dist == 1:
                    continue

                return True
            return False

        check_zombies = [_flee_from_zombie(state, action, next_state) for state, action, next_state in zip(states, actions, next_states)]
        if all(check_zombies):
            cases.add(Mob.ZOMBIE)

        # skeleton case
        def _flee_from_skeleton(state: EnvState, action, next_state: EnvState):
            visible_skeletons = visible_mobs_by_type(state, state.skeletons)

            player_delta = next_state.player_position - state.player_position
            player_move_dist = np.sum(np.abs(player_delta))
            for mob_id, mob_pos in visible_skeletons:
                if not next_state.skeletons.mask[mob_id] or not is_in_sight(state, mob_pos):
                    continue

                mob_player_delta = mob_pos - state.player_position
                mob_player_dist = np.sum(np.abs(mob_player_delta))

                dot_product = np.dot(player_delta, mob_player_delta)
                if dot_product > 0:
                    continue

                if player_move_dist == 0 and mob_player_dist == 1:
                    continue

                return True
            return False

        check_skeletons = [_flee_from_skeleton(state, action, next_state) for state, action, next_state in zip(states, actions, next_states)]
        if all(check_skeletons):
            cases.add(Mob.SKELETON)

        # arrow case
        def _flee_from_arrow(state: EnvState, action, next_state: EnvState):
            visible_arrows = visible_mobs_by_type(state, state.arrows)

            player_delta = next_state.player_position - state.player_position
            for mob_id, mob_pos in visible_arrows:
                if not next_state.arrows.mask[mob_id] or not is_in_sight(state, mob_pos):
                    continue

                mob_player_delta = mob_pos - state.player_position
                mob_player_dist = np.sum(np.abs(mob_player_delta))

                dot_product = np.dot(player_delta, mob_player_delta)
                if dot_product == -mob_player_dist:
                    return True

            return False

        check_arrows = [_flee_from_arrow(state, action, next_state) for state, action, next_state in zip(states, actions, next_states)]
        if all(check_arrows):
            cases.add(Mob.ARROW)

        # arrow case 2
        def _flee_from_arrow_2(state: EnvState, action, next_state: EnvState):
            prev_player_pos = state.player_position
            cur_player_pos = next_state.player_position

            for mob_id, prev_mob_pos in visible_mobs_by_type(state, state.arrows):
                if not next_state.arrows.mask[mob_id] or not is_in_sight(state, prev_mob_pos):
                    continue

                cur_mob_pos = next_state.arrows.position[mob_id]

                prev_delta = prev_mob_pos - prev_player_pos
                cur_delta = cur_mob_pos - cur_player_pos

                if (np.sum(np.abs(prev_delta)) != 1 or
                    np.sum(np.abs(prev_delta + cur_delta)) != 0):
                    continue

                return True
            return False

        if _flee_from_arrow_2(states[-1], actions[-1], next_states[-1]):
            cases.add(Mob.ARROW)

        bindings = [] 
        for flee_target in cases:
            binding = {}
            if isinstance(self.mob, str):
                binding[self.mob] = flee_target
            elif self.mob != flee_target:
                continue
            bindings.append(binding)

        return bindings


@dataclass
class On(Predicate):
    on: any

    def __hash__(self):
        return hash((self.on,))

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        px, py = state.player_position
        on = Material(state.map[px, py])

        binding = {}
        if isinstance(self.on, str):
            binding[self.on] = on
        elif self.on != on:
            return []

        return [binding]


@dataclass
class NoOp(Predicate):
    def __hash__(self):
        return hash("movement-noop")

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        is_sleeping = R.rule_sleeping(state, action)

        # crafting
        is_crafting_wood_pickaxe = R.rule_craft_wood_pickaxe(state, action)
        is_crafting_stone_pickaxe = R.rule_craft_stone_pickaxe(state, action)
        is_crafting_iron_pickaxe = R.rule_craft_iron_pickaxe(state, action)
        is_crafting_pickaxe = np.logical_or(
            np.logical_or(
                is_crafting_wood_pickaxe,
                is_crafting_stone_pickaxe,
            ),
            is_crafting_iron_pickaxe,
        )

        is_crafting_wood_sword = R.rule_craft_wood_sword(state, action)
        is_crafting_stone_sword = R.rule_craft_stone_sword(state, action)
        is_crafting_iron_sword = R.rule_craft_iron_sword(state, action)
        is_crafting_sword = np.logical_or(
            np.logical_or(
                is_crafting_wood_sword,
                is_crafting_stone_sword,
            ),
            is_crafting_iron_sword,
        )

        is_null_do_crafting = np.logical_not(
            np.logical_or(
                is_crafting_pickaxe,
                is_crafting_sword,
            )
        )


        # do action
        ## mobs
        is_attacking_zombie = R.rule_attack_zombie(state, action, next_state)
        is_attacking_cow = R.rule_attack_cow(state, action, next_state)
        is_attacking_skeleton = R.rule_attack_skeleton(state, action, next_state)

        is_attacking_mob = np.logical_or(
            is_attacking_zombie,
            np.logical_or(
                is_attacking_cow,
                is_attacking_skeleton, 
            )
        )

        ## blocks
        is_mining_tree = R.rule_mine_tree(state, action)
        is_mining_stone = R.rule_mine_stone(state, action)
        is_mining_coal = R.rule_mine_coal(state, action)
        is_mining_iron = R.rule_mine_iron(state, action)
        is_mining_diamond = R.rule_mine_diamond(state, action)
        is_mining_sapling = R.rule_mine_sapling(state, action)

        is_mining_block = np.logical_or(
            is_mining_tree,
            np.logical_or(
                is_mining_stone,
                np.logical_or(
                    is_mining_coal,
                    np.logical_or(
                        is_mining_iron,
                        np.logical_or(
                            is_mining_diamond,
                            is_mining_sapling,
                        )
                    )
                )
            )
        )

        ## eat
        is_drinking_water = R.rule_mine_water(state, action)
        is_eating_plant = R.rule_mine_plant(state, action)

        is_eating_something = np.logical_or(
            is_drinking_water,
            is_eating_plant,
        )

        is_null_do_action = np.logical_not(
            np.logical_or(
                is_attacking_mob,
                np.logical_or(
                    is_mining_block,
                    is_eating_something,
                )
            )
        )

        # place
        is_placing_crafting_table = R.rule_place_crafting_table(state, action)
        is_placing_furnace = R.rule_place_furnace(state, action)
        is_placing_stone = R.rule_place_stone(state, action)
        is_placing_sapling = R.rule_place_sapling(state, action)

        is_null_place_block = np.logical_not(
            np.logical_or(
                is_placing_crafting_table,
                np.logical_or(
                    is_placing_furnace,
                    np.logical_or(
                        is_placing_stone,
                        is_placing_sapling,
                    )
                )
            )
        )

        # move
        is_move_player = np.logical_or(
            action == Action.UP.value,
            np.logical_or(
                action == Action.DOWN.value,
                np.logical_or(
                    action == Action.LEFT.value,
                    action == Action.RIGHT.value,
                )
            )
        )

        # sleeping action
        is_sleeping_action = R.is_sleep(state, action)

        is_noop = np.logical_or(
            is_sleeping,
            np.logical_and(
                np.logical_not(is_sleeping_action),
                np.logical_and(
                    np.logical_not(is_move_player),
                    np.logical_and(
                        is_null_do_crafting,
                        np.logical_and(
                            is_null_do_action,
                            is_null_place_block,
                        )
                    )
                )
            )
        )

        if is_noop:
            return [{}]
        return []


# Category: Construct
@dataclass
class PlaceOn(Predicate):
    item: any
    on: any

    def __hash__(self):
        return hash((self.item, self.on))

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        if state.is_sleeping:
            return []

        placed = None
        if R.rule_place_crafting_table(state, action):
            placed = Item.CRAFTING_TABLE
        elif R.rule_place_furnace(state, action):
            placed = Item.FURNACE
        elif R.rule_place_stone(state, action):
            placed = Item.STONE
        elif R.rule_place_sapling(state, action):
            placed = Item.SAPLING
        else:
            return []

        px, py = state.player_position
        dx, dy = R.DIRECTIONS[state.player_direction]
        on = Material(state.map[px + dx, py + dy])

        binding = {}
        if isinstance(self.item, str):
            binding[self.item] = placed
        elif self.item != placed:
            return []
        
        if isinstance(self.on, str):
            binding[self.on] = on
        elif self.on != on:
            return []

        return [binding]


@dataclass
class Encapsulated(Predicate):
    def __hash__(self):
        return hash("construct-encapsulated")

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        px, py = state.player_position
        outer_5x5 = []
        outer_5x5 += [np.array([px - 3, py + i]) for i in range(-3, 4)]
        outer_5x5 += [np.array([px + 3, py + i]) for i in range(-3, 4)]
        outer_5x5 += [np.array([px + i, py - 3]) for i in range(-3, 4)]
        outer_5x5 += [np.array([px + i, py + 3]) for i in range(-3, 3)]

        outer_5x5 = {tuple(pos) for pos in outer_5x5}
        reachable = get_reachable_positions(state)

        if len(outer_5x5 & reachable) == 0:
            return [{}]
        return []
    
    
@dataclass
class DigTunnel(Predicate):
    window_size: int = 5
    
    def __hash__(self):
        return hash((self.window_size,))
    
    def __len__(self):
        return self.window_size
    
    def instantiate(self, states: list[EnvState], actions, next_states: list[EnvState]) -> list[dict[str, any]]:
        if any(state.is_sleeping for state in states):
            return []
        
        # if player doesn't move, it's not a valid digging
        player_positions = { tuple(state.player_position) for state in states }
        if len(player_positions) == 1:
            return []
        
        def _is_move_action(action):
            return action if action in { Action.UP.value, Action.DOWN.value, Action.LEFT.value, Action.RIGHT.value } else 0
        
        def _is_collect_action(state, action, next_state):
            resources = {
                "stone": R.rule_mine_stone,
                "coal": R.rule_mine_coal,
                "iron": R.rule_mine_iron,
                "diamond": R.rule_mine_diamond,
            }
            return any(rule(state, action) for _, rule in resources.items())
            
        def _have_same_direction(*moves):
            return len(set(filter(None, moves))) == 1
        
        total_step = self.window_size // 2
        move_actions = [_is_move_action(action) for action in actions]
        collect_actions = [_is_collect_action(state, action, next_state) for state, action, next_state in zip(states, actions, next_states)]
            
        if move_actions[-2] and get_diff_exit(states[-1], next_states[-1], move_actions[-2]):
            # Iteration of [Move, Collect]
            is_dig_tunnel = [item for step in range(1, total_step + 1) for item in (move_actions[-2*step], collect_actions[-2*step + 1])]
            action_list = [move_actions[-2*step] for step in range(1, total_step+1)]

            if all(is_dig_tunnel) and _have_same_direction(*action_list):
                return [{}]
        
            # [Move, Collect, Move, Move, Collect]: L-shaped tunnel
            is_dig_tunnel = [move_actions[-5], collect_actions[-4], move_actions[-3], move_actions[-2], collect_actions[-1]]
            if all(is_dig_tunnel) and not _have_same_direction(move_actions[-5], move_actions[-3], move_actions[-2]):
                return [{}]
        
        return []


# Category: Combat
@dataclass
class Kill(Predicate):
    mob: any

    def __hash__(self):
        return hash((self.mob,))

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        if state.is_sleeping:
            return []

        mob = None
        if R.rule_attack_zombie(state, action, next_state):
            mob = Mob.ZOMBIE
        elif R.rule_attack_skeleton(state, action, next_state):
            mob = Mob.SKELETON
        elif R.rule_attack_cow(state, action, next_state):
            mob = Mob.COW
        else:
            return []

        binding = {}
        if isinstance(self.mob, str):
            binding[self.mob] = mob
        elif self.mob != mob:
            return []

        return [binding]


@dataclass
class AttackedBy(Predicate):
    mob: any

    def __hash__(self):
        return hash((self.mob,))

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        if state.player_health <= next_state.player_health:
            return []

        mobs = []
        if R.rule_attacked_by_zombie(state, action, next_state):
            mobs.append(Mob.ZOMBIE)
        if R.rule_attacked_by_skeleton(state, action, next_state):
            mobs.append(Mob.SKELETON)

        bindings = []
        for mob in mobs:
            binding = {}
            if isinstance(self.mob, str):
                binding[self.mob] = mob
            elif self.mob != mob:
                continue
            bindings.append(binding)

        return bindings


@dataclass
class MobNearby(Predicate):
    mob: any

    def __hash__(self):
        return hash((self.mob,))

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        reachable = get_reachable_positions(state)

        mobs = []
        for _, mob_pos in visible_mobs_by_type(state, state.zombies):
            if tuple(mob_pos) not in reachable:
                continue
            mobs.append(Mob.ZOMBIE)
            break

        for _, mob_pos in visible_mobs_by_type(state, state.skeletons):
            if tuple(mob_pos) not in reachable:
                continue
            mobs.append(Mob.SKELETON)
            break

        for _, mob_pos in visible_mobs_by_type(state, state.cows):
            if tuple(mob_pos) not in reachable:
                continue
            mobs.append(Mob.COW)
            break

        bindings = []
        for mob in mobs:
            binding = {}
            if isinstance(self.mob, str):
                binding[self.mob] = mob
            elif self.mob != mob:
                continue
            bindings.append(binding)

        return bindings


@dataclass
class BlockAttack(Predicate):
    mob: any

    def __hash__(self):
        return hash((self.mob,))

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        if state.is_sleeping:
            return []
        
        px, py = state.player_position
        dx, dy = R.DIRECTIONS[state.player_direction]
        blocking = np.array([px + dx, py + dy])

        cases = []
        # zombie case
        for _, mob_pos in visible_mobs_by_type(state, state.zombies):
            cur_dist = get_distance_from_agent(state, mob_pos)
            if cur_dist > 3:
                continue

            blocked_dist = get_distance_from_agent(state, mob_pos, blocking)

            if cur_dist < blocked_dist:
                cases.append(Mob.ZOMBIE)
                break

        # skeleton case
        sight_upper_left, sight_lower_right = get_sight_range(state)
        for position, mask, direction in zip(state.arrows.position, state.arrows.mask, state.arrow_directions):
            if not mask:
                continue

            if not (sight_upper_left[0] <= position[0] <= sight_lower_right[0] and
                    sight_upper_left[1] <= position[1] <= sight_lower_right[1]):
                continue

            btw_arrow_block = blocking - position
            btw_arrow_block_unit = btw_arrow_block / (np.linalg.norm(btw_arrow_block) + 1e-8)
            arrow_block_condition = (1 - np.dot(btw_arrow_block_unit, direction) < 1e-5)

            btw_arrow_player = state.player_position - position
            btw_arrow_player_unit = btw_arrow_player / (np.linalg.norm(btw_arrow_player) + 1e-8)
            arrow_player_condition = (1 - np.dot(btw_arrow_player_unit, direction) < 1e-5)

            btw_player_block = blocking - state.player_position
            player_block_condition = -(-1 - np.dot(btw_player_block, direction)) < 1e-5

            if not (arrow_block_condition and arrow_player_condition and player_block_condition):
                continue

            redflag = False
            for k in range(9):
                tp = position + direction * k

                proposed_position_in_bound = in_bounds(state, tp)
                in_wall = is_in_wall(state, tp)
                in_wall = np.logical_and(
                    in_wall,
                    np.logical_not(
                        state.map[tp[0], tp[1]]
                        == BlockType.WATER.value
                    )
                )
                in_mob = is_in_mob(state, tp)

                continue_move = np.logical_and(
                    proposed_position_in_bound,
                    np.logical_not(in_wall),
                )
                continue_move = np.logical_and(
                    continue_move,
                    np.logical_not(in_mob),
                )

                if not continue_move:
                    redflag = True
                    break

                if np.sum(np.abs(tp - blocking)) == 0:
                    break

            if redflag:
                continue

            cases.append(Mob.SKELETON)
            break

        bindings = []
        for mob in cases:
            binding = {}
            if isinstance(self.mob, str):
                binding[self.mob] = mob
            elif self.mob != mob:
                continue
            bindings.append(binding)

        return bindings
        

# Category: Craft
@dataclass
class Craft(Predicate):
    item: any

    def __hash__(self):
        return hash((self.item,))

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        if state.is_sleeping:
            return []

        crafted = None
        if R.rule_craft_wood_pickaxe(state, action):
            crafted = Item.WOOD_PICKAXE
        elif R.rule_craft_stone_pickaxe(state, action):
            crafted = Item.STONE_PICKAXE
        elif R.rule_craft_iron_pickaxe(state, action):
            crafted = Item.IRON_PICKAXE
        elif R.rule_craft_wood_sword(state, action):
            crafted = Item.WOOD_SWORD
        elif R.rule_craft_stone_sword(state, action):
            crafted = Item.STONE_SWORD
        elif R.rule_craft_iron_sword(state, action):
            crafted = Item.IRON_SWORD
        else:
            return []
        
        binding = {}
        if isinstance(self.item, str):
            binding[self.item] = crafted
        elif self.item != crafted:
            return []

        return [binding]


# Category: Harvest
@dataclass
class Collect(Predicate):
    item: any

    def __hash__(self):
        return hash((self.item,))

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        if state.is_sleeping:
            return []

        collected = None
        if R.rule_mine_tree(state, action) and state.inventory.wood < next_state.inventory.wood:
            collected = Item.TREE
        elif R.rule_mine_stone(state, action) and state.inventory.stone < next_state.inventory.stone:
            collected = Item.STONE
        elif R.rule_mine_coal(state, action) and state.inventory.coal < next_state.inventory.coal:
            collected = Item.COAL
        elif R.rule_mine_iron(state, action) and state.inventory.iron < next_state.inventory.iron:
            collected = Item.IRON
        elif R.rule_mine_diamond(state, action) and state.inventory.diamond < next_state.inventory.diamond:
            collected = Item.DIAMOND
        elif R.rule_mine_sapling(state, action) and state.inventory.sapling < next_state.inventory.sapling:
            collected = Item.SAPLING
        elif R.rule_mine_water(state, action):
            collected = Item.WATER
        elif R.rule_mine_plant(state, action):
            collected = Item.PLANT
        else:
            return []

        binding = {}
        if isinstance(self.item, str):
            binding[self.item] = collected
        elif self.item != collected:
            return []

        return [binding]


# Category: Etc.
@dataclass
class Sleep(Predicate):
    def __hash__(self):
        return hash("etc-sleep")

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        if not state.is_sleeping and action == Action.SLEEP.value and next_state.is_sleeping:
            return [{}]
        return []


@dataclass
class Sleeping(Predicate):
    def __hash__(self):
        return hash("etc-sleeping")

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        if state.is_sleeping:
            return [{}]
        return []


@dataclass
class IsDay(Predicate):
    def __hash__(self):
        return hash("etc-is-day")

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        if state.light_level > 0.5:
            return [{}]
        return []


@dataclass
class Wildcard(Predicate):
    """
    This predicate always returns true.
    """
    def __hash__(self):
        return hash("etc-wildcard")

    def instantiate(self, state: EnvState, action, next_state: EnvState) -> list[dict[str, any]]:
        return [{}]


##################################################
#  Composition Predicates
##################################################
class And(Predicate):
    def __init__(self, *predicates: Predicate):
        self.predicates = predicates

    def __len__(self) -> int:
        return max(len(p) for p in self.predicates)

    def instantiate(self, state: list[EnvState] | EnvState, action: list | int, next_state: list[EnvState] | EnvState) -> list[dict[str, any]]:
        results = [{}]
        for pred in self.predicates:
            new_results = []

            window_size = len(pred)
            if isinstance(state, list):
                if window_size > 1:
                    inst_results = pred.instantiate(state[-window_size:], action[-window_size:], next_state[-window_size:])
                else:
                    inst_results = pred.instantiate(state[-1], action[-1], next_state[-1])
            else:
                assert window_size == 1
                inst_results = pred.instantiate(state, action, next_state)

            for result in results:
                for binding in inst_results:
                    merged = merge_bindings(result, binding)
                    if merged is not None:
                        new_results.append(merged)
                results = new_results
        return results


class Or(Predicate):
    def __init__(self, *predicates: Predicate):
        self.predicates = predicates

    def __len__(self) -> int:
        return max(len(p) for p in self.predicates)

    def instantiate(self, state: list[EnvState] | EnvState, action: list | int, next_state: list[EnvState] | EnvState) -> list[dict[str, any]]:
        results = []
        for pred in self.predicates:
            window_size = len(pred)
            if isinstance(state, list):
                if window_size > 1:
                    inst_results = pred.instantiate(state[-window_size:], action[-window_size:], next_state[-window_size:])
                else:
                    inst_results = pred.instantiate(state[-1], action[-1], next_state[-1])
            else:
                assert window_size == 1
                inst_results = pred.instantiate(state, action, next_state)

            results.extend(inst_results)
        return results


class Not(Predicate):
    def __init__(self, predicate: Predicate):
        self.predicate = predicate

    def __len__(self) -> int:
        return len(self.predicate)

    def instantiate(self, state: list[EnvState] | EnvState, action: list | int, next_state: list[EnvState] | EnvState) -> list[dict[str, any]]:
        window_size = len(self.predicate)
        if isinstance(state, list):
            if window_size > 1:
                results = self.predicate.instantiate(state[-window_size:], action[-window_size:], next_state[-window_size:])
            else:
                results = self.predicate.instantiate(state[-1], action[-1], next_state[-1])
        else:
            assert window_size == 1
            results = self.predicate.instantiate(state, action, next_state)

        if results:
            return []
        return [{}]


class Sequence(Predicate):
    def __init__(self, *predicates: Predicate):
        self.predicates = predicates

    def __len__(self) -> int:
        return sum(len(p) for p in self.predicates)

    def instantiate(self, state: list[EnvState], action: list, next_state: list[EnvState]) -> list[dict[str, any]]:
        results = [{}]
        start_idx = 0
        for pred in self.predicates:
            new_results = []
            window_size = len(pred)
            end_idx = start_idx + window_size
            for res in results:
                if window_size > 1:
                    inst_results = pred.instantiate(state[start_idx:end_idx], action[start_idx:end_idx], next_state[start_idx:end_idx])
                else:
                    inst_results = pred.instantiate(state[start_idx], action[start_idx], next_state[start_idx])

                for b in inst_results:
                    merged = merge_bindings(res, b)
                    if merged is not None:
                        new_results.append(merged)
            results = new_results
            start_idx = end_idx

        final_results = []
        for binding in results:
            valid = True
            new_binding = {}
            for key, value in binding.items():
                if isinstance(value, list):
                    if len(value) != 1:
                        valid = False
                        break
                    new_binding[key] = value[0]
                else:
                    new_binding[key] = value
            if valid:
                final_results.append(new_binding)
        return final_results


class Repeat(Predicate):
    def __init__(self, predicate: Predicate, repeat: int):
        self.repeat = repeat
        self.impl = Sequence(*[predicate] * repeat)

    def __len__(self) -> int:
        return len(self.impl)

    def instantiate(self, states: list[EnvState], actions: list, next_states: list[EnvState]) -> list[dict[str, any]]:
        return self.impl.instantiate(states, actions, next_states)


class IsTargetFloorMaterial(Predicate):
    def __init__(self, var: str):
        self.var = var
        
    def __len__(self) -> int:
        return 1
    
    def instantiate(self, state, action, next_state):
        px, py = next_state.player_position
        dx, dy = R.DIRECTIONS[next_state.player_direction]
        directed_px, directed_py = px+dx, py+dy
        
        if not get_is_reachable(next_state, (directed_px, directed_py)):
            return []
        
        on = Material(next_state.map[directed_px, directed_py])
        return [{}] if on == self.var else []