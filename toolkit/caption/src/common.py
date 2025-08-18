import numpy as np
import heapq
from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import Any, Tuple

WINDOW_SIZE = 5

@dataclass
class Inventory:
    wood: int = 0
    stone: int = 0
    coal: int = 0
    iron: int = 0
    diamond: int = 0
    sapling: int = 0
    wood_pickaxe: int = 0
    stone_pickaxe: int = 0
    iron_pickaxe: int = 0
    wood_sword: int = 0
    stone_sword: int = 0
    iron_sword: int = 0


@dataclass
class Mobs:
    position: np.ndarray
    health: int
    mask: bool
    attack_cooldown: int


@dataclass
class EnvState:
    map: np.ndarray
    mob_map: np.ndarray

    player_position: np.ndarray
    player_direction: int

    # Intrinsics
    player_health: int
    player_food: int
    player_drink: int
    player_energy: int
    is_sleeping: bool

    # Second order intrinsics
    player_recover: float
    player_hunger: float
    player_thirst: float
    player_fatigue: float

    inventory: Inventory

    zombies: Mobs
    cows: Mobs
    skeletons: Mobs
    arrows: Mobs
    arrow_directions: np.ndarray

    growing_plants_positions: np.ndarray
    growing_plants_age: np.ndarray
    growing_plants_mask: np.ndarray

    light_level: float

    achievements: np.ndarray

    state_rng: Any

    timestep: int

    fractal_noise_ans: tuple[int, int, int, int] = (None, None, None, None)


@dataclass
class EnvParams:
    max_timesteps: int = 10000
    day_length: int = 300

    always_diamond: bool = True

    zombie_health: int = 5
    cow_health: int = 3
    skeleton_health: int = 3

    mob_despawn_distance: int = 14

    spawn_cow_chance: float = 0.1
    spawn_zombie_base_chance: float = 0.02
    spawn_zombie_night_chance: float = 0.1
    spawn_skeleton_chance: float = 0.05

    fractal_noise_ans: tuple[int, int, int, int] = (None, None, None, None)


@dataclass
class StaticEnvParams:
    map_size: Tuple[int, int] = (64, 64)

    # Mobs
    max_zombies: int = 3
    max_cows: int = 3
    max_growing_plants: int = 10
    max_skeletons: int = 2
    max_arrows: int = 3
    
    max_drink: int = 9


# GAME CONSTANTS
OBS_DIM = (7, 9)
MAX_OBS_DIM = max(OBS_DIM)
assert OBS_DIM[0] % 2 == 1 and OBS_DIM[1] % 2 == 1
BLOCK_PIXEL_SIZE_HUMAN = 64
BLOCK_PIXEL_SIZE_IMG = 16
BLOCK_PIXEL_SIZE_AGENT = 7
INVENTORY_OBS_HEIGHT = 2

# ENUMS
class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16


class Action(Enum):
    NOOP = 0  #
    LEFT = 1  # a
    RIGHT = 2  # d
    UP = 3  # w
    DOWN = 4  # s
    DO = 5  # space
    SLEEP = 6  # tab
    PLACE_STONE = 7  # r
    PLACE_TABLE = 8  # t
    PLACE_FURNACE = 9  # f
    PLACE_PLANT = 10  # p
    MAKE_WOOD_PICKAXE = 11  # 1
    MAKE_STONE_PICKAXE = 12  # 2
    MAKE_IRON_PICKAXE = 13  # 3
    MAKE_WOOD_SWORD = 14  # 4
    MAKE_STONE_SWORD = 15  # 5
    MAKE_IRON_SWORD = 16  # 6


# GAME MECHANICS
DIRECTIONS = np.concatenate(
    (
        np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int32),
        np.zeros((11, 2), dtype=np.int32),
    ),
    axis=0,
)

CLOSE_BLOCKS = np.array(
    [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    dtype=np.int32,
)

# Can't walk through these
SOLID_BLOCKS = np.array(
    [
        BlockType.WATER.value,
        BlockType.STONE.value,
        BlockType.TREE.value,
        BlockType.COAL.value,
        BlockType.IRON.value,
        BlockType.DIAMOND.value,
        BlockType.CRAFTING_TABLE.value,
        BlockType.FURNACE.value,
        BlockType.PLANT.value,
        BlockType.RIPE_PLANT.value,
    ],
    dtype=np.int32,
)


# ACHIEVEMENTS
class Achievement(Enum):
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    MAKE_WOOD_PICKAXE = 5
    MAKE_WOOD_SWORD = 6
    PLACE_PLANT = 7
    DEFEAT_ZOMBIE = 8
    COLLECT_STONE = 9
    PLACE_STONE = 10
    EAT_PLANT = 11
    DEFEAT_SKELETON = 12
    MAKE_STONE_PICKAXE = 13
    MAKE_STONE_SWORD = 14
    WAKE_UP = 15
    PLACE_FURNACE = 16
    COLLECT_COAL = 17
    COLLECT_IRON = 18
    COLLECT_DIAMOND = 19
    MAKE_IRON_PICKAXE = 20
    MAKE_IRON_SWORD = 21


def in_bounds(state, position):
    in_bounds_x = np.logical_and(0 <= position[0], position[0] < state.map.shape[0])
    in_bounds_y = np.logical_and(0 <= position[1], position[1] < state.map.shape[1])
    return np.logical_and(in_bounds_x, in_bounds_y)


def is_in_wall(state, position):
    is_in_block = np.zeros(len(SOLID_BLOCKS))
    for i in range(len(SOLID_BLOCKS)):
        is_in_block[i] = state.map[position[0], position[1]] == SOLID_BLOCKS[i]

    return is_in_block.sum() > 0


def is_in_mob(state, position):
    return np.logical_or(
        state.mob_map[position[0], position[1]],
        (state.player_position == position).all(),
    )


def is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(state, position):
    pos_in_bounds = in_bounds(state, position)
    in_wall = is_in_wall(state, position)
    in_mob = is_in_mob(state, position)
    in_lava = state.map[position[0], position[1]] == BlockType.LAVA.value
    valid_move = np.logical_and(pos_in_bounds, np.logical_not(in_wall))
    valid_move = np.logical_and(valid_move, np.logical_not(in_mob))
    valid_move = np.logical_and(valid_move, np.logical_not(in_lava))

    return valid_move


def is_near_block(state: EnvState, block_type):
    is_block = np.zeros(len(CLOSE_BLOCKS))
    for i in range(len(CLOSE_BLOCKS)):
        pos = state.player_position + CLOSE_BLOCKS[i]
        if not in_bounds(state, pos):
            continue

        is_correct_block = state.map[pos[0], pos[1]] == block_type
        is_block[i] = is_correct_block
    return is_block.sum() > 0


def is_sleep(state: EnvState, action):
    return np.logical_and(
        np.logical_not(state.is_sleeping),
        action == Action.SLEEP.value
    )


def near_crafting_table(state: EnvState, action):
    return is_near_block(state, BlockType.CRAFTING_TABLE.value)


def near_furnace(state: EnvState, action):
    return is_near_block(state, BlockType.FURNACE.value)


def get_sight_range(state: EnvState):
    px, py = state.player_position
    DX = OBS_DIM[1] // 2
    DY = OBS_DIM[0] // 2

    sight_upper_left = (px - DY, py - DX)
    sight_lower_right = (px + DY, py + DX)

    return sight_upper_left, sight_lower_right


def is_in_sight(state: EnvState, pos: np.ndarray) -> bool:
    sight_upper_left, sight_lower_right = get_sight_range(state)

    sulx, suly = sight_upper_left
    slrx, slry = sight_lower_right

    return sulx <= pos[0] <= slrx and suly <= pos[1] <= slry


def visible_mobs_by_type(state: EnvState, mobs) -> list[tuple[int, np.ndarray]]:
    visible = []
    for i, (mob_pos, mask) in enumerate(zip(mobs.position, mobs.mask)):
        if not mask:
            continue

        if is_in_sight(state, mob_pos):
            visible.append((i, mob_pos))
    return visible

def get_diff_exit(state: EnvState, next_state: EnvState, direction: int):
    px,py = state.player_position
    next_px,next_py = next_state.player_position
    
    exit_count, next_exit_count, diag_count = 0, 0, 0
    dir = [(-1,0), (1,0), (0,-1), (0,1)]
    
    for dx,dy in dir:
        new_px, new_py = px+dx, py+dy
        new_next_px, new_next_py = next_px+dx, next_py+dy
        
        exit_count += get_is_reachable(state, (new_px, new_py))
        next_exit_count += get_is_reachable(next_state, (new_next_px, new_next_py))
        
    if direction == Action.LEFT.value: # Heading west
        diag_dir = [(-1,-1), (1,-1)]
    elif direction == Action.RIGHT.value: # Heading east
        diag_dir = [(1,1), (-1,1)]
    elif direction == Action.UP.value: # Heading  north
        diag_dir = [(-1,1), (-1,-1)]
    elif direction == Action.DOWN.value: # Heading south
        diag_dir = [(1,1), (1,-1)]
        
    for diag_dx,diag_dy in diag_dir:
        new_px, new_py = px+diag_dx, py+diag_dy 
        diag_count += get_is_reachable(state, (new_px, new_py))
    
    if (next_exit_count != exit_count) and (next_exit_count == 1 or exit_count == 1) and (not diag_count):
        return True
    return False

def get_is_reachable(state: EnvState, pos):
    pos_in_bounds = in_bounds(state, pos)
    if not pos_in_bounds:
        return False
    
    in_wall = is_in_wall(state, pos)
    in_lava = state.map[pos[0], pos[1]] == BlockType.LAVA.value
    
    is_valid = np.logical_and(pos_in_bounds, np.logical_not(in_wall))
    is_valid = np.logical_and(is_valid, np.logical_not(in_lava))
        
    return is_valid

def get_reachable_positions(state: EnvState) -> set[tuple[int, int]]:
    reachable = set()
    checked = set()
    queue = deque()
    queue.append(state.player_position)
    checked.add(tuple(state.player_position))
    while queue:
        pos = queue.popleft()
        is_valid = get_is_reachable(state, pos)

        if is_valid:
            reachable.add(tuple(pos))

            checks = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in checks:
                npos = pos + np.array([dx, dy])
                npos_t = tuple(npos)
                in_sight = is_in_sight(state, npos)
                if (in_bounds(state, npos) and in_sight and
                    npos_t not in checked):
                    queue.append(npos)
                    checked.add(npos_t)

    return reachable


def get_distance_from_agent(state: EnvState, position: np.ndarray, blocking: np.ndarray = None) -> float:
    start_pos = tuple(state.player_position)
    goal_pos = tuple(position)

    distances = {start_pos: 0}
    pq = [(0, start_pos)]
    
    while pq:
        current_dist, pos = heapq.heappop(pq)

        if current_dist > distances[pos]:
            continue

        if pos == goal_pos:
            return current_dist

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            next_pos = (nx, ny)

            if blocking is not None and (blocking[0] == nx and blocking[1] == ny):
                continue
            if not in_bounds(state, next_pos):
                continue
            if is_in_wall(state, next_pos):
                continue
            if state.map[nx, ny] == BlockType.LAVA.value:
                continue
            if not is_in_sight(state, next_pos):
                continue

            new_dist = current_dist + 1

            if next_pos not in distances or new_dist < distances[next_pos]:
                distances[next_pos] = new_dist
                heapq.heappush(pq, (new_dist, next_pos))

    return float('inf')


def manhattan_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    return np.sum(np.abs(pos1 - pos2))


def euclidean_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    return np.linalg.norm(pos1 - pos2)


##################################################
#  Rules
##################################################

def rule_craft_wood_pickaxe(state: EnvState, action):
    is_at_crafting_table = near_crafting_table(state, action)

    can_craft_wood_pickaxe = state.inventory.wood >= 1

    is_crafting_wood_pickaxe = np.logical_and(
        action == Action.MAKE_WOOD_PICKAXE.value,
        np.logical_and(can_craft_wood_pickaxe, is_at_crafting_table),
    )
    return is_crafting_wood_pickaxe


def rule_craft_stone_pickaxe(state: EnvState, action):
    is_at_crafting_table = near_crafting_table(state, action)

    can_craft_stone_pickaxe = np.logical_and(
        state.inventory.wood >= 1, state.inventory.stone >= 1
    )

    is_crafting_stone_pickaxe = np.logical_and(
        action == Action.MAKE_STONE_PICKAXE.value,
        np.logical_and(can_craft_stone_pickaxe, is_at_crafting_table),
    )
    return is_crafting_stone_pickaxe


def rule_craft_iron_pickaxe(state: EnvState, action):
    is_at_crafting_table = near_crafting_table(state, action)
    is_at_furnace = near_furnace(state, action)

    can_craft_iron_pickaxe = np.logical_and(
        state.inventory.wood >= 1,
        np.logical_and(
            state.inventory.stone >= 1,
            np.logical_and(
                state.inventory.iron >= 1,
                state.inventory.coal >= 1,
            )
        )
    )

    is_crafting_iron_pickaxe = np.logical_and(
        action == Action.MAKE_IRON_PICKAXE.value,
        np.logical_and(can_craft_iron_pickaxe, np.logical_and(is_at_crafting_table, is_at_furnace)),
    )
    return is_crafting_iron_pickaxe


def rule_craft_wood_sword(state: EnvState, action):
    is_at_crafting_table = near_crafting_table(state, action)

    can_craft_wood_sword = state.inventory.wood >= 1

    is_crafting_wood_sword = np.logical_and(
        action == Action.MAKE_WOOD_SWORD.value,
        np.logical_and(can_craft_wood_sword, is_at_crafting_table),
    )
    return is_crafting_wood_sword


def rule_craft_stone_sword(state: EnvState, action):
    is_at_crafting_table = near_crafting_table(state, action)

    can_craft_stone_sword = np.logical_and(
        state.inventory.wood >= 1, state.inventory.stone >= 1
    )

    is_crafting_stone_sword = np.logical_and(
        action == Action.MAKE_STONE_SWORD.value,
        np.logical_and(can_craft_stone_sword, is_at_crafting_table),
    )
    return is_crafting_stone_sword


def rule_craft_iron_sword(state: EnvState, action):
    is_at_crafting_table = near_crafting_table(state, action)
    is_at_furnace = near_furnace(state, action)

    can_craft_iron_sword = np.logical_and(
        state.inventory.wood >= 1,
        np.logical_and(
            state.inventory.stone >= 1,
            state.inventory.iron >= 1,
        )
    )

    is_crafting_iron_sword = np.logical_and(
        action == Action.MAKE_IRON_SWORD.value,
        np.logical_and(can_craft_iron_sword, np.logical_and(is_at_crafting_table, is_at_furnace)),
    )
    return is_crafting_iron_sword


def rule_attack_zombie(state: EnvState, action, next_state: EnvState):
    if action != Action.DO.value:
        return False

    static_params = StaticEnvParams()
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    is_attacking_zombie = False
    for i in range(static_params.max_zombies):
        in_zombie = (state.zombies.position[i] == block_position).all()
        is_attacking_zombie = np.logical_and(in_zombie, state.zombies.mask[i])
        is_attacking_zombie = np.logical_and(is_attacking_zombie, np.logical_not(next_state.zombies.mask[i]))
        if is_attacking_zombie:
            break

    return is_attacking_zombie


def rule_attack_cow(state: EnvState, action, next_state: EnvState):
    if action != Action.DO.value:
        return False

    static_params = StaticEnvParams()
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    is_attacking_cow = False
    for i in range(static_params.max_cows):
        in_cow = (state.cows.position[i] == block_position).all()
        is_attacking_cow = np.logical_and(in_cow, state.cows.mask[i])
        is_attacking_cow = np.logical_and(is_attacking_cow, np.logical_not(next_state.cows.mask[i]))
        if is_attacking_cow:
            break

    return is_attacking_cow


def rule_attack_skeleton(state: EnvState, action, next_state: EnvState):
    if action != Action.DO.value:
        return False

    static_params = StaticEnvParams()
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    is_attacking_skeleton = False
    for i in range(static_params.max_skeletons):
        in_skeleton = (state.skeletons.position[i] == block_position).all()
        is_attacking_skeleton = np.logical_and(in_skeleton, state.skeletons.mask[i])
        is_attacking_skeleton = np.logical_and(is_attacking_skeleton, np.logical_not(next_state.skeletons.mask[i]))
        if is_attacking_skeleton:
            break

    return is_attacking_skeleton


def rule_mine_tree(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]
    if not in_bounds(state, block_position):
        return False

    can_mine_tree = True
    is_mining_tree = np.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.TREE.value,
        can_mine_tree
    )
    return np.logical_and(
        is_mining_tree,
        action == Action.DO.value
    )


def rule_mine_stone(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]
    if not in_bounds(state, block_position):
        return False

    can_mine_stone = state.inventory.wood_pickaxe > 0
    is_mining_stone = np.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.STONE.value,
        can_mine_stone
    )
    return np.logical_and(
        is_mining_stone,
        action == Action.DO.value
    )


def rule_mine_coal(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]
    if not in_bounds(state, block_position):
        return False

    can_mine_coal = state.inventory.wood_pickaxe > 0
    is_mining_coal = np.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.COAL.value,
        can_mine_coal
    )
    return np.logical_and(
        is_mining_coal,
        action == Action.DO.value
    )


def rule_mine_iron(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]
    if not in_bounds(state, block_position):
        return False

    can_mine_iron = state.inventory.stone_pickaxe > 0
    is_mining_iron = np.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.IRON.value,
        can_mine_iron
    )
    return np.logical_and(
        is_mining_iron,
        action == Action.DO.value
    )


def rule_mine_diamond(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]
    if not in_bounds(state, block_position):
        return False

    can_mine_diamond = state.inventory.iron_pickaxe > 0
    is_mining_diamond = np.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.DIAMOND.value,
        can_mine_diamond
    )
    return np.logical_and(
        is_mining_diamond,
        action == Action.DO.value
    )


def rule_mine_sapling(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]
    if not in_bounds(state, block_position):
        return False

    is_mining_sapling = state.map[block_position[0], block_position[1]] == BlockType.GRASS.value
    return np.logical_and(
        is_mining_sapling,
        action == Action.DO.value
    )


def rule_mine_water(state: EnvState, action):
    static_params = StaticEnvParams()
    block_position = state.player_position + DIRECTIONS[state.player_direction]
    if not in_bounds(state, block_position):
        return False
    
    is_mining_water = state.map[block_position[0], block_position[1]] == BlockType.WATER.value
    is_mining_water = np.logical_and(
        is_mining_water,
        state.player_drink < static_params.max_drink
    )
    
    return np.logical_and(
        is_mining_water,
        action == Action.DO.value
    )


def rule_mine_plant(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]
    if not in_bounds(state, block_position):
        return False

    is_mining_plant = state.map[block_position[0], block_position[1]] == BlockType.RIPE_PLANT.value
    return np.logical_and(
        is_mining_plant,
        action == Action.DO.value
    )


def rule_place_crafting_table(state: EnvState, action):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]
    if not in_bounds(state, placing_block_position):
        return False

    crafting_table_key_down = action == Action.PLACE_TABLE.value
    has_wood = state.inventory.wood >= 2
    is_placing_crafting_table = np.logical_and(
        crafting_table_key_down,
        np.logical_and(
            is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(state, placing_block_position), has_wood
        )
    )
    return is_placing_crafting_table


def rule_place_furnace(state: EnvState, action):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]
    if not in_bounds(state, placing_block_position):
        return False

    furnace_key_down = action == Action.PLACE_FURNACE.value
    has_stone = state.inventory.stone > 0
    is_placing_furnace = np.logical_and(
        furnace_key_down,
        np.logical_and(
            is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(state, placing_block_position), has_stone
        )
    )
    return is_placing_furnace


def rule_place_stone(state: EnvState, action):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]
    if not in_bounds(state, placing_block_position):
        return False

    stone_key_down = action == Action.PLACE_STONE.value
    has_stone = state.inventory.stone > 0
    is_water_or_lava = np.logical_or(
        state.map[placing_block_position[0], placing_block_position[1]] == BlockType.WATER.value,
        state.map[placing_block_position[0], placing_block_position[1]] == BlockType.LAVA.value,
    )
    is_placing_on_valid_block = np.logical_or(
        is_water_or_lava,
        is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(state, placing_block_position)
    )
    is_placing_stone = np.logical_and(
        stone_key_down,
        np.logical_and(is_placing_on_valid_block, has_stone)
    )
    return is_placing_stone


def rule_place_sapling(state: EnvState, action):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]
    if not in_bounds(state, placing_block_position):
        return False

    sapling_key_down = action == Action.PLACE_PLANT.value
    has_sapling = state.inventory.sapling > 0
    is_valid_sapling = np.logical_and(
        state.map[placing_block_position[0], placing_block_position[1]] == BlockType.GRASS.value,
        np.logical_not(is_in_mob(state, placing_block_position))
    )
    is_placing_sapling = np.logical_and(
        sapling_key_down,
        np.logical_and(
            is_valid_sapling,
            has_sapling
        )
    )
    return is_placing_sapling


def rule_sleeping(state: EnvState, action):
    return state.is_sleeping


def rule_attacked_by_zombie(state: EnvState, action, next_state: EnvState):
    is_attacking_player = (
        np.sum(np.abs(state.zombies.position - state.player_position), axis=-1)
        == 1
    )
    is_attacking_player = np.logical_and(
        is_attacking_player, state.zombies.attack_cooldown <= 0
    )
    is_attacking_player = np.logical_and(
        is_attacking_player, state.zombies.mask
    )
    is_attacking_player = np.logical_and(
        is_attacking_player, next_state.zombies.mask
    )
    return np.any(is_attacking_player)


def rule_attacked_by_skeleton(state: EnvState, action, next_state: EnvState):
    proposed_position = (
        state.arrows.position + state.arrow_directions
    )

    proposed_position_in_player = (
        np.sum(np.abs(proposed_position - state.player_position), axis=-1)
        == 0
    )

    hit_player = np.logical_and(
        proposed_position_in_player, state.arrows.mask
    )

    return np.any(hit_player)
