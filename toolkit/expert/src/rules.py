import jax
import jax.numpy as jnp

import craftax.craftax_classic.game_logic as gl
from craftax.craftax_classic.constants import Action, BlockType, DIRECTIONS
from craftax.craftax_classic.envs.craftax_state import EnvState, StaticEnvParams


def is_sleep(state: EnvState, action):
    return jnp.logical_and(
        jnp.logical_not(state.is_sleeping),
        action == Action.SLEEP.value
    )


def near_crafting_table(state: EnvState, action):
    return gl.is_near_block(state, BlockType.CRAFTING_TABLE.value)


def near_furnace(state: EnvState, action):
    return gl.is_near_block(state, BlockType.FURNACE.value)


def rule_craft_wood_pickaxe(state: EnvState, action):
    is_at_crafting_table = near_crafting_table(state, action)

    can_craft_wood_pickaxe = state.inventory.wood >= 1

    is_crafting_wood_pickaxe = jnp.logical_and(
        action == Action.MAKE_WOOD_PICKAXE.value,
        jnp.logical_and(can_craft_wood_pickaxe, is_at_crafting_table),
    )
    return is_crafting_wood_pickaxe


def rule_craft_stone_pickaxe(state: EnvState, action):
    is_at_crafting_table = near_crafting_table(state, action)

    can_craft_stone_pickaxe = jnp.logical_and(
        state.inventory.wood >= 1, state.inventory.stone >= 1
    )

    is_crafting_stone_pickaxe = jnp.logical_and(
        action == Action.MAKE_STONE_PICKAXE.value,
        jnp.logical_and(can_craft_stone_pickaxe, is_at_crafting_table),
    )
    return is_crafting_stone_pickaxe


def rule_craft_iron_pickaxe(state: EnvState, action):
    is_at_crafting_table = near_crafting_table(state, action)
    is_at_furnace = near_furnace(state, action)

    can_craft_iron_pickaxe = jnp.logical_and(
        state.inventory.wood >= 1,
        jnp.logical_and(
            state.inventory.stone >= 1,
            jnp.logical_and(
                state.inventory.iron >= 1,
                state.inventory.coal >= 1,
            )
        )
    )

    is_crafting_iron_pickaxe = jnp.logical_and(
        action == Action.MAKE_IRON_PICKAXE.value,
        jnp.logical_and(can_craft_iron_pickaxe, jnp.logical_and(is_at_crafting_table, is_at_furnace)),
    )
    return is_crafting_iron_pickaxe


def rule_craft_wood_sword(state: EnvState, action):
    is_at_crafting_table = near_crafting_table(state, action)

    can_craft_wood_sword = state.inventory.wood >= 1

    is_crafting_wood_sword = jnp.logical_and(
        action == Action.MAKE_WOOD_SWORD.value,
        jnp.logical_and(can_craft_wood_sword, is_at_crafting_table),
    )
    return is_crafting_wood_sword


def rule_craft_stone_sword(state: EnvState, action):
    is_at_crafting_table = near_crafting_table(state, action)

    can_craft_stone_sword = jnp.logical_and(
        state.inventory.wood >= 1, state.inventory.stone >= 1
    )

    is_crafting_stone_sword = jnp.logical_and(
        action == Action.MAKE_STONE_SWORD.value,
        jnp.logical_and(can_craft_stone_sword, is_at_crafting_table),
    )
    return is_crafting_stone_sword


def rule_craft_iron_sword(state: EnvState, action):
    is_at_crafting_table = near_crafting_table(state, action)
    is_at_furnace = near_furnace(state, action)

    can_craft_iron_sword = jnp.logical_and(
        state.inventory.wood >= 1,
        jnp.logical_and(
            state.inventory.stone >= 1,
            state.inventory.iron >= 1,
        )
    )

    is_crafting_iron_sword = jnp.logical_and(
        action == Action.MAKE_IRON_SWORD.value,
        jnp.logical_and(can_craft_iron_sword, jnp.logical_and(is_at_crafting_table, is_at_furnace)),
    )
    return is_crafting_iron_sword


def rule_attack_zombie(state: EnvState, action):
    static_params = StaticEnvParams()
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    def is_attacking_zombie_at_index(unused, zombie_index):
        in_zombie = (state.zombies.position[zombie_index] == block_position).all()
        return None, jnp.logical_and(in_zombie, state.zombies.mask[zombie_index])

    _, is_attacking_zombie_array = jax.lax.scan(
        is_attacking_zombie_at_index, None, jnp.arange(static_params.max_zombies)
    )
    is_attacking_zombie = is_attacking_zombie_array.sum() > 0
    return is_attacking_zombie


def rule_attack_cow(state: EnvState, action):
    static_params = StaticEnvParams()
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    def is_attacking_cow_at_index(unused, cow_index):
        in_cow = (state.cows.position[cow_index] == block_position).all()
        return None, jnp.logical_and(in_cow, state.cows.mask[cow_index])

    _, is_attacking_cow_array = jax.lax.scan(
        is_attacking_cow_at_index, None, jnp.arange(static_params.max_cows)
    )
    is_attacking_cow = is_attacking_cow_array.sum() > 0
    return is_attacking_cow


def rule_attack_skeleton(state: EnvState, action):
    static_params = StaticEnvParams()
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    def is_attacking_skeleton_at_index(unused, skeleton_index):
        in_skeleton = (state.skeletons.position[skeleton_index] == block_position).all()
        return None, jnp.logical_and(in_skeleton, state.skeletons.mask[skeleton_index])

    _, is_attacking_skeleton_array = jax.lax.scan(
        is_attacking_skeleton_at_index, None, jnp.arange(static_params.max_skeletons)
    )
    is_attacking_skeleton = is_attacking_skeleton_array.sum() > 0
    return is_attacking_skeleton


def target_location_validation(state: EnvState, action, condition2):
    block_position = state.player_position + DIRECTIONS[state.player_direction]
    return jnp.logical_and(gl.in_bounds(state, block_position), condition2)


def do_action_validation(state: EnvState, action, condition2):
    condition = jnp.logical_and(
        target_location_validation(state, action, condition2),
        action == Action.DO.value
    )
    return condition


def rule_mine_tree(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    can_mine_tree = True
    is_mining_tree = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.TREE.value,
        can_mine_tree
    )
    return do_action_validation(state, action, is_mining_tree)


def rule_mine_stone(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    can_mine_stone = state.inventory.wood_pickaxe > 0
    is_mining_stone = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.STONE.value,
        can_mine_stone
    )
    return do_action_validation(state, action, is_mining_stone)


def rule_mine_coal(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    can_mine_coal = state.inventory.wood_pickaxe > 0
    is_mining_coal = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.COAL.value,
        can_mine_coal
    )
    return do_action_validation(state, action, is_mining_coal)


def rule_mine_iron(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    can_mine_iron = state.inventory.stone_pickaxe > 0
    is_mining_iron = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.IRON.value,
        can_mine_iron
    )
    return do_action_validation(state, action, is_mining_iron)


def rule_mine_diamond(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    can_mine_diamond = state.inventory.iron_pickaxe > 0
    is_mining_diamond = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.DIAMOND.value,
        can_mine_diamond
    )
    return do_action_validation(state, action, is_mining_diamond)


def rule_mine_sapling(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    is_mining_sapling = state.map[block_position[0], block_position[1]] == BlockType.GRASS.value
    return do_action_validation(state, action, is_mining_sapling)


def rule_mine_water(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    is_mining_water = state.map[block_position[0], block_position[1]] == BlockType.WATER.value
    return do_action_validation(state, action, is_mining_water)


def rule_mine_plant(state: EnvState, action):
    block_position = state.player_position + DIRECTIONS[state.player_direction]

    is_mining_plant = state.map[block_position[0], block_position[1]] == BlockType.RIPE_PLANT.value
    return do_action_validation(state, action, is_mining_plant)


def rule_place_crafting_table(state: EnvState, action):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]

    crafting_table_key_down = action == Action.PLACE_TABLE.value
    has_wood = state.inventory.wood >= 2
    is_placing_crafting_table = jnp.logical_and(
        crafting_table_key_down,
        jnp.logical_and(
            jnp.logical_not(gl.is_in_wall(state, placing_block_position)), has_wood
        )
    )
    return target_location_validation(state, action, is_placing_crafting_table)


def rule_place_furnace(state: EnvState, action):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]

    furnace_key_down = action == Action.PLACE_FURNACE.value
    has_stone = state.inventory.stone > 0
    is_placing_furnace = jnp.logical_and(
        furnace_key_down,
        jnp.logical_and(
            jnp.logical_not(gl.is_in_wall(state, placing_block_position)), has_stone
        )
    )
    return target_location_validation(state, action, is_placing_furnace)


def rule_place_stone(state: EnvState, action):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]

    stone_key_down = action == Action.PLACE_STONE.value
    has_stone = state.inventory.stone > 0
    is_placing_on_valid_block = jnp.logical_or(
        state.map[placing_block_position[0], placing_block_position[1]] == BlockType.WATER.value,
        jnp.logical_not(gl.is_in_wall(state, placing_block_position))
    )
    is_placing_stone = jnp.logical_and(
        stone_key_down,
        jnp.logical_and(is_placing_on_valid_block, has_stone)
    )
    return target_location_validation(state, action, is_placing_stone)


def rule_place_sapling(state: EnvState, action):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]

    sapling_key_down = action == Action.PLACE_PLANT.value
    has_sapling = state.inventory.sapling > 0
    is_placing_sapling = jnp.logical_and(
        sapling_key_down,
        jnp.logical_and(
            state.map[placing_block_position[0], placing_block_position[1]] == BlockType.GRASS.value,
            has_sapling
        )
    )
    return target_location_validation(state, action, is_placing_sapling)


def rule_sleeping(state: EnvState, action):
    return state.is_sleeping
