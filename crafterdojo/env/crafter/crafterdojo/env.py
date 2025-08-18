import copy
import numpy as np

from crafterdojo.env.crafter.crafter import constants
from crafterdojo.env.crafter.crafter import engine
from crafterdojo.env.crafter.crafter import objects
from crafterdojo.env.crafter.crafter import worldgen
from crafterdojo.env.crafter.crafter.env import Env as CrafterBaseEnv


class Env(CrafterBaseEnv):
    def __init__(
        self,
        area=(64, 64), view=(9, 9), size=(144, 144),
        length=1000, seed=None,
        initial_inventory=None,
        fixed_time=False,
        peaceful=False,
        easy_survive=False,
        no_cow=False,
    ):
        view = np.array(view if hasattr(view, '__len__') else (view, view))
        size = np.array(size if hasattr(size, '__len__') else (size, size))
        seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
        self._area = area
        self._view = view
        self._size = size
        self._length = length
        self._seed = seed
        self._episode = 0
        self._world = engine.World(area, constants.materials, (12, 12))
        self._textures = engine.Textures(constants.root / 'assets')
        item_rows = int(np.ceil(len(constants.items) / view[0]))
        self._local_view = engine.LocalView(
            self._world, self._textures, [view[0], view[1] - item_rows])
        self._item_view = engine.ItemView(
            self._textures, [view[0], item_rows])
        self._sem_view = engine.SemanticView(self._world, [
            objects.Player, objects.Cow, objects.Zombie,
            objects.Skeleton, objects.Arrow, objects.Plant])
        self._step = None
        self._player = None
        self._last_health = None
        # Some libraries expect these attributes to be set.
        self.reward_range = None
        self.metadata = None
        self._last_obs = None
        self._last_info = None

        # Task-specific attributes
        self._initial_inventory = initial_inventory
        self._fixed_time = fixed_time
        self._peaceful = peaceful
        self._easy_survive = easy_survive
        self._no_cow = no_cow

    def reset(self, *args, **kwargs):
        center = (self._world.area[0] // 2, self._world.area[1] // 2)
        self._episode += 1
        self._step = 0
        self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
        self._update_time()

        self._player = objects.Player(self._world, center, self._initial_inventory)
        self._last_health = self._player.health
        self._world.add(self._player)

        worldgen.generate_world(self._world, self._player)

        if self._peaceful:
            self._remove_all_hostiles()

        if self._no_cow:
            self._remove_all_cows()

        self._last_obs = self._obs()
        self._last_info = {
            'inventory': self._player.inventory.copy(),
            'semantic': self._sem_view(),
            'player_pos': self._player.pos.copy(),
            **self._get_nearby_info()
        }

        self._on_reset()
        return self._last_obs, self._last_info.copy()

    def step(self, action):
        prev_world = copy.deepcopy(self._world)
        prev_player = copy.deepcopy(self._player)

        self._step += 1
        if not self._fixed_time:
            self._update_time()
        self._player.action = constants.actions[action]

        for obj in self._world.objects:
            if self._player.distance(obj) < 2 * max(self._view):
                obj.update()
        if self._step % 10 == 0:
            for chunk, objs in self._world.chunks.items():
                self._balance_chunk(chunk, objs)

        if self._peaceful:
            self._remove_all_hostiles()

        if self._no_cow:
            self._remove_all_cows()

        if self._easy_survive:
            self._player.health = max(self._player.health, prev_player.health)

        obs = self._obs()

        success = self._check_success(prev_world, prev_player, self._world, self._player)
        success_done = success is not None
        reward = 1 if success == True else 0

        self._last_health = self._player.health
        dead = self._player.health <= 0
        over = self._length and self._step >= self._length
        done = success_done or dead or over
        info = {
            'inventory': self._player.inventory.copy(),
            'semantic': self._sem_view(),
            'player_pos': self._player.pos,
            **self._get_nearby_info()
        }

        if done:
            info['is_success'] = True if success == True else False

        self._last_obs = obs
        self._last_info = info

        return obs, reward, done, info

    def _get_nearby_info(self):
        nearby, _ = self._world.nearby(self._player.pos, 1)
        nearby_table = "table" in nearby
        nearby_furnace = "furnace" in nearby

        return {
            "nearby_table": nearby_table,
            "nearby_furnace": nearby_furnace,
        }

    def _on_reset(self):
        pass

    def _check_success(self, prev_world, prev_player, cur_world, cur_player):
        return None

    @property
    def get_last_obs(self):
        return self._last_obs

    @property
    def get_last_info(self):
        return self._last_info

    def _remove_all_hostiles(self):
        world_objs = self._world.objects
        for obj in world_objs:
            if isinstance(obj, objects.Zombie) or isinstance(obj, objects.Skeleton):
                self._world.remove(obj)

    def _remove_all_cows(self):
        world_objs = self._world.objects
        for obj in world_objs:
            if isinstance(obj, objects.Cow):
                self._world.remove(obj)
