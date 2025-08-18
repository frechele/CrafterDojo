from .. import env


class HarvestMeta(env.Env):
    def __init__(
        self,
        target_item: str,
        *args, **kwargs,
    ):
        super(HarvestMeta, self).__init__(*args, **kwargs)
        self.target_item = target_item

    def _check_success(self, prev_world, prev_player, cur_world, cur_player):
        prev_quantity = prev_player.inventory[self.target_item]
        cur_quantity = cur_player.inventory[self.target_item]
        if cur_quantity > prev_quantity:
            return True

        return None
