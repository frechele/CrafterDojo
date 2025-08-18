from .. import env


class AchievementMeta(env.Env):
    def __init__(
        self,
        achievement: str,
        *args, **kwargs,
    ):
        super(AchievementMeta, self).__init__(*args, **kwargs)
        self.achievement = achievement

    def _check_success(self, prev_world, prev_player, cur_world, cur_player):
        if cur_player.achievements[self.achievement] > 0:
            return True
        return None
