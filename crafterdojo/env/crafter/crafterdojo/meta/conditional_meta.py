from .. import env


class ConditionalAchievementMeta(env.Env):
    def __init__(
        self,
        first: str,
        second: str,
        *args, **kwargs,
    ):
        super(ConditionalAchievementMeta, self).__init__(*args, **kwargs)
        self.first_achievement = first
        self.second_achievement = second

    def _on_reset(self):
        self.second_at_first_done = None
        self.is_first_for_first = True

    def _check_success(self, prev_world, prev_player, cur_world, cur_player):
        n_first_done = cur_player.achievements[self.first_achievement]
        n_second_done = cur_player.achievements[self.second_achievement]

        first_done = n_first_done > 0

        if self.is_first_for_first and first_done:
            self.second_at_first_done = n_second_done
            self.is_first_for_first = False

        if first_done and n_second_done > self.second_at_first_done:
            return True

        return None


class ConditionalAchievement3Meta(env.Env):
    def __init__(
        self,
        first: str,
        second: str,
        third: str,
        *args, **kwargs,
    ):
        super(ConditionalAchievement3Meta, self).__init__(*args, **kwargs)
        self.first_achievement = first
        self.second_achievement = second
        self.third_achievement = third

    def _on_reset(self):
        self.second_at_first_done = None
        self.third_at_second_done = None
        self.phase = 0

    def _check_success(self, prev_world, prev_player, cur_world, cur_player):
        n_first_done = cur_player.achievements[self.first_achievement]
        n_second_done = cur_player.achievements[self.second_achievement]
        n_third_done = cur_player.achievements[self.third_achievement]

        first_done = n_first_done > 0

        if self.phase == 0 and first_done:
            self.second_at_first_done = n_second_done
            self.phase = 1

        if self.phase == 1 and n_second_done > self.second_at_first_done:
            self.third_at_second_done = n_third_done
            self.phase = 2

        if self.phase == 2 and n_third_done > self.third_at_second_done:
            return True

        return None
