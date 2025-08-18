import src.rule.achievement as achievement
import src.rule.construction as construction
import src.rule.movement as movement
import src.rule.combat as combat


RULES = [
    # Achievement
    achievement.Harvest,
    achievement.Construct,
    achievement.Craft,
    achievement.Kill,
    achievement.Sleep,

    # Movement
    movement.Stay,
    movement.Move,
    movement.Approach,
    movement.Flee,
    movement.Explore,

    # Construction
    construction.BuildShelter,
    construction.BuildPathOver,
    construction.DigTunnel,

    # Combat
    combat.BlockAttack,
    combat.Attacked,
]

RULE_NAMES = [rule.name for rule in RULES]
MOVEMENTS = [rule.name for rule in RULES if rule.name in {"Stay", "Move Torwards", "Approach", "Flee", "Explore"}]
