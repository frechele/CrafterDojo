from dataset_generation.caption import predicate as P
from dataset_generation.caption.common import WINDOW_SIZE
from dataset_generation.caption.engine import Rule
from dataset_generation.caption.predicate import Item, Material

BuildShelter = Rule(
    name="Build Shelter",
    checker=P.Sequence(
        P.And(
            P.PlaceOn("item", "material"),
            P.Not(P.Encapsulated()),
        ),
        P.And(
            P.Not(P.MobNearby("mob")),
            P.Encapsulated(),
        ),
    ),
    templates=["place {item} to build shelter", "build shelter"],
    pivot_offset=-1,
)

BuildPathOver = Rule(
    name="Build Path Over",
    checker=P.Sequence(
        P.Not(P.IsTargetFloorMaterial(Material.PATH)),
        P.PlaceOn(Item.STONE, "material"),
        P.Collect(Item.STONE),
    ),
    templates=["build path over {material}", "build path"],
)

DigTunnel = Rule(
    name="Dig Tunnel",
    checker=P.DigTunnel(WINDOW_SIZE),
    templates=["dig a tunnel"],
)
