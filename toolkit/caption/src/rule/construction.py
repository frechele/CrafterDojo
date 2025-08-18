import src.predicate as P
from src.engine import Rule
from src.common import WINDOW_SIZE
from src.predicate import Item, Material


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
    templates=["place {item} to build shelter"],
    pivot_offset=-1,
)

BuildPathOver = Rule(
    name = "Build Path Over",
    checker=P.Sequence(
        P.Not(P.IsTargetFloorMaterial(Material.PATH)),
        P.PlaceOn(Item.STONE, "material"),
        P.Collect(Item.STONE),
    ),
    templates=["build path over {material}"],
)

DigTunnel = Rule(
    name="Dig Tunnel",
    checker=P.DigTunnel(WINDOW_SIZE),
    templates=["dig a tunnel"],
)
