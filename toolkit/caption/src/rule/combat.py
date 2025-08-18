import src.predicate as P
from src.engine import Rule


BlockAttack = Rule(
    name="Block Attack",
    checker=P.And(
        P.PlaceOn("item", "material"),
        P.BlockAttack("mob"),
    ),
    templates=["block attack from {mob} with {item}"],
)

Attacked = Rule(
    name="Attacked",
    checker=P.AttackedBy("mob"),
    templates=["attacked by {mob}"],
)
