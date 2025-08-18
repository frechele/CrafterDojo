import src.predicate as P
from src.common import WINDOW_SIZE
from src.engine import Rule

Stay = Rule(
    name="Stay",
    checker=P.Repeat(
        P.And(
            P.Not(P.Sleeping()),
            P.NoOp(),
        ),
        WINDOW_SIZE
    ),
    templates=["stay"],
)

Move = Rule(
    name="Move Towards",
    checker=P.Repeat(
        P.Move("dir"), WINDOW_SIZE
    ),
    templates=["move to {dir}"],
)

Approach = Rule(
    name="Approach",
    checker=P.Approach("mob", WINDOW_SIZE),
    templates=["approach {mob}"],
)

Flee = Rule(
    name="Flee",
    checker=P.Flee("mob", WINDOW_SIZE),
    templates=["flee from {mob}"],
)

Explore = Rule(
    name="Explore",
    checker=P.Sequence(
        *[P.Move(f"dir_{i}") for i in range(WINDOW_SIZE)]
    ),
    templates=["go explore"],
)
