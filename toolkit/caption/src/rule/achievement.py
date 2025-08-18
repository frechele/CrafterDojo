import src.predicate as P
from src.engine import Rule


Harvest = Rule(
    name="Harvest",
    checker=P.Collect("item"),
    templates=["obtain {item}"],
)

Construct = Rule(
    name="Construct",
    checker=P.PlaceOn("item", "material"),
    templates=["place {item} on {material}"],
)

Craft = Rule(
    name="Craft",
    checker=P.Craft("item"),
    templates=["craft {item}"],
)

Kill = Rule(
    name="Kill",
    checker=P.Kill("mob"),
    templates=["kill {mob}"],
)

Sleep = Rule(
    name="Sleep",
    checker=P.Sleep(),
    templates=["go to sleep"],
)
