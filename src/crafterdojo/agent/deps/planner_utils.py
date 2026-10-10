def inventory_description(info: dict) -> str:
    inventory = info.get("inventory", {})
    inventory = {k: v for k, v in inventory.items() if v > 0}

    if not inventory:
        return "nothing"

    return ", ".join([f"{v} {k}" for k, v in inventory.items()])


def nearby_description(info: dict) -> str:
    nearby_table = info.get("nearby_table", False)
    nearby_furnace = info.get("nearby_furnace", False)

    if not nearby_table and not nearby_furnace:
        return " You cannot see a table or furnace nearby."

    desc = " You can see "
    nearbys = []
    if nearby_table:
        nearbys.append("a table")
    if nearby_furnace:
        nearbys.append("a furnace")

    desc += " and ".join(nearbys) + " nearby."
    return desc


def resolve_template(template_str: str, **kargs) -> str:
    for k, v in kargs.items():
        placeholder = '{' + k + '}'
        template_str = template_str.replace(placeholder, str(v))
    return template_str
