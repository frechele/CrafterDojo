import argparse
import sys


COMMANDS = {
    "filter": "dataset_generation.expert_behavior.filter",
    "generate": "dataset_generation.expert_behavior.generate",
    "process": "dataset_generation.expert_behavior.process",
    "train": "dataset_generation.expert_behavior.train",
}


def main_cli(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Expert behavior generation pipeline",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(COMMANDS),
        help="Pipeline stage to run.",
    )
    if not argv:
        parser.print_help()
        return

    args = parser.parse_args(argv[:1])
    remainder = argv[1:]

    module_name = COMMANDS[args.command]
    module = __import__(module_name, fromlist=["main_cli"])
    module.main_cli(remainder)


if __name__ == "__main__":
    main_cli()
