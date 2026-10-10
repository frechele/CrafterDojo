import argparse
import sys


COMMANDS = {
    "caption": "dataset_generation.caption.main",
    "caption-gather": "dataset_generation.caption.gather",
    "caption-sample": "dataset_generation.caption.sampling",
    "caption-balance": "dataset_generation.caption.balance",
    "expert": "dataset_generation.expert_behavior.main",
}


def main_cli(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="CrafterDojo dataset generation entrypoints",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(COMMANDS),
        help="Which dataset generation pipeline to run.",
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
