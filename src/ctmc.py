import sys

import ctmc_examples


def main() -> None:
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python ctmc.py <example> ")
        print("Available examples are:")
        print("    - maquina")
        print("    - sis")
        return

    if args[0] == "maquina":
        ctmc_examples.maquina()
    elif args[0] == "sis":
        ctmc_examples.sis()
    else:
        print(f"Invalid example '{args[0]}'. Available examples are:")
        print("    - maquina")
        print("    - sis")


if __name__ == "__main__":
    main()
