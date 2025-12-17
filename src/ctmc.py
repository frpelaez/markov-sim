import sys

import ctmc_examples


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: python ctmc.py <example> ")
        print("Available examples are:")
        print("    - maquina")
        print("    - sis")
        return

    if args[0] == "maquina":
        ctmc_examples.maquina()
    elif args[0] == "sis":
        if len(args) < 2:
            print(
                "Provide a value for beta and a flag `--global`/`-g` to treat the value as global or `--personal`/`-p` to treat it as personal"
            )
            print(
                "You can also provide arguments like `N=100` and `T=30` for population size and max time"
            )
            return
        beta = float(args[1])
        global_beta = (
            True if len(args) >= 3 and args[2] in ["--global", "-g"] else False
        )
        if len(args) >= 4:
            if len(pair := args[3].split("=")) == 2:
                par, value = pair
            else:
                print(
                    f"Invalid argument notation `{args[3]}`. Make sure it follows the pattern `(N or T)=<number>`"
                )
                return
            if len(args) >= 5:
                if len(pair2 := args[4].split("=")) == 2:
                    par2, value2 = pair2
                else:
                    print(
                        f"Invalid argument notation `{args[4]}`. Make sure it follows the pattern `(N or T)=<number>`"
                    )
                    return
                if par == "N" and par2 == "T":
                    N = int(value)
                    T = float(value2)
                elif par == "T" and par2 == "N":
                    T = float(value)
                    N = int(value2)
                ctmc_examples.sis(beta, global_beta, N=N, T=T)
                return
            if par == "N":
                N = int(value)
                ctmc_examples.sis(beta, global_beta, N=N)
                return
            elif par == "T":
                T = float(value)
                ctmc_examples.sis(beta, global_beta, T=T)
                return
        ctmc_examples.sis(beta, global_beta)
    else:
        print(f"Invalid example '{args[0]}'. Available examples are:")
        print("    - maquina")
        print("    - sis")


if __name__ == "__main__":
    main()
