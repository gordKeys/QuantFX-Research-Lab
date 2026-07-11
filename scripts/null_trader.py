from bootstrap import add_project_root
add_project_root()

import sys

from live_runner import main


def run():
    if "--mirror-signals" not in sys.argv:
        sys.argv.append("--mirror-signals")
    print("\n=== NULL TRADER MODE ===")
    print("This run mirrors the main bot: buys become sells, sells become buys.")
    print("========================\n")
    main()


if __name__ == "__main__":
    run()
