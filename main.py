import argparse

from cvlabkit.core.config import Config
from cvlabkit.core.creator import Creator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--fast", "-f", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    valid_trials = []
    has_missing_key = False

    # Expand the base config into all combinations (grid search)
    configs = Config(args.config).expand()
    print(f"Grid search: {len(configs)} configuration(s)")

    # Step 1: Validate each configuration using dry_run
    for cfg in configs:
        if args.fast:
            cfg.proxy.deactivate()
            valid_trials.append(cfg)
        else:
            # try:
            agent = Creator(cfg).agent()
            agent.dry_run()

            # If config is incomplete, mark it for template generation
            if cfg.has_missing():
                has_missing_key = True
            else:
                valid_trials.append(cfg)
            # except Exception as e:
                # print(f"[dry_run failed] {e}")
                # return

    # Step 2: Output a config template and halt if any missing keys were found
    if has_missing_key:
        cfg.dump_template("templates/generated.yaml")
        print("Missing keys found. Template generated. Rerun after filling config.")
        return

    # Step 3: Run training for all validated configs
    for cfg in valid_trials:
        create = Creator(cfg)
        agent = create.agent()
        agent.fit()


if __name__ == "__main__":
    main()
