import argparse

from cvlabkit.core.config import Config
from cvlabkit.core.creator import Creator


def parse_args():
    """Parse command line arguments for the training script.
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--fast", "-f", action="store_true")
    return p.parse_args()


def main():
    """Main function to run the training process.
    This Process parses command line arguments, expands the configuration,
    and validates all configurations if not in fast mode,
    or runs the training directly if in fast mode.
    """
    args = parse_args()
    approved_trials = []
    has_missing_key = False

    # Expand the base config into all combinations (Grid search)
    configs = Config(args.config).expand()
    print(f"Grid search: {len(configs)} configuration(s) found.")

    # Step 1: Validate each configuration if not in fast mode, or deactivate the proxy and skip dry_run.
    if args.fast:
        print("Fast mode activated. Skipping dry_run.")
        for cfg in configs:
            # Deactivate the proxy to skip dry_run
            cfg.proxy.deactivate()
            approved_trials.append(cfg)
    else:
        print("Validating configurations with dry_run...")
        for cfg in configs:
            try:
                agent = Creator(cfg).agent()
                agent.dry_run()

                # If config is incomplete, mark it for template generation
                if cfg.has_missing():
                    has_missing_key = True
                else:
                    approved_trials.append(cfg)
            except Exception as e:
                print(f"[dry_run failed] {e}")
                return

    # Step 2: Create a config template if any configuration has missing keys
    if has_missing_key:
        import time
        cfg.dump_template(f"config/templates/generated_by_dry_run_{time.strftime('%Y%m%d_%H%M%S')}.yaml")
        print("Missing keys found. Template generated in config/templates/ directory.\nPlease Rerun after filling config.")
        return

    # Step 3: Run training if all configurations are valid
    print(f"Training {len(approved_trials)} configuration(s).")
    for cfg in approved_trials:
        create = Creator(cfg)
        agent = create.agent()
        agent.fit()


if __name__ == "__main__":
    main()