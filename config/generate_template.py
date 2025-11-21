import os
import re
from pathlib import Path

# Determine project root and paths to the agent and component directories
base_dir = Path(__file__).resolve().parent.parent
agent_dir = base_dir / "cvlabkit" / "agent"
component_dir = base_dir / "cvlabkit" / "component"

# Prepare list of placeholders with comments and defaults
# Each tuple: (Comment text, key name, default value)
placeholders = []

# For agent: pick first .py file (excluding dunder) as default, if any
if agent_dir.exists():
    agent_files = sorted(
        [f.stem for f in agent_dir.glob("*.py") if not f.name.startswith("_")]
    )
    default_agent = agent_files[0] if agent_files else ""
    placeholders.append(("Agent (select implementation)", "agent", default_agent))

# For components: add a header and then each category with its default = first .py file in that category
if component_dir.exists():
    placeholders.append(("Components: (select implementation)", "", ""))
    for category in sorted(
        [p.name for p in component_dir.iterdir() if p.is_dir() and p.name != "base"]
    ):
        cat_dir = component_dir / category
        impl_files = sorted(
            [f.stem for f in cat_dir.glob("*.py") if not f.name.startswith("_")]
        )
        default_impl = impl_files[0] if impl_files else ""
        placeholders.append(("", category, default_impl))

# Compile regex patterns for cfg.attribute and cfg.get("attribute")
pattern_attr = re.compile(r"cfg\.([a-zA-Z_]\w*)")
pattern_get = re.compile(r'cfg\.get\(["\']([a-zA-Z_]\w*)["\']')

# Traverse agent and component directories
# Collect all flat cfg keys (depth=1) referenced in code files
cfg_keys = set()
for subdir in [agent_dir, component_dir]:
    if not subdir.exists():
        continue
    for root, _, files in os.walk(subdir):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                try:
                    content = file_path.read_text(encoding="utf-8")
                # Skip files that cannot be decoded
                except:
                    continue
                cfg_keys.update(pattern_attr.findall(content))
                cfg_keys.update(pattern_get.findall(content))

# Remove keys that correspond to placeholders
for _, key, _ in placeholders:
    cfg_keys.discard(key)

# Write the resulting template with comments to YAML
output_path = base_dir / "config" / "templates" / "generated_basic.yaml"
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    # Write placeholders section
    for comment, key, default in placeholders:
        if comment:
            f.write(f"# {comment}\n")
        if key:
            f.write(f"{key}: {default}\n")
    # Write other cfg keys
    f.write("# Other configuration keys (flat, depth=1)\n")
    for key in sorted(cfg_keys):
        f.write(f"{key}: \n")

print(f"[✓] Generated YAML template with comments at: {output_path.resolve()}")
