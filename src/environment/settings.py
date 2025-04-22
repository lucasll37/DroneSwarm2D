# type: ignore
import os
import glob
import json
import pygame

# — Initialize pygame to get display info —
if not pygame.get_init():
    pygame.init()
    
# Encontre o primeiro .json em ./config e extraia seu nome (sem extensão)
config_dir   = os.path.join(os.path.dirname(__file__), "../../config")
json_paths   = sorted(glob.glob(os.path.join(config_dir, "*.json")))

if not json_paths:
    raise FileNotFoundError(f"Nenhum arquivo .json encontrado em {config_dir!r}")

first_path   = json_paths[0]
config_name  = os.path.splitext(os.path.basename(first_path))[0]
print(f"Carregando config '{config_name}' de: {first_path}")

# — 2. Leia o JSON desse arquivo —
with open(first_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

# — Override screen size with actual display info —
display_info            = pygame.display.Info()
raw["FULL_WIDTH"]       = display_info.current_w
raw["FULL_HEIGHT"]      = display_info.current_h

# — Evaluate any string expressions or nested lists —
cfg = {}
for key, val in raw.items():
    # case: simple expression in a string
    if isinstance(val, str):
        try:
            cfg[key] = eval(val, {}, cfg)
        except Exception:
            cfg[key] = val
    # case: list of values or expressions
    elif isinstance(val, list):
        lst = []
        for item in val:
            if isinstance(item, str):
                try:
                    lst.append(eval(item, {}, cfg))
                except Exception:
                    lst.append(item)
            else:
                lst.append(item)
        cfg[key] = lst
    # case: literal number, boolean, etc.
    else:
        cfg[key] = val

# — Create top-level variables exactly as in your original .py —
globals().update(cfg)

# — Compute the derived constants exactly as in your original script —
INTEREST_POINT_CENTER          = pygame.math.Vector2(SIM_WIDTH/2, SIM_HEIGHT/2)
CENTER                          = INTEREST_POINT_CENTER.copy()
TYPE_OF_SCENARIO               = config_name
DMZ = [
    (
        float(eval(expr_x, globals())),    # avalia "SIM_WIDTH * 0.35"
        float(eval(expr_y, globals())),    # avalia "SIM_HEIGHT * 0.30"
        int(radius)                        # já é literal
    )
    for expr_x, expr_y, radius in DMZ
]

# — Finally, set up your display and clock —
screen = pygame.display.set_mode(
    (FULL_WIDTH, FULL_HEIGHT),
    pygame.FULLSCREEN
)
clock = pygame.time.Clock()