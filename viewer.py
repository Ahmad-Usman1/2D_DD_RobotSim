import os
import pygame
import yaml
import ast

# --------------------------------------
# CONFIG
# --------------------------------------
IN2D_ROOT = "in2d"  # root folder
SCALE = 50  # pixels per meter for map scaling
LINE_COLOR = (0, 200, 0)
POINT_COLOR = (255, 0, 0)
BG_COLOR = (30, 30, 30)


# --------------------------------------
# LOAD A SINGLE MAP FOLDER
# --------------------------------------
def load_in2d_folder(path):
    png = None
    yaml_file = None
    tasks_file = None
    episodes_file = None

    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if f.endswith(".png"):
            png = fp
        elif f.endswith(".yaml"):
            yaml_file = fp
        elif f.endswith("tasks.txt"):
            tasks_file = fp
        elif f.endswith("episodes.txt"):
            episodes_file = fp

    return png, yaml_file, tasks_file, episodes_file


# --------------------------------------
# PARSING HELPERS
# --------------------------------------
def parse_pairs_line(line):
    """Parses a line containing something like: [(x1,y1), (x2,y2), ...]"""
    try:
        return ast.literal_eval(line.strip())
    except:
        return []


def load_tasks(tasks_path):
    if tasks_path is None:
        return []
    with open(tasks_path, "r") as f:
        lines = f.readlines()
    if len(lines) < 2:
        return []

    starts = parse_pairs_line(lines[0])
    goals = parse_pairs_line(lines[1])
    return list(zip(starts, goals))


def load_episodes(ep_path):
    if ep_path is None:
        return []
    episodes = []
    with open(ep_path, "r") as f:
        for line in f:
            episodes.append(parse_pairs_line(line))
    return episodes


# --------------------------------------
# LOAD MAP IMAGE (convert to 24/32 bit)
# --------------------------------------
def load_and_scale_map(png_path, scale):
    img = pygame.display.set_mode((800, 600))
    img = pygame.image.load(png_path).convert_alpha()

    w, h = img.get_size()
    new_w = int(w * scale / 100)  # because IN2D maps use 1 cm per pixel → 100 px = 1 m
    new_h = int(h * scale / 100)

    img = pygame.transform.smoothscale(img, (new_w, new_h))
    return img, new_w, new_h


# --------------------------------------
# MAIN VIEWER
# --------------------------------------
def view_map(folder_index=1):
    pygame.init()

    # Find all subfolders in in2d root
    subfolders = sorted(
        [
            os.path.join(IN2D_ROOT, d)
            for d in os.listdir(IN2D_ROOT)
            if os.path.isdir(os.path.join(IN2D_ROOT, d))
        ]
    )

    if folder_index - 1 >= len(subfolders):
        print("Folder index out of range.")
        return

    folder = subfolders[folder_index - 1]
    print(f"Loading: {folder}")

    png, yaml_file, tasks_file, episodes_file = load_in2d_folder(folder)

    print("Map:", png)
    print("YAML:", yaml_file)
    print("Tasks:", tasks_file)
    print("Episodes:", episodes_file)

    # Load data
    tasks = load_tasks(tasks_file)
    episodes = load_episodes(episodes_file)

    # Print counts
    print(f"Tasks found: {len(tasks)}")
    print(f"Episodes found: {len(episodes)}")

    # Load & scale map
    map_img, w, h = load_and_scale_map(png, SCALE)

    # Create window exactly equal to map size
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("IN2D Viewer")

    running = True
    clock = pygame.time.Clock()

    # --------------------------------------
    # Draw loop
    # --------------------------------------
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.fill(BG_COLOR)
        window.blit(map_img, (0, 0))

        # Draw tasks (start–goal)
        for start, goal in tasks:
            sx = start[0] * SCALE
            sy = h - start[1] * SCALE
            gx = goal[0] * SCALE
            gy = h - goal[1] * SCALE

            pygame.draw.circle(window, POINT_COLOR, (int(sx), int(sy)), 5)
            pygame.draw.circle(window, (0, 255, 255), (int(gx), int(gy)), 5)
            pygame.draw.line(window, LINE_COLOR, (sx, sy), (gx, gy), 2)

        # Draw first episode (path)
        if len(episodes) > 0:
            ep = episodes[0]
            pts = [(p[0] * SCALE, h - p[1] * SCALE) for p in ep]
            for i in range(len(pts) - 1):
                pygame.draw.line(window, (255, 255, 0), pts[i], pts[i + 1], 2)
                pygame.draw.circle(
                    window, (0, 0, 255), (int(pts[i][0]), int(pts[i][1])), 4
                )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    view_map(folder_index=1)
