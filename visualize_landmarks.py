import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import trimesh


CLASS_COLORS = {
    "Mesial": "red",
    "Distal": "green",
    "InnerPoint": "gold",
    "OuterPoint": "cyan",
    "FacialPoint": "magenta",
    "Cusp": "blue",
}


def plot_mesh(ax, mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color="lightgray",
        alpha=0.35,
        linewidth=0.1,
        edgecolor="none",
    )


def plot_landmarks(ax, df):
    for cls, group in df.groupby("class"):
        color = CLASS_COLORS.get(cls, "black")
        ax.scatter(
            group["coord_x"],
            group["coord_y"],
            group["coord_z"],
            label=cls,
            s=40,
            c=color,
            depthshade=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize tooth landmarks on a mesh.")
    parser.add_argument("--mesh", required=True, help="Path to the input tooth mesh (.obj)")
    parser.add_argument("--csv", required=True, help="Path to the landmark CSV")
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title",
    )
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    csv_path = Path(args.csv)

    mesh = trimesh.load(mesh_path, force="mesh")
    df = pd.read_csv(csv_path)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    plot_mesh(ax, mesh)
    plot_landmarks(ax, df)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(args.title or mesh_path.stem)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
