import tkinter as tk
from tkinter import ttk

CELL_SIZE = 20
ROWS = 16
COLUMNS = 30

# Holds references to the cell widgets for interaction.
cells = []

def build_grid(container):
    """Creates a ROWS x COLUMNS grid using Frame widgets (no canvas)."""
    # Clear any existing cells
    for child in container.winfo_children():
        child.destroy()
    cells.clear()

    # Outer border on the container
    container.configure(highlightthickness=2, highlightbackground="#444")

    for r in range(ROWS):
        row_cells = []
        for c in range(COLUMNS):
            cell = tk.Frame(
                container,
                width=CELL_SIZE,
                height=CELL_SIZE,
                bg="#dcdcdc",
                highlightthickness=1,
                highlightbackground="#999",
            )
            cell.grid(row=r, column=c)
            cell.grid_propagate(False)  # keep the fixed pixel size
            cell.bind("<Button-1>", lambda _e, rr=r, cc=c: on_cell_click(rr, cc))
            row_cells.append(cell)
        cells.append(row_cells)


def on_cell_click(row, col):
    """Highlight the clicked cell to show interaction."""
    if 0 <= row < ROWS and 0 <= col < COLUMNS:
        cell = cells[row][col]
        cell.configure(bg="#a0d8ff")


def main():
    root = tk.Tk()
    root.title("Minesweeper GUI")

    # Left: grid built with Frames laid out by grid geometry manager
    grid_frame = ttk.Frame(root, padding=10)
    grid_frame.pack(side="left", fill="both", expand=True)

    grid_container = tk.Frame(grid_frame, bg="#f5f5f5")
    grid_container.pack()

    build_grid(grid_container)

    # Right: vertical buttons
    sidebar = ttk.Frame(root, padding=10)
    sidebar.pack(side="right", fill="y")

    ttk.Button(sidebar, text="Start").pack(fill="x", pady=2)
    ttk.Button(sidebar, text="Reset", command=lambda: build_grid(grid_container)).pack(fill="x", pady=2)
    ttk.Button(sidebar, text="Quit", command=root.destroy).pack(fill="x", pady=2)

    root.mainloop()


if __name__ == "__main__":
    main()
