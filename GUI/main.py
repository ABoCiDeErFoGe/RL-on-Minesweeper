import tkinter as tk
from tkinter import ttk

CELL_SIZE = 20
ROWS = 16
COLUMNS = 30

def build_grid(canvas):
    """Draws a ROWS x COLUMNS grid of square cells on the canvas."""
    canvas.delete("all")
    for r in range(ROWS):
        for c in range(COLUMNS):
            x0 = c * CELL_SIZE
            y0 = r * CELL_SIZE
            x1 = x0 + CELL_SIZE
            y1 = y0 + CELL_SIZE
            canvas.create_rectangle(x0, y0, x1, y1, outline="#999", fill="#dcdcdc")


def on_cell_click(event, canvas):
    """Highlight the clicked cell to show interaction."""
    col = event.x // CELL_SIZE
    row = event.y // CELL_SIZE
    if 0 <= col < COLUMNS and 0 <= row < ROWS:
        x0 = col * CELL_SIZE
        y0 = row * CELL_SIZE
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE
        canvas.create_rectangle(x0, y0, x1, y1, outline="#555", fill="#a0d8ff")


def main():
    root = tk.Tk()
    root.title("Minesweeper GUI")

    # Left: grid canvas
    grid_frame = ttk.Frame(root, padding=10)
    grid_frame.pack(side="left", fill="both", expand=True)

    canvas_width = COLUMNS * CELL_SIZE
    canvas_height = ROWS * CELL_SIZE
    grid_canvas = tk.Canvas(grid_frame, width=canvas_width, height=canvas_height, bg="#f5f5f5", highlightthickness=0)
    grid_canvas.pack()

    build_grid(grid_canvas)
    grid_canvas.bind("<Button-1>", lambda e: on_cell_click(e, grid_canvas))

    # Right: vertical buttons
    sidebar = ttk.Frame(root, padding=10)
    sidebar.pack(side="right", fill="y")

    ttk.Button(sidebar, text="Start").pack(fill="x", pady=2)
    ttk.Button(sidebar, text="Reset", command=lambda: build_grid(grid_canvas)).pack(fill="x", pady=2)
    ttk.Button(sidebar, text="Quit", command=root.destroy).pack(fill="x", pady=2)

    root.mainloop()


if __name__ == "__main__":
    main()
