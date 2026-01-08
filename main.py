import threading
import tkinter as tk
from playwright.sync_api import sync_playwright

ROWS = 16
COLUMNS = 30
CELL_SIZE = 20

class App:
    def __init__(self):
        self.cells = []
        self.root = tk.Tk()
        self.root.title("Controller")
        self.root.attributes("-topmost", True)

        self.initialize_interface()

        # Event used to signal the worker thread to stop Playwright
        self._stop_playwright = threading.Event()

        # Start the worker thread for Playwright
        self.worker_thread = threading.Thread(target=self.initialize_playwright, daemon=True)
        self.worker_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def initialize_playwright(self):
        """Initialize Playwright in a worker thread."""
        try:
            self.update_status("Status: Initializing Playwright...")
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page()
            self.page.goto("https://minesweeperonline.com")
            self.update_status("Status: Ready")
        except Exception:
            # If initialization fails, ensure status reflects it and exit thread
            self.update_status("Status: Playwright failed to start")
            return

        # Wait until main thread signals shutdown; cleanup must happen in this worker thread
        self._stop_playwright.wait()
        try:
            self.update_status("Status: Shutting down Playwright...")
            if hasattr(self, 'browser'):
                self.browser.close()
            if hasattr(self, 'playwright'):
                self.playwright.stop()
            self.update_status("Status: Playwright stopped")
        except Exception:
            # ignore shutdown errors; thread will exit
            pass

    def update_status(self, message):
        """Update the status label in a thread-safe manner."""
        self.root.after(0, lambda: self.status_label.config(text=message))

    def on_click(self):
        self.page.locator("text=More information").click()

    def on_close(self):
        # Signal the worker thread to stop Playwright and wait for it to exit
        self.update_status("Status: Exiting...")
        if hasattr(self, '_stop_playwright'):
            self._stop_playwright.set()

        # Poll until the worker thread has finished cleanup, then destroy the root
        self.root.after(100, self._check_worker_and_destroy)

    def _check_worker_and_destroy(self):
        if self.worker_thread.is_alive():
            self.root.after(100, self._check_worker_and_destroy)
        else:
            self.root.destroy()

    def run(self):
        self.root.mainloop()
        
    def build_grid(self, container):
        """Creates a ROWS x COLUMNS grid using Frame widgets (no canvas)."""
        # Clear any existing cells
        for child in container.winfo_children():
            child.destroy()
        self.cells.clear()

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
                cell.bind("<Button-1>", lambda _e, rr=r, cc=c: self.on_cell_click(rr, cc))
                row_cells.append(cell)
            self.cells.append(row_cells)


    def on_cell_click(self, row, col):
        """Highlight the clicked cell to show interaction."""
        if 0 <= row < ROWS and 0 <= col < COLUMNS:
            cell = self.cells[row][col]
            cell.configure(bg="#a0d8ff")

    def initialize_interface(self):
        # Left column: contains grid on top and status label below
        self.left_column = tk.Frame(self.root)
        # add outer padding around the left column (around the grid)
        self.left_column.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # grid frame sits inside the left column
        self.grid_frame = tk.Frame(self.left_column)
        self.grid_frame.pack(side="top", fill="both", expand=True)

        self.grid_container = tk.Frame(self.grid_frame, bg="#f5f5f5")
        self.grid_container.pack()
        self.build_grid(self.grid_container)

        # Right: vertical buttons
        # Right: vertical buttons
        self.sidebar = tk.Frame(self.root)
        self.sidebar.pack(side="right", fill="y", padx=8, pady=8)

        tk.Button(self.sidebar, text="Start").pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Reset", command=lambda: self.build_grid(self.grid_container)).pack(fill="x", pady=2)
        tk.Button(self.sidebar, text="Quit", command=self.root.destroy).pack(fill="x", pady=2)
        
        # Label indicating the progress, placed below the grid in the left column
        self.status_label = tk.Label(self.left_column, text="Status: Initializing...", anchor="w")
        self.status_label.pack(side="bottom", fill="x", padx=4, pady=(6,0))
        

App().run()