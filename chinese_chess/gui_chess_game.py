import tkinter as tk
from tkinter import messagebox
from chinese_chess_search_tree import load_scenario, CurrentBoard, SearchTreeNode

TILE_SIZE = 60
ROWS, COLS = 10, 9
PIECE_COLORS = {"R": "red", "B": "black"}
TEXT_COLORS = {"R": "white", "B": "white"}
MARGIN = 30


class XiangqiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chinese Chess Endgame GUI")
        self.canvas = tk.Canvas(
            root,
            width=MARGIN * 2 + (COLS - 1) * TILE_SIZE,
            height=MARGIN * 2 + (ROWS - 1) * TILE_SIZE,
            bg="burlywood",
        )

        self.canvas.pack()

        self.status_label = tk.Label(
            root,
            text="Chinese Chess Endgame",
        )
        self.status_label.pack()

        self.canvas.bind("<Button-1>", self.on_click)

        self.selected = None
        self.player = "R"
        self.human_side = "R"
        self.cb = None
        self.valid_moves = []

        self.ask_scenario()

    def ask_scenario(self):
        def start():
            idx = int(entry.get())
            self.cb = CurrentBoard(load_scenario(idx))
            top.destroy()
            self.draw_board()

        top = tk.Toplevel(self.root)
        tk.Label(top, text="Enter Scenario (1-3):").pack()
        entry = tk.Entry(top)
        entry.pack()
        tk.Button(top, text="Start", command=start).pack()

    def draw_board(self):
        self.canvas.delete("all")

        # Draw horizontal lines
        # Horizontal lines
        for r in range(ROWS):
            y = MARGIN + r * TILE_SIZE
            self.canvas.create_line(MARGIN, y, MARGIN + (COLS - 1) * TILE_SIZE, y)

        # Vertical lines
        for c in range(COLS):
            x = MARGIN + c * TILE_SIZE
            if c == 0 or c == COLS - 1:
                self.canvas.create_line(x, MARGIN, x, MARGIN + (ROWS - 1) * TILE_SIZE)
            else:
                self.canvas.create_line(x, MARGIN, x, MARGIN + 4 * TILE_SIZE)
                self.canvas.create_line(
                    x, MARGIN + 5 * TILE_SIZE, x, MARGIN + (ROWS - 1) * TILE_SIZE
                )

        # Draw palace diagonals (optional)
        for palace_c in [3, 5]:
            # Draw palace diagonals (top and bottom), adjusted for MARGIN
            self.canvas.create_line(
                MARGIN + 3 * TILE_SIZE,
                MARGIN + 0 * TILE_SIZE,
                MARGIN + 5 * TILE_SIZE,
                MARGIN + 2 * TILE_SIZE,
            )
            self.canvas.create_line(
                MARGIN + 5 * TILE_SIZE,
                MARGIN + 0 * TILE_SIZE,
                MARGIN + 3 * TILE_SIZE,
                MARGIN + 2 * TILE_SIZE,
            )
            self.canvas.create_line(
                MARGIN + 3 * TILE_SIZE,
                MARGIN + 7 * TILE_SIZE,
                MARGIN + 5 * TILE_SIZE,
                MARGIN + 9 * TILE_SIZE,
            )
            self.canvas.create_line(
                MARGIN + 5 * TILE_SIZE,
                MARGIN + 7 * TILE_SIZE,
                MARGIN + 3 * TILE_SIZE,
                MARGIN + 9 * TILE_SIZE,
            )

        # Draw pieces on intersections
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.cb.board[r][c]
                if piece != ".":
                    x = MARGIN + c * TILE_SIZE
                    y = MARGIN + r * TILE_SIZE
                    color = PIECE_COLORS.get(piece[0], "blue")
                    text_color = TEXT_COLORS.get(piece[0], "white")
                    self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill=color)
                    self.canvas.create_text(
                        x,
                        y,
                        text=piece[1],
                        font=("Arial", 14, "bold"),
                        fill=text_color,
                    )

        # Check if the game is over after drawing the board
        game_state = self.cb.state_of_board()
        if game_state != "U":
            self.show_game_result(game_state)

    def show_game_result(self, result):
        result_text = ""
        if result == "R_WIN":
            result_text = "Red (You) wins!"
        elif result == "B_WIN":
            result_text = "Black (AI) wins!"
        elif result == "D":
            result_text = "Game is a draw!"
        else:
            result_text = f"Game over: {result}"

        self.status_label.config(text=result_text)
        messagebox.showinfo("Game Over", result_text)

    def on_click(self, event):
        if self.cb.state_of_board() != "U":
            self.show_game_result(self.cb.state_of_board())
            return

        # Adjust for board margin
        x = event.x - MARGIN
        y = event.y - MARGIN

        if x < 0 or y < 0:
            return

        r = round(y / TILE_SIZE)
        c = round(x / TILE_SIZE)

        if not self.cb.in_bounds(r, c):
            return

        piece = self.cb.board[r][c]

        print(f"Clicked: ({r}, {c}) - Piece: {piece}")  # Debug line

        if self.selected:
            # Try to move
            move_made = False
            for new_board in self.valid_moves:
                from_r, from_c = self.selected
                moved_piece = self.cb.board[from_r][from_c]

                if new_board.board[r][c] == moved_piece:
                    self.cb = new_board
                    self.draw_board()
                    self.selected = None
                    self.valid_moves = []

                    # Check if game is over after player's move
                    if self.cb.state_of_board() == "U":
                        self.root.after(500, self.ai_move)
                    move_made = True
                    return

            # Invalid move
            if not move_made:
                self.selected = None
                self.valid_moves = []
                messagebox.showwarning("Invalid Move", "That move is not allowed.")
                self.status_label.config(text="Invalid move. Select another piece.")
                self.draw_board()
        else:
            if piece != "." and self.cb.get_piece_owner(piece) == self.human_side:
                self.selected = (r, c)
                self.valid_moves = []

                for new_board in self.cb.get_moves_for_piece(piece, r, c):
                    self.valid_moves.append(new_board)

                if self.valid_moves:
                    self.status_label.config(
                        text=f"Selected {piece} at ({r},{c}). Click destination."
                    )
                else:
                    self.selected = None
                    messagebox.showinfo("No Moves", "This piece has no valid moves.")
                    self.status_label.config(
                        text="This piece has no valid moves. Select another piece."
                    )

    def ai_move(self):
        ai_side = self.cb.other(self.human_side)
        tree = SearchTreeNode(self.cb, ai_side)
        tree.min_max_value()

        # Check if AI has any valid moves
        if not tree.children:
            # AI has no valid moves, might be checkmate or stalemate
            game_state = self.cb.state_of_board()
            self.show_game_result(game_state)
            return

        # Find best AI move
        best_move = max(tree.children, key=lambda x: x.value)
        self.cb = best_move.current_board
        self.draw_board()

        # Check if game is over after AI's move
        game_state = self.cb.state_of_board()
        if game_state != "U":
            self.show_game_result(game_state)


if __name__ == "__main__":
    root = tk.Tk()
    app = XiangqiGUI(root)
    root.mainloop()
