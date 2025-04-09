from copy import deepcopy


def load_scenario(index):
    board = [["." for _ in range(9)] for _ in range(10)]

    if index == 1:
        board[0][3] = "BA"
        board[1][4] = "BA"
        board[1][5] = "BK"

        board[2][2] = "RS"
        board[3][3] = "RS"
        board[9][4] = "RK"

        # board[9][3] = "RA"  # Advisor
        # board[8][4] = "RA"  # Advisor
        # board[8][5] = "RK"  # King

        # board[7][2] = "BS"  # Soldier
        # board[6][3] = "BS"  # Soldier
        # board[0][4] = "BK"  # King

    elif index == 2:
        board[0][4] = "BK"
        board[1][4] = "BA"
        board[5][4] = "BS"

        board[9][4] = "RK"
        board[9][2] = "RE"
        board[3][3] = "RS"
        board[3][5] = "RS"

    elif index == 3:
        board[0][4] = "BK"
        board[4][6] = "BH"
        board[6][4] = "BS"
        board[6][5] = "BS"

        board[9][4] = "RK"
        board[8][0] = "RC"
        board[5][2] = "RE"
        board[5][6] = "RH"

    return board


class CurrentBoard:
    def __init__(self, board_state=None, skip_state_check=False):
        if board_state:
            self.board = [row[:] for row in board_state]
        else:
            self.board = load_scenario(1)

        self.state = "U"
        if not skip_state_check:
            self.state = self.state_of_board()

    def display(self):
        for r, row in enumerate(self.board):
            print(f"{r:2}: " + " ".join(row))
        print("     " + " ".join(str(i) for i in range(9)))
        print("\n")

    def other(self, piece):
        return "B" if piece == "R" else "R"

    def get_piece_owner(self, piece):
        return piece[0] if piece != "." else None

    def in_bounds(self, r, c):
        return 0 <= r < 10 and 0 <= c < 9

    def in_palace(self, r, c, owner):
        if owner == "B":
            return 0 <= r <= 2 and 3 <= c <= 5
        elif owner == "R":
            return 7 <= r <= 9 and 3 <= c <= 5
        return False

    def get_king_position(self, owner):
        king_symbol = f"{owner}K"
        for r in range(10):
            for c in range(9):
                if self.board[r][c] == king_symbol:
                    return r, c
        return -1, -1

    def is_valid_move(self, from_r, from_c, to_r, to_c):
        piece = self.board[from_r][from_c]
        if piece == ".":
            return False

        owner = self.get_piece_owner(piece)
        # Check if destination has own piece
        dest = self.board[to_r][to_c]
        if dest != "." and self.get_piece_owner(dest) == owner:
            return False

        # Basic move validity is checked in get_moves_for_piece
        for move in self.get_moves_for_piece_simple(piece, from_r, from_c):
            new_r, new_c = move
            if new_r == to_r and new_c == to_c:
                # Check if move puts own king in check
                new_board = [row[:] for row in self.board]
                new_board[to_r][to_c] = piece
                new_board[from_r][from_c] = "."
                temp_board = CurrentBoard(new_board, skip_state_check=True)
                if not temp_board.is_king_in_check(owner):
                    return True
        return False

    def kings_face_each_other(self):
        bk_row, bk_col = self.get_king_position("B")
        rk_row, rk_col = self.get_king_position("R")

        # Check if kings are in the same column
        if bk_col == rk_col:
            # Check if there are pieces between them
            min_row = min(bk_row, rk_row)
            max_row = max(bk_row, rk_row)
            for r in range(min_row + 1, max_row):
                if self.board[r][bk_col] != ".":
                    return False  # There's a piece between kings
            return True  # Kings face each other with nothing in between
        return False  # Kings are not in the same column

    def is_king_in_check(self, owner):
        kr, kc = self.get_king_position(owner)
        if kr == -1:
            return True  # King not found, considered in check

        # Check if kings are facing each other (illegal position)
        if self.kings_face_each_other():
            return True

        opponent = self.other(owner)
        for r in range(10):
            for c in range(9):
                piece = self.board[r][c]
                if piece != "." and self.get_piece_owner(piece) == opponent:
                    # Get simple moves (no recursion check)
                    for nr, nc in self.get_moves_for_piece_simple(piece, r, c):
                        if nr == kr and nc == kc:
                            return True
        return False

    def state_of_board(self):
        red_king = black_king = False
        for row in self.board:
            for cell in row:
                if cell == "RK":
                    red_king = True
                elif cell == "BK":
                    black_king = True

        red_moves = self.all_possible_moves("R")
        black_moves = self.all_possible_moves("B")

        if not red_king:
            return "B_WIN"
        if not black_king:
            return "R_WIN"
        if not red_moves and not black_moves:
            return "D"
        if not red_moves:
            return "B_WIN"
        if not black_moves:
            return "R_WIN"

        return "U"

    def all_possible_moves(self, player):
        moves = []
        for r in range(10):
            for c in range(9):
                piece = self.board[r][c]
                if piece != "." and self.get_piece_owner(piece) == player:
                    moves.extend(self.get_moves_for_piece(piece, r, c))
        return moves

    def get_moves_for_piece_simple(self, piece, r, c):
        """Simplified move generation without recursion for check detection"""
        moves = []
        owner = self.get_piece_owner(piece)

        def add_move(nr, nc):
            if self.in_bounds(nr, nc):
                target = self.board[nr][nc]
                if target == "." or self.get_piece_owner(target) != owner:
                    if piece[1] == "K" and not self.in_palace(nr, nc, owner):
                        return
                    moves.append((nr, nc))

        if piece[1] == "K":
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if self.in_palace(nr, nc, owner):
                    add_move(nr, nc)

        elif piece[1] == "A":
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if self.in_palace(nr, nc, owner):
                    add_move(nr, nc)

        elif piece[1] == "E":
            for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                nr, nc = r + dr, c + dc
                eye_r, eye_c = r + dr // 2, c + dc // 2
                if (
                    self.in_bounds(nr, nc)
                    and self.board[eye_r][eye_c] == "."
                    and ((owner == "R" and nr >= 5) or (owner == "B" and nr <= 4))
                ):
                    add_move(nr, nc)

        elif piece[1] == "H":
            horse_moves = [
                (-2, -1),
                (-2, 1),
                (2, -1),
                (2, 1),
                (-1, -2),
                (-1, 2),
                (1, -2),
                (1, 2),
            ]
            legs = [(-1, 0), (-1, 0), (1, 0), (1, 0), (0, -1), (0, -1), (0, 1), (0, 1)]
            for i, (dr, dc) in enumerate(horse_moves):
                leg_r, leg_c = r + legs[i][0], c + legs[i][1]
                if self.in_bounds(leg_r, leg_c) and self.board[leg_r][leg_c] == ".":
                    add_move(r + dr, c + dc)

        elif piece[1] == "R":
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                while self.in_bounds(nr, nc):
                    if self.board[nr][nc] == ".":
                        add_move(nr, nc)
                    else:
                        if self.get_piece_owner(self.board[nr][nc]) != owner:
                            add_move(nr, nc)
                        break
                    nr += dr
                    nc += dc

        elif piece[1] == "C":
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                jumped = False
                while self.in_bounds(nr, nc):
                    if not jumped:
                        if self.board[nr][nc] == ".":
                            add_move(nr, nc)
                        else:
                            jumped = True
                    else:
                        if (
                            self.board[nr][nc] != "."
                            and self.get_piece_owner(self.board[nr][nc]) != owner
                        ):
                            add_move(nr, nc)
                            break
                        elif self.board[nr][nc] != ".":
                            break
                    nr += dr
                    nc += dc

        elif piece[1] == "S":
            dr = 1 if owner == "R" else -1
            add_move(r - dr, c)
            if (owner == "R" and r <= 4) or (owner == "B" and r >= 5):
                add_move(r, c - 1)
                add_move(r, c + 1)

        return moves

    def get_moves_for_piece(self, piece, r, c):
        moves = []
        owner = self.get_piece_owner(piece)

        # Get simple moves
        for nr, nc in self.get_moves_for_piece_simple(piece, r, c):
            # Check if move puts king in check
            new_board = [row[:] for row in self.board]
            new_board[nr][nc] = piece
            new_board[r][c] = "."
            temp_cb = CurrentBoard(new_board, skip_state_check=True)

            # Only add move if it doesn't put own king in check
            if not temp_cb.is_king_in_check(owner):
                moves.append(temp_cb)

        return moves


class SearchTreeNode:
    def __init__(self, board_instance, playing_as, ply=0):
        # Store the node properties
        self.children = []
        self.value_is_assigned = False
        self.ply_depth = ply
        self.current_board = board_instance
        self.move_for = playing_as

        # Define how deep we want to search
        # Increased depth helps with endgames
        MAX_PLY_DEPTH = 5
        board_state = board_instance.state_of_board()

        # Only expand the node if it's not a terminal state and we haven't reached max depth
        if board_state == "U" and ply < MAX_PLY_DEPTH:
            self.generate_children()
        else:
            # For terminal nodes or max depth, evaluate the position
            self.value = self.evaluate_terminal_state(board_state)
            self.value_is_assigned = True

    def evaluate_terminal_state(self, board_state):
        # Score terminal positions
        # Draw = 0, win = 1, loss = -1
        if board_state == "D":
            return 0
        elif board_state == f"{self.move_for}_WIN":
            return 1
        else:
            return -1

    def min_max_value(self, alpha=-float("inf"), beta=float("inf")):
        """
        Alpha-beta pruning version of minimax algorithm
        Alpha is the best value the maximizing player has found
        Beta is the best value the minimizing player has found
        """
        # Return cached value if already calculated
        if self.value_is_assigned:
            return self.value

        # Check if maximizing or minimizing player's turn
        if (self.ply_depth % 2) == 0:
            self.value = -float("inf")

            for child in self.children:
                self.value = max(self.value, child.min_max_value(alpha, beta))

                alpha = max(alpha, self.value)

                if alpha >= beta:
                    break
        else:
            self.value = float("inf")

            for child in self.children:
                self.value = min(self.value, child.min_max_value(alpha, beta))

                beta = min(beta, self.value)

                if beta <= alpha:
                    break

        self.value_is_assigned = True
        return self.value

    def generate_children(self):
        """
        Create all possible move nodes from this position
        Sort them to improve alpha-beta pruning efficiency
        """
        # Create all possible next positions
        possible_moves = self.current_board.all_possible_moves(self.move_for)

        # Create child nodes for each move
        for next_board in possible_moves:
            self.children.append(
                SearchTreeNode(
                    next_board,
                    self.current_board.other(self.move_for),
                    self.ply_depth + 1,
                )
            )

        # Sort moves to improve pruning efficiency
        # For maximizer, try highest value moves first
        # For minimizer, try lowest value moves first
        is_maximizing = self.ply_depth % 2 == 0

        # Pre-evaluate each child to help with sorting
        for child in self.children:
            if not child.value_is_assigned:
                # Score based on improved evaluation function
                child.value = self.evaluate_position(child.current_board)
                child.value_is_assigned = True

        # Sort children for better pruning efficiency
        self.children.sort(key=lambda x: x.value, reverse=is_maximizing)

    def evaluate_position(self, board):
        """
        Improved evaluation function with endgame knowledge
        """
        # Material evaluation
        red_material = black_material = 0
        piece_values = {
            "K": 1000,
            "R": 9,
            "H": 4,
            "C": 4.5,
            "A": 2,
            "E": 2,
            "S": 1,
        }

        # Position evaluation
        red_position = black_position = 0

        # Get kings' positions
        red_king_r, red_king_c = board.get_king_position("R")
        black_king_r, black_king_c = board.get_king_position("B")

        # Count material and evaluate positions
        for r in range(10):
            for c in range(9):
                piece = board.board[r][c]
                if piece != ".":
                    piece_type = piece[1]
                    piece_worth = piece_values.get(piece_type, 0)
                    if piece[0] == "R":
                        red_material += piece_worth

                        # Endgame specific: Reward positions closer to enemy king
                        if piece_type in ["A", "S"]:
                            distance_to_enemy = abs(r - black_king_r) + abs(
                                c - black_king_c
                            )
                            red_position += (10 - distance_to_enemy) * 0.1

                    else:  # Black pieces
                        black_material += piece_worth

                        # Endgame specific: Reward positions closer to enemy king
                        if piece_type in ["A", "S"]:
                            distance_to_enemy = abs(r - red_king_r) + abs(
                                c - red_king_c
                            )
                            black_position += (10 - distance_to_enemy) * 0.1

        # King safety evaluation
        red_king_safety = black_king_safety = 0

        # Count defenders around king
        red_defenders = self.count_defenders(board, "R", red_king_r, red_king_c)
        black_defenders = self.count_defenders(board, "B", black_king_r, black_king_c)

        red_king_safety = red_defenders * 0.2
        black_king_safety = black_defenders * 0.2

        # Reward kings in the center of their palaces in endgames
        if red_king_c == 4:
            red_king_safety += 0.1

        if black_king_c == 4:
            black_king_safety += 0.1

        # Endgame specific: King opposition
        king_file_distance = abs(red_king_c - black_king_c)

        if self.is_endgame(board):
            # In king vs king endgames, reward opposition
            if king_file_distance == 0:
                if self.move_for == "R":
                    # Check if it's an odd number of rows between kings (good opposition)
                    if (abs(red_king_r - black_king_r) % 2) == 1:
                        red_position += 0.3
                else:
                    if (abs(red_king_r - black_king_r) % 2) == 1:
                        black_position += 0.3

            # Reward central king position in pure king endgames
            if self.is_king_only_endgame(board):
                if self.move_for == "R":
                    # Kings closer to center are better
                    center_dist_red = abs(red_king_c - 4)
                    red_position += (4 - center_dist_red) * 0.05
                else:
                    center_dist_black = abs(black_king_c - 4)
                    black_position += (4 - center_dist_black) * 0.05

        # Check status
        if board.is_king_in_check("R"):
            red_king_safety -= 0.5

        if board.is_king_in_check("B"):
            black_king_safety -= 0.5

        # Calculate final score
        red_score = red_material + red_position + red_king_safety
        black_score = black_material + black_position + black_king_safety

        # Return normalized score from perspective of moving player
        if self.move_for == "R":
            return (red_score - black_score) / 100.0
        else:
            return (black_score - red_score) / 100.0

    def count_defenders(self, board, player, king_r, king_c):
        """Count pieces defending the king"""
        count = 0
        for r in range(max(0, king_r - 1), min(10, king_r + 2)):
            for c in range(max(0, king_c - 1), min(9, king_c + 2)):
                if (
                    board.board[r][c] != "."
                    and board.board[r][c][0] == player
                    and board.board[r][c][1] != "K"
                ):
                    count += 1
        return count

    def is_endgame(self, board):
        """Check if the position is an endgame"""
        piece_count = 0
        for r in range(10):
            for c in range(9):
                if board.board[r][c] != ".":
                    piece_count += 1
        return piece_count <= 6

    def is_king_only_endgame(self, board):
        """Check if it's a king vs king (maybe with advisors) endgame"""
        has_non_royal = False
        for r in range(10):
            for c in range(9):
                piece = board.board[r][c]
                if piece != "." and piece[1] not in ["K", "A"]:
                    has_non_royal = True
                    break
        return not has_non_royal
