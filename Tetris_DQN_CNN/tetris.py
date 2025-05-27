import random
import cv2
import numpy as np
from PIL import Image
from time import sleep

class Tetris:
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: {0: [(0,0),(1,0),(2,0),(3,0)], 90: [(1,0),(1,1),(1,2),(1,3)],
            180: [(3,0),(2,0),(1,0),(0,0)], 270: [(1,3),(1,2),(1,1),(1,0)]},
        1: {0: [(1,0),(0,1),(1,1),(2,1)], 90: [(0,1),(1,2),(1,1),(1,0)],
            180: [(1,2),(2,1),(1,1),(0,1)], 270: [(2,1),(1,0),(1,1),(1,2)]},
        2: {0: [(1,0),(1,1),(1,2),(2,2)], 90: [(0,1),(1,1),(2,1),(2,0)],
            180: [(1,2),(1,1),(1,0),(0,0)], 270: [(2,1),(1,1),(0,1),(0,2)]},
        3: {0: [(1,0),(1,1),(1,2),(0,2)], 90: [(0,1),(1,1),(2,1),(2,2)],
            180: [(1,2),(1,1),(1,0),(2,0)], 270: [(2,1),(1,1),(0,1),(0,0)]},
        4: {0: [(0,0),(1,0),(1,1),(2,1)], 90: [(0,2),(0,1),(1,1),(1,0)],
            180: [(2,1),(1,1),(1,0),(0,0)], 270: [(1,0),(1,1),(0,1),(0,2)]},
        5: {0: [(2,0),(1,0),(1,1),(0,1)], 90: [(0,0),(0,1),(1,1),(1,2)],
            180: [(0,1),(1,1),(1,0),(2,0)], 270: [(1,2),(1,1),(0,1),(0,0)]},
        6: {0: [(1,0),(2,0),(1,1),(2,1)], 90: [(1,0),(2,0),(1,1),(2,1)],
            180: [(1,0),(2,0),(1,1),(2,1)], 270: [(1,0),(2,0),(1,1),(2,1)]}
    }

    COLORS = {0: (255,255,255), 1: (247,64,99), 2: (0,167,247)}

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.score = 0
        self.lines_cleared_this_episode = 0
        self.reward_accumulated_this_episode = 0.0
        self.clears_1_this_episode = 0
        self.clears_2_this_episode = 0
        self.clears_3_this_episode = 0
        self.clears_4_this_episode = 0

        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        return self._get_board_props(self.board)

    def get_lines_cleared_this_episode(self): return self.lines_cleared_this_episode
    def get_reward_accumulated_this_episode(self): return self.reward_accumulated_this_episode
    def get_clears_1_this_episode(self): return self.clears_1_this_episode
    def get_clears_2_this_episode(self): return self.clears_2_this_episode
    def get_clears_3_this_episode(self): return self.clears_3_this_episode
    def get_clears_4_this_episode(self): return self.clears_4_this_episode

    def _new_round(self):
        if not self.bag:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)
        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0
        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True

    def _get_rotated_piece(self):
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]

    def _check_collision(self, piece, pos):
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH or y < 0 or y >= Tetris.BOARD_HEIGHT:
                return True
            if self.board[y][x] == Tetris.MAP_BLOCK:
                return True
        return False

    def _add_piece_to_board(self, piece, pos):
        board = [row[:] for row in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = Tetris.MAP_BLOCK
        return board

    def _clear_lines(self, board):
        lines = [i for i, row in enumerate(board) if all(cell == Tetris.MAP_BLOCK for cell in row)]
        if lines:
            for i in sorted(lines, reverse=True):
                board.pop(i)
            for _ in lines:
                board.insert(0, [0] * Tetris.BOARD_WIDTH)
        return len(lines), board

    def _number_of_holes(self, board):
        holes = 0
        for col in zip(*board):
            seen_block = False
            for cell in col:
                if cell == Tetris.MAP_BLOCK:
                    seen_block = True
                elif seen_block:
                    holes += 1
        return holes

    def _bumpiness(self, board):
        heights = []
        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == Tetris.MAP_EMPTY:
                i += 1
            heights.append(Tetris.BOARD_HEIGHT - i)
        return sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))

    def _height(self, board):
        return sum(Tetris.BOARD_HEIGHT - next((i for i, cell in enumerate(col) if cell != 0), Tetris.BOARD_HEIGHT)
                   for col in zip(*board))

    def _get_board_props(self, board):
        _, board = self._clear_lines([row[:] for row in board])
        holes = self._number_of_holes(board)
        bumpiness = self._bumpiness(board)
        height = self._height(board)
        return np.array([holes, bumpiness, height], dtype=np.float32)

    def get_state_size(self):
        return 3

    def get_next_states(self):
        states = {}
        rotations = [0] if self.current_piece == 6 else ([0, 90] if self.current_piece == 0 else [0, 90, 180, 270])
        for r in rotations:
            piece = Tetris.TETROMINOS[self.current_piece][r]
            min_x = min(p[0] for p in piece)
            max_x = max(p[0] for p in piece)
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1
                if pos[1] >= 0:
                    temp_board = self._add_piece_to_board(piece, pos)
                    states[(x, r)] = self._get_board_props(temp_board)
        return states

    def play(self, x, rotation, render=False, render_delay=None):
        self.current_pos = [x, 0]
        self.current_rotation = rotation
        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True
            return -20.0, True, 0, self._get_board_props(self.board)

        while not self._check_collision(self._get_rotated_piece(), [self.current_pos[0], self.current_pos[1] + 1]):
            if render:
                self.render()
                if render_delay: sleep(render_delay)
            self.current_pos[1] += 1

        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines, self.board = self._clear_lines(self.board)

        if lines == 1: self.clears_1_this_episode += 1
        elif lines == 2: self.clears_2_this_episode += 1
        elif lines == 3: self.clears_3_this_episode += 1
        elif lines >= 4: self.clears_4_this_episode += 1

        reward = 1.0 + [0, 50, 150, 400, 1000][min(lines, 4)]
        holes = self._number_of_holes(self.board)
        height = self._height(self.board)
        bumpiness = self._bumpiness(self.board)
        reward -= holes * 0.5
        reward -= max(0, height - 12) * 0.2
        reward -= bumpiness * 0.1

        self.score += 1 + (lines ** 2) * Tetris.BOARD_WIDTH
        self.lines_cleared_this_episode += lines
        self.reward_accumulated_this_episode += reward

        self._new_round()
        return reward, self.game_over, lines, self._get_board_props(self.board)

    def get_game_score(self):
        return self.score

    def render(self):
        img = [Tetris.COLORS[p] for row in self.board for p in row]
        img = np.array(img).reshape((Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3)).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, 'RGB')
        img = img.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25), Image.NEAREST)
        img = np.array(img)
        cv2.putText(img, f'Score: {self.score}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(img, f'Lines: {self.lines_cleared_this_episode}', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(img, f'1:{self.clears_1_this_episode} 2:{self.clears_2_this_episode} 3:{self.clears_3_this_episode} 4:{self.clears_4_this_episode}', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imshow('Tetris MLP', img)
        cv2.waitKey(1)
