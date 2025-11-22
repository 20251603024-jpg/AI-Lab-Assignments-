import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class MENACE:
    def __init__(self):
        # Matchboxes: dictionary of board states -> bead counts for each move
        self.matchboxes = defaultdict(lambda: [10] * 9)  # Start with 10 beads for each position
        self.history = []  # Store (state, move) pairs for current game
        self.symbols = {0: ' ', 1: 'X', -1: 'O'}
        
    def get_state_key(self, board):
        """Convert board to string key for matchbox lookup"""
        return ''.join(str(int(x)) for x in board.flatten())
    
    def get_available_moves(self, board):
        """Get list of available moves (empty positions)"""
        return [i for i in range(9) if board[i // 3, i % 3] == 0]
    
    def choose_move(self, board):
        """Choose a move based on current board state"""
        state_key = self.get_state_key(board)
        available_moves = self.get_available_moves(board)
        
        if not available_moves:
            return None
            
        # Get bead counts for available moves
        bead_counts = []
        for move in available_moves:
            bead_counts.append(self.matchboxes[state_key][move])
        
        # Choose move weighted by bead counts
        try:
            chosen_move = random.choices(available_moves, weights=bead_counts)[0]
            self.history.append((state_key, chosen_move))
            return chosen_move
        except:
            # Fallback: choose randomly if weights fail
            chosen_move = random.choice(available_moves)
            self.history.append((state_key, chosen_move))
            return chosen_move
    
    def update_matchboxes(self, result):
        """Update bead counts based on game result"""
        # result: 1 for win, 0 for draw, -1 for loss
        
        for state_key, move in self.history:
            if result == 1:  # Win - add beads
                self.matchboxes[state_key][move] += 3
            elif result == 0:  # Draw - add one bead
                self.matchboxes[state_key][move] += 1
            else:  # Loss - remove bead (but not below 1)
                self.matchboxes[state_key][move] = max(1, self.matchboxes[state_key][move] - 1)
        
        self.history = []  # Clear history for next game
    
    def check_winner(self, board):
        """Check if there's a winner"""
        # Check rows
        for i in range(3):
            if abs(board[i].sum()) == 3:
                return board[i, 0]
        
        # Check columns
        for i in range(3):
            if abs(board[:, i].sum()) == 3:
                return board[0, i]
        
        # Check diagonals
        if abs(board[0, 0] + board[1, 1] + board[2, 2]) == 3:
            return board[0, 0]
        if abs(board[0, 2] + board[1, 1] + board[2, 0]) == 3:
            return board[0, 2]
        
        # Check for draw
        if 0 not in board:
            return 0  # Draw
        
        return None  # Game continues
    
    def play_game(self, first_player='menace'):
        """Play a single game against a random opponent"""
        board = np.zeros((3, 3), dtype=int)
        current_player = 1 if first_player == 'menace' else -1
        
        while True:
            # Check for game end
            winner = self.check_winner(board)
            if winner is not None:
                return winner
            
            if current_player == 1:  # MENACE's turn
                move = self.choose_move(board)
                if move is None:
                    return 0  # Draw if no moves available
                board[move // 3, move % 3] = 1
            else:  # Random opponent's turn
                available_moves = self.get_available_moves(board)
                if not available_moves:
                    return 0  # Draw
                move = random.choice(available_moves)
                board[move // 3, move % 3] = -1
            
            current_player *= -1  # Switch player
    
    def train(self, num_games):
        """Train MENACE by playing multiple games"""
        results = []
        
        for game in range(num_games):
            # Alternate who starts first
            first_player = 'menace' if game % 2 == 0 else 'opponent'
            result = self.play_game(first_player)
            
            # Update matchboxes based on result
            if result == 1:  # MENACE won
                self.update_matchboxes(1)
                results.append('win')
            elif result == -1:  # MENACE lost
                self.update_matchboxes(-1)
                results.append('loss')
            else:  # Draw
                self.update_matchboxes(0)
                results.append('draw')
            
            # Print progress
            if (game + 1) % 100 == 0:
                wins = results.count('win')
                losses = results.count('loss')
                draws = results.count('draw')
                print(f"Game {game + 1}: Wins: {wins}, Losses: {losses}, Draws: {draws}")
        
        return results

    def print_board(self, board):
        """Print the current board state"""
        print("Current board:")
        for i in range(3):
            row = [self.symbols[board[i, j]] for j in range(3)]
            print(" " + " | ".join(row))
            if i < 2:
                print("-----------")
        print()

    def demonstrate_learning(self, num_demos=3):
        """Demonstrate MENACE's learned behavior"""
        print("\n=== DEMONSTRATING LEARNED BEHAVIOR ===")
        
        for demo in range(num_demos):
            print(f"\nDemonstration Game {demo + 1}:")
            board = np.zeros((3, 3), dtype=int)
            
            # Play a complete game
            for move_num in range(9):
                winner = self.check_winner(board)
                if winner is not None:
                    break
                
                if move_num % 2 == 0:  # MENACE's turn
                    move = self.choose_move(board)
                    if move is not None:
                        board[move // 3, move % 3] = 1
                        print(f"Move {move_num + 1}: MENACE plays at position {move}")
                        self.print_board(board)
                else:  # Random opponent
                    available_moves = self.get_available_moves(board)
                    if available_moves:
                        move = random.choice(available_moves)
                        board[move // 3, move % 3] = -1
                        print(f"Move {move_num + 1}: Opponent plays at position {move}")
                        self.print_board(board)
            
            winner = self.check_winner(board)
            if winner == 1:
                print("RESULT: MENACE WINS!")
            elif winner == -1:
                print("RESULT: OPPONENT WINS!")
            else:
                print("RESULT: DRAW!")
            
            # Clear history for next demonstration
            self.history = []

    def analyze_learning(self, results):
        """Analyze and plot learning progress"""
        # Convert results to numerical values for plotting
        numeric_results = []
        for result in results:
            if result == 'win':
                numeric_results.append(1)
            elif result == 'loss':
                numeric_results.append(-1)
            else:
                numeric_results.append(0)
        
        # Calculate moving average
        window_size = 50
        moving_avg = []
        for i in range(len(numeric_results)):
            if i < window_size:
                moving_avg.append(np.mean(numeric_results[:i+1]))
            else:
                moving_avg.append(np.mean(numeric_results[i-window_size+1:i+1]))
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(moving_avg)
        plt.title('MENACE Learning Progress')
        plt.xlabel('Game Number')
        plt.ylabel('Performance (Moving Average)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        final_results = results[-100:] if len(results) >= 100 else results
        win_rate = final_results.count('win') / len(final_results) * 100
        loss_rate = final_results.count('loss') / len(final_results) * 100
        draw_rate = final_results.count('draw') / len(final_results) * 100
        
        labels = ['Wins', 'Losses', 'Draws']
        sizes = [win_rate, loss_rate, draw_rate]
        colors = ['green', 'red', 'blue']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Final Performance Distribution')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n=== LEARNING ANALYSIS ===")
        print(f"Total games played: {len(results)}")
        print(f"Final 100 games performance:")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Loss rate: {loss_rate:.1f}%")
        print(f"  Draw rate: {draw_rate:.1f}%")

# Main execution
if __name__ == "__main__":
    print("=== MENACE (Machine Educable Noughts And Crosses Engine) ===")
    print("Training an AI using matchboxes and beads...")
    
    # Initialize and train MENACE
    menace = MENACE()
    
    print("\nStarting training...")
    results = menace.train(500)
    
    print(f"\nTraining completed. Final game result: {results[-1]}")
    
    # Analyze learning
    menace.analyze_learning(results)
    
    # Demonstrate learned behavior
    menace.demonstrate_learning(3)
    
    # Show some statistics
    print(f"\n=== STATISTICS ===")
    print(f"Number of matchboxes (learned states): {len(menace.matchboxes)}")
    print(f"Most common bead distribution example:")
    
    # Show one example matchbox
    if menace.matchboxes:
        example_state = list(menace.matchboxes.keys())[0]
        print(f"State: {example_state}")
        print(f"Beads: {menace.matchboxes[example_state]}")