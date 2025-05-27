from dqn_agent import DQNAgent # Assuming this is the MLP version now
from tetris import Tetris # Assuming this is the 3-state feature version now
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard # Assuming this is the custom logger
from tqdm import tqdm
import numpy as np
from collections import deque
import tensorflow as tf # Needed for GPU config
from typing import Optional, List, Tuple, Union, Dict, Any # Add imports for typing


# --- Hyperparameters ---
EPISODES: int = 3000 # Number of training episodes
MAX_STEPS_PER_EPISODE: Optional[int] = 1500 # Limit steps per episode to add diversity to replay buffer
EPSILON_STOP_EPISODE_RATIO: float = 0.6 # Ratio of episodes for epsilon decay
MEMORY_SIZE: int = 20000 # Size of replay memory
DISCOUNT_FACTOR: float = 0.99 # Discount factor (gamma)
BATCH_SIZE: int = 64 # Batch size for training
EPOCHS_PER_TRAIN_STEP: int = 1 # Number of epochs to train on each batch

RENDER_EVERY_N_EPISODES: int = 0 # Set > 0 to render training episodes periodically (very slow!)
RENDER_DELAY_TRAINING: float = 0.001 # Delay for rendering training games

LOG_EVERY_N_EPISODES: int = 50 # Frequency to log training stats to console and TensorBoard
REPLAY_START_SIZE_FRACTION: float = 0.1 # Fraction of memory size needed before training starts
TRAIN_EVERY_N_EPISODES: int = 1 # Frequency to trigger training (every episode)
SAVE_BEST_MODEL: bool = True # Save best model based on training avg score

# --- Evaluation Settings (Copied from CNN version) ---
EVALUATE_EVERY_N_EPISODES: int = 50 # Frequency to run evaluation games
NUM_EVAL_GAMES: int = 10 # Number of games per evaluation run
RENDER_EVAL_GAMES: bool = True # Whether to render evaluation games
RENDER_DELAY_EVAL: float = 0.01 # Delay for rendering evaluation games

# --- Model Architecture (MLP based on 3 state features) ---
# These should match what the MLP DQNAgent expects
# n_neurons should be list of hidden layer sizes
N_NEURONS: List[int] = [64, 64] # Example: 2 hidden layers with 64 neurons each
# activations should be list of activations for all layers, including input and output
# len(activations) == len(n_neurons) + 1
ACTIVATIONS: List[str] = ['relu', 'relu', 'linear'] # relu for hidden, linear for output

LEARNING_RATE_DQN: float = 0.0001 # Learning rate for the optimizer
TARGET_NETWORK_UPDATE_FREQUENCY: int = 250 # Frequency to update target network weights

def configure_gpu_memory_growth() -> None:
    """Configures GPU memory growth for TensorFlow."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured for memory growth.")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

# --- Copied and Adapted evaluate_agent function from run-cnn.py ---
def evaluate_agent(agent: DQNAgent, env: Tetris, num_games: int, logger: Optional[CustomTensorBoard], episode: int, render: bool, render_delay: Optional[float]) -> Dict[str, float]:
    """
    Runs evaluation games using the main model (epsilon=0).
    Collects detailed stats and logs them.

    Args:
        agent (DQNAgent): The DQN agent to evaluate.
        env (Tetris): The evaluation environment (should be a separate instance).
        num_games (int): Number of games to play for evaluation.
        logger (Optional[CustomTensorBoard]): The logger for TensorBoard.
        episode (int): The current training episode number (used for logging step).
        render (bool): Whether to render the evaluation games.
        render_delay (Optional[float]): Delay between steps when rendering.

    Returns:
        Dict[str, float]: Dictionary containing average evaluation statistics.
    """
    eval_scores: List[int] = []
    eval_lines_cleared: List[int] = []
    eval_pieces_played: List[int] = []
    eval_clears_1: List[int] = []
    eval_clears_2: List[int] = []
    eval_clears_3: List[int] = []
    eval_clears_4: List[int] = []


    print(f"\n--- Starting Evaluation after Episode {episode} ({num_games} games) ---")

    # Store original epsilon and set to 0 for pure exploitation during evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for game_idx in range(num_games):
        # Reset environment for a new game. Returns the initial state vector.
        current_state_vector: Optional[np.ndarray] = env.reset()
        done: bool = False
        steps_this_game: int = 0 # Count pieces played in this evaluation game

        # Game loop for a single evaluation game
        while not done and (MAX_STEPS_PER_EPISODE is None or steps_this_game < MAX_STEPS_PER_EPISODE):
             # Check if current state is valid (e.g., not None)
             if current_state_vector is None:
                 done = True
                 break

             # Get all possible next states for the current piece and their corresponding actions (final placements)
             # env.get_next_states() in the 3-state version returns Dict[Tuple[int, int], np.ndarray]
             # where the key is the action (x, rotation) and the value is the resulting state vector
             next_possible_moves: Dict[Tuple[int, int], np.ndarray] = env.get_next_states()

             # If no possible moves (shouldn't happen unless board is full in unusual way or game over on spawn)
             if not next_possible_moves:
                 done = True
                 break

             # Choose the best action (x, rotation) using the agent's model (epsilon is 0)
             # agent.best_state now expects Dict[Tuple[int, int], np.ndarray] and returns Tuple[int, int]
             chosen_action: Optional[Tuple[int, int]] = agent.best_state(next_possible_moves)

             # If agent couldn't select an action (shouldn't happen if next_possible_moves is not empty)
             if chosen_action is None:
                 done = True
                 break

             # Execute the chosen action in the environment
             # env.play() in the 3-state version returns (reward, done, lines_cleared_this_step, next_state_vector)
             # We need to assign the return values correctly based on this signature
             reward_from_play, done, lines_cleared_this_step, next_state_vector_from_env_play = env.play(
                 chosen_action[0], chosen_action[1],
                 render=render,
                 render_delay=render_delay # Use evaluation render delay
             )

             # Update the current state for the next step in this game
             # The state for the next decision is the state *after* the piece landed and the new piece spawned
             current_state_vector = next_state_vector_from_env_play
             steps_this_game += 1 # Count pieces played


        # Collect episode stats after the game is over
        # These methods retrieve stats accumulated by the env instance during this game
        eval_scores.append(env.get_game_score())
        eval_lines_cleared.append(env.get_lines_cleared_this_episode())
        eval_pieces_played.append(steps_this_game)
        eval_clears_1.append(env.get_clears_1_this_episode())
        eval_clears_2.append(env.get_clears_2_this_episode())
        eval_clears_3.append(env.get_clears_3_this_episode())
        eval_clears_4.append(env.get_clears_4_this_episode())


        # Print stats for the current evaluation game
        print(f"  Eval Game {game_idx + 1}/{num_games}: Score = {env.get_game_score()}, Lines = {env.get_lines_cleared_this_episode()}, Pieces = {steps_this_game}, 1-Clear={env.get_clears_1_this_episode()}, 2-Clear={env.get_clears_2_this_episode()}, 3-Clear={env.get_clears_3_this_episode()}, 4-Clear={env.get_clears_4_this_episode()}")


    # Restore original epsilon after evaluation is complete
    agent.epsilon = original_epsilon
    print("--- Evaluation Complete ---")

    # Calculate average stats over all evaluation games
    avg_eval_score = mean(eval_scores) if eval_scores else 0
    avg_eval_lines_cleared = mean(eval_lines_cleared) if eval_lines_cleared else 0
    avg_eval_pieces_played = mean(eval_pieces_played) if eval_pieces_played else 0
    avg_eval_clears_1 = mean(eval_clears_1) if eval_clears_1 else 0
    avg_eval_clears_2 = mean(eval_clears_2) if eval_clears_2 else 0
    avg_eval_clears_3 = mean(eval_clears_3) if eval_clears_3 else 0
    avg_eval_clears_4 = mean(eval_clears_4) if eval_clears_4 else 0


    # Log evaluation stats to TensorBoard
    eval_stats = {
        'evaluation/avg_score': avg_eval_score,
        'evaluation/avg_lines_cleared': avg_eval_lines_cleared,
        'evaluation/avg_pieces_played': avg_eval_pieces_played,
        'evaluation/avg_clears_1': avg_eval_clears_1,
        'evaluation/avg_clears_2': avg_eval_clears_2,
        'evaluation/avg_clears_3': avg_eval_clears_3,
        'evaluation/avg_clears_4': avg_eval_clears_4,
    }
    if logger:
        # Log with the episode number as the step
        logger.log(episode, **eval_stats)

    return eval_stats # Return stats dictionary

# --- END evaluate_agent function ---


def dqn():
    """Main training loop."""
    # Configure GPU memory growth if available
    configure_gpu_memory_growth()

    # Create separate environments for training and evaluation
    # This ensures evaluation doesn't affect the training environment state
    env = Tetris() # Training environment instance
    eval_env = Tetris() # Evaluation environment instance

    # Determine absolute episode number for epsilon decay stop
    epsilon_stop_episode_abs = int(EPISODES * EPSILON_STOP_EPISODE_RATIO)

    # Determine absolute size for replay start
    # If REPLAY_START_SIZE is set, use it, otherwise calculate from fraction
    replay_start_size_abs = int(MEMORY_SIZE * REPLAY_START_SIZE_FRACTION)


    # Initialize the DQNAgent (MLP version)
    # Pass state_size obtained from the environment (which is 3)
    agent = DQNAgent(
        state_size=env.get_state_size(), # Pass the state size (3 for MLP)
        n_neurons=N_NEURONS, # Hidden layer neuron counts
        activations=ACTIVATIONS, # Activation functions for all layers
        epsilon_stop_episode=epsilon_stop_episode_abs, # Episode to stop epsilon decay
        mem_size=MEMORY_SIZE, # Replay memory size
        discount=DISCOUNT_FACTOR, # Discount factor
        replay_start_size=replay_start_size_abs, # Number of steps before training starts
        target_update_freq=TARGET_NETWORK_UPDATE_FREQUENCY, # Target network update frequency
        # optimizer=Adam(learning_rate=LEARNING_RATE_DQN) # Can pass optimizer object, or configure within agent
        # loss='mse' # Can pass loss function string or object
    )

    # Setup TensorBoard logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Use a distinct name for MLP logs
    log_dir_name = (f'tetris-MLP-logs-{timestamp}')
    log_dir_path = f'logs/{log_dir_name}'

    logger: Optional[CustomTensorBoard] = None
    try:
        # Initialize the custom logger
        logger = CustomTensorBoard(log_dir=log_dir_path)
    except Exception as e:
        print(f"Could not initialize CustomTensorBoard: {e}. Logging will be basic (console only).")

    # Deque to store recent scores for calculating rolling average during training
    scores_window: deque = deque(maxlen=LOG_EVERY_N_EPISODES)
    # Deque to store recent losses for calculating rolling average
    losses_window: deque = deque(maxlen=LOG_EVERY_N_EPISODES)
    # Variable to keep track of the best average training score for saving the model
    best_avg_train_score: float = -np.inf

    # Main training loop iterating over episodes
    # tqdm provides a progress bar
    for episode in tqdm(range(1, EPISODES + 1), unit="ep"):
        # Reset environment for a new episode. Returns the initial state vector.
        current_state_vector: np.ndarray = env.reset()
        done: bool = False # Flag indicating if the episode is finished
        steps_this_episode: int = 0 # Count steps (pieces placed) in this episode
        current_loss: Optional[float] = None # Variable to store loss from training step

        # Determine if rendering should be active for this training episode
        render_this_episode = (RENDER_EVERY_N_EPISODES > 0 and episode % RENDER_EVERY_N_EPISODES == 0)

        # Episode loop: play game until done or max steps reached
        while not done and (MAX_STEPS_PER_EPISODE is None or steps_this_episode < MAX_STEPS_PER_EPISODE):
            # Get all possible next states (as state vectors) for the current piece
            # env.get_next_states() returns Dict[Tuple[int, int], np.ndarray]
            next_possible_moves: Dict[Tuple[int, int], np.ndarray] = env.get_next_states()

            # If no possible moves (game over on spawn or blocked)
            if not next_possible_moves:
                done = True # End episode
                break

            # Choose an action (x, rotation) using the agent's epsilon-greedy policy
            # agent.best_state now expects Dict[Tuple[int, int], np.ndarray] and returns Tuple[int, int]
            chosen_action: Optional[Tuple[int, int]] = agent.best_state(next_possible_moves)

            # If no action was chosen (shouldn't happen if next_possible_moves is not empty)
            if chosen_action is None:
                done = True
                break

            # --- Experience Collection ---
            # Get the state vector resulting from the chosen action *before* env.play updates the board
            # This is the s' used in the Bellman update R + gamma * V(s')
            # Note: In the 3-state version, get_next_states already returns the state *after* the piece lands and lines clear.
            # So, the state vector corresponding to the chosen action is directly available from next_possible_moves.
            resulting_state_vector_for_memory: np.ndarray = next_possible_moves[chosen_action]

            # Execute the chosen action in the environment
            # env.play() in the 3-state version returns (reward, done, lines_cleared_this_step, next_state_vector_after_env_update)
            # next_state_vector_after_env_update is the state *after* the piece lands, lines clear, AND a new piece is spawned.
            reward_from_play, done, lines_cleared_this_step, next_state_vector_after_env_update = env.play(
                chosen_action[0], chosen_action[1],
                render=render_this_episode,
                render_delay=RENDER_DELAY_TRAINING if render_this_episode else None # Use training render delay
            )

            # Add the experience tuple to the agent's replay memory
            # The tuple is (state_before_action, state_after_action_and_clears, reward, done)
            # Here, state_before_action is current_state_vector
            # state_after_action_and_clears is resulting_state_vector_for_memory (from get_next_states)
            # reward is reward_from_play
            # done is the done flag returned by env.play
            agent.add_to_memory(current_state_vector, resulting_state_vector_for_memory, reward_from_play, done)

            # Update the current state vector for the next iteration of the loop
            # This next state includes the newly spawned piece
            current_state_vector = next_state_vector_after_env_update
            steps_this_episode += 1 # Increment step count (pieces played)


        # Collect episode score after the game is over
        episode_score: int = env.get_game_score()
        scores_window.append(episode_score) # Add score to the rolling window

        # --- Train Agent ---
        # Trigger training periodically, only if enough data in memory
        if episode % TRAIN_EVERY_N_EPISODES == 0 and len(agent.memory) >= agent.replay_start_size:
            # agent.train() performs one training step and returns the loss
            train_loss_value = agent.train(batch_size=BATCH_SIZE, epochs=EPOCHS_PER_TRAIN_STEP)
            # Store the loss if training occurred
            if train_loss_value is not None:
                 current_loss = train_loss_value
                 losses_window.append(current_loss) # Add loss to the rolling window


        # --- Logging Training Stats ---
        # Log training stats to console and TensorBoard periodically
        if logger and episode % LOG_EVERY_N_EPISODES == 0:
            # Calculate average score and loss over the window
            avg_train_score_val = mean(scores_window) if scores_window else -np.inf
            avg_train_loss_val = mean(losses_window) if losses_window else None # Only calculate if window is not empty

            # Print current stats to console
            print(f"\n[Train] Ep: {episode}, Avg Train Score ({LOG_EVERY_N_EPISODES} ep): {avg_train_score_val:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Train Loss: {avg_train_loss_val if avg_train_loss_val is not None else 'N/A'}")

            # Prepare data for TensorBoard logging
            log_data_train = {
                'training/avg_score': avg_train_score_val,
                'training/loss': avg_train_loss_val, # Log average loss over the window
                'training/epsilon': agent.epsilon,
                'training/episode_lines_cleared_total': float(env.get_lines_cleared_this_episode()), # Total lines cleared in the just finished episode
            }
            # Log the data using the logger, with episode number as step
            logger.log(episode, **{k: v for k, v in log_data_train.items() if v is not None}) # Filter out None values


            # --- Save Best Model based on Training Avg Score ---
            if SAVE_BEST_MODEL and avg_train_score_val > best_avg_train_score and scores_window:
                print(f"New best average training score: {avg_train_score_val:.2f} (was {best_avg_train_score:.2f}). Saving model...")
                best_avg_train_score = avg_train_score_val
                # Save the model to a file
                agent.save_model("best_tetris_mlp_train_avg.keras") # Use .keras format


        # --- Evaluate Agent ---
        # Perform evaluation periodically
        if episode % EVALUATE_EVERY_N_EPISODES == 0:
             # Call the evaluate_agent function
             # Pass the evaluation environment, number of games, current episode, logger, and rendering settings
             evaluate_agent(
                 agent,
                 eval_env, # Use the dedicated evaluation environment
                 NUM_EVAL_GAMES,
                 logger,
                 episode, # Log evaluation stats at this episode number
                 RENDER_EVAL_GAMES, # Whether to render eval games
                 RENDER_DELAY_EVAL # Delay for rendering eval games
             )
             # You could add logic here to save best model based on evaluation score instead


    # --- Training Finished ---
    print("Training finished.")
    # Save the final model
    agent.save_model("final_tetris_mlp.keras") # Use .keras format

    # Close the logger
    if logger:
        logger.close()

    # Close any open OpenCV windows
    if hasattr(env, 'cv2') and env.cv2: # Check if cv2 module was used by the env
        env.cv2.destroyAllWindows()


if __name__ == "__main__":
    # Start the training process
    dqn() # Call the main training function