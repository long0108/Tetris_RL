import numpy as np
import random
from collections import deque
from keras.models import Sequential, load_model, clone_model
from keras.layers import Dense
from keras.optimizers import Adam
from typing import Tuple, List, Union, Optional, Dict, Any

class DQNAgent:
    def __init__(self, state_size: int, mem_size: int = 10000, discount: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_stop_episode: int = 1000,
                 n_neurons: List[int] = [32, 32], activations: List[str] = ['relu', 'relu', 'linear'],
                 loss: str = 'mse', optimizer: Optional[Adam] = None, replay_start_size: Optional[int] = None[str] = None):

        if not isinstance(state_size, int) or state_size <= 0:
             raise ValueError("state_size must be a positive integer")
        if not (0 <= discount <= 1):
             raise ValueError("discount must be between 0 and 1")
        if not (0 <= epsilon <= 1):
             raise ValueError("epsilon must be between 0 and 1")
        if not (0 <= epsilon_min <= 1) or epsilon_min > epsilon:
             raise ValueError("epsilon_min must be between 0 and 1 and <= epsilon")
        if epsilon_stop_episode < 0:
             raise ValueError("epsilon_stop_episode must be non-negative")
        if mem_size <= 0:
             raise ValueError("mem_size must be > 0")
        if not n_neurons or not all(isinstance(n, int) and n > 0 for n in n_neurons):
             raise ValueError("n_neurons must be a list of positive integers")
        if len(activations) != len(n_neurons) + 1:
             raise ValueError("activations must have length len(n_neurons) + 1")
        if activations[-1] != 'linear':
             print("Warning: Final activation is not 'linear'. Recommended for value prediction.")


        self.state_size: int = state_size
        self.discount: float = discount
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = (epsilon - epsilon_min) / epsilon_stop_episode if epsilon_stop_episode > 0 and epsilon > epsilon_min else 0
        self.mem_size: int = mem_size
        self.memory: deque = deque(maxlen=mem_size)
        self.replay_start_size: int = replay_start_size or mem_size // 2
        if self.replay_start_size > self.mem_size:
             print(f"Warning: replay_start_size ({self.replay_start_size}) is greater than mem_size ({self.mem_size}). Setting replay_start_size = mem_size.")
             self.replay_start_size = self.mem_size


        self.target_update_freq: int = target_update_freq
        self.train_step: int = 0

        self.loss: str = loss
        self.optimizer: Adam = optimizer or Adam(learning_rate=0.0003)

        if modelFile:
            try:
                self.model = load_model(modelFile)
                if self.model.input_shape[1] != self.state_size:
                     print(f"Warning: Loaded model input shape ({self.model.input_shape[1]}) does not match expected state_size ({self.state_size}). Proceeding but may cause errors.")
                if self.model.output_shape[1] != 1:
                     print(f"Warning: Loaded model output shape ({self.model.output_shape[1]}) is not 1. Proceeding but may cause errors.")

            except Exception as e:
                print(f"Error loading model from {modelFile}: {e}")
                print("Building a new model instead.")
                self.model = self._build_model(n_neurons, activations)
        else:
            self.model = self._build_model(n_neurons, activations)

        self.target_model: Sequential = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        print("Model Architecture (MLP):")
        self.model.summary()


    def _build_model(self, n_neurons: List[int], activations: List[str]) -> Sequential:
        model = Sequential()
        model.add(Dense(n_neurons[0], input_dim=self.state_size, activation=activations[0]))
        for i in range(1, len(n_neurons)):
            model.add(Dense(n_neurons[i], activation=activations[i]))
        model.add(Dense(1, activation=activations[-1]))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def predict_value(self, state: np.ndarray, target: bool = False) -> float:
        model_to_use = self.target_model if target else self.model
        state_reshaped = np.reshape(state, [1, self.state_size])
        predicted_value = model_to_use.predict(state_reshaped, verbose=0)[0][0]
        return predicted_value

    def best_state(self, next_possible_moves: Dict[Tuple[int, int], np.ndarray]) -> Optional[Tuple[int, int]]:
        if not next_possible_moves:
            return None

        actions = list(next_possible_moves.keys())

        if random.random() <= self.epsilon:
            return random.choice(actions)
        else:
            state_vectors_batch = np.array(list(next_possible_moves.values()))
            predicted_values = self.model.predict(state_vectors_batch, verbose=0).flatten()
            best_action_idx = np.argmax(predicted_values)
            return actions[best_action_idx]

    def add_to_memory(self, state: np.ndarray, next_state: np.ndarray, reward: float, done: bool) -> None:
        self.memory.append((state, next_state, reward, done))


    def train(self, batch_size: int = 32, epochs: int = 1) -> Optional[float]:
        if len(self.memory) < max(self.replay_start_size, batch_size):
            return None

        batch = random.sample(self.memory, batch_size)

        states = np.array([b[0] for b in batch])
        next_states = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        dones = np.array([b[3] for b in batch], dtype=np.bool_)

        next_qs = self.target_model.predict(next_states, verbose=0).flatten()
        target_q_values = rewards + self.discount * next_qs * (~dones).astype(np.float32)
        target_q_values_reshaped = target_q_values.reshape(-1, 1)

        history = self.model.fit(states, target_q_values_reshaped,
                                 batch_size=batch_size, epochs=epochs, verbose=0)

        if self.epsilon_decay > 0 and self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        if history and history.history and 'loss' in history.history:
            return history.history['loss'][-1]
        else:
            return None


    def save_model(self, path: str) -> None:
        try:
            self.model.save(path)
            print(f"Model saved successfully to {path}")
        except Exception as e:
            print(f"Error saving model to {path}: {e}")