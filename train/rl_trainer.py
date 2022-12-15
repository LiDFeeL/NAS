from fvcore.nn import FlopCountAnalysis

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import argparse
from datetime import datetime
import logging
import multiprocessing
import os
import time
from typing import Union

from agents.lstm_agent import LSTMAgent
from exploration.rnd_model import RNDModel
from train.train_baseline import (
    HeadWoLinear, Layer, LayerType,
    load_model, train
)
from train.train_custom_head import Head, load_extracted_features
import utils

class ReplayBuffer:
    def __init__(
        self,
        size_or_from_file: Union[int, str],
        max_seq_len: int = 128
    ):
        if isinstance(size_or_from_file, int):
            self.size = size_or_from_file
            self.next_idx = 0
            self.num_in_buffer = 0

            self.heads = torch.zeros((max_seq_len, self.size)).long()
            self.rewards = torch.zeros(self.size)
            self.log_probs = torch.zeros(self.size)

        elif isinstance(size_or_from_file, str):
            data = torch.load(size_or_from_file, map_location="cpu")

            self.size = data["size"]
            self.next_idx = data["next_idx"]
            self.num_in_buffer = data["num_in_buffer"]

            self.heads = data["heads"]
            self.rewards = data["rewards"]
            self.log_probs = data["log_probs"]

        else:
            logger = logging.getLogger("rl_train")
            logger.error("Must pass in either size of or file storing replay buffer")

    def sample(self, batch_size):
        """
        Sample `batch_size` samples from the replay buffer. May produce fewer
        than `batch_size` samples if there are not sufficient number of
        samples in the buffer.
        """
        indices = torch.randperm(self.num_in_buffer)[:batch_size]

        heads = self.heads[:, indices]
        rewards = self.rewards[indices]
        log_probs = self.log_probs[indices]

        return heads, rewards, log_probs

    def store_sample(self, head, reward, log_prob):
        """
        Store a sample into the replay buffer. All tensors need to be stored
        on CPU.
        """
        self.heads[:head.size(0), self.next_idx] = head
        self.rewards[self.next_idx] = reward
        self.log_probs[self.next_idx] = log_prob

        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = max(self.next_idx + 1, self.size)

class RLTrainer:
    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        num_classes: int,
        num_gpus: int = 1,
        embedding_dim: int = 32,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 1,
        rnd_output_dim: int = 16,
        rnd_gamma: float = 0.99,
        agent_initial_lr: float = 1e-3,
        rnd_initial_lr: float = 1e-3,
        baseline_reward: float = 0,
        reward_scale: float = 1,
        sample_per_step: int = 10,
        head_max_seq_len: int = 128,
        head_batch_size: int = 128,
        exploration_steps: int = 100,
        exploitation_steps: int = 100,
        replay_buffer: ReplayBuffer = None,
        training_lr: float = 4e-3,
        training_batch_size: int = 128,
        training_epochs: int = 30,
        logdir: str = "./output/"
    ):
        args = locals()

        self.exploration_model = RNDModel(
            embedding_dim,
            lstm_hidden_dim,
            lstm_num_layers,
            rnd_output_dim,
            rnd_initial_lr
        )

        self.exploration_actor = LSTMAgent(
            0,
            1,
            head_max_seq_len,
            embedding_dim,
            lstm_hidden_dim,
            lstm_num_layers,
            agent_initial_lr
        )
        self.exploitation_actor = LSTMAgent(
            baseline_reward,
            reward_scale,
            head_max_seq_len,
            embedding_dim,
            lstm_hidden_dim,
            lstm_num_layers,
            agent_initial_lr
        )

        self.running_rnd_rew_std = 1
        self.rnd_gamma = rnd_gamma

        self.num_classes = num_classes
        self.num_gpus = num_gpus

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.sample_per_step = sample_per_step
        self.heads_batch_size = head_batch_size

        self.exploration_steps = exploration_steps
        self.exploitation_steps = exploitation_steps
        if replay_buffer is None:
            self.replay_buffer = ReplayBuffer(
                head_batch_size * sample_per_step,
                head_max_seq_len
            )
        else:
            self.replay_buffer = replay_buffer

        self.training_lr = training_lr
        self.training_batch_size = training_batch_size
        self.training_epochs = training_epochs

        # Make sure INFO-level logging is not ignored
        logging.basicConfig(level=logging.INFO)
        self.logdir = logdir
        self.logger = logging.getLogger("rl_trainer")
        self.logger.info(f"Instantiating RL trainer with hyperparameters: {args}")
        self.writer = SummaryWriter(log_dir=logdir)

    def form_head(self, head: torch.LongTensor) -> nn.Module:
        """
        Form one single head from its tensor representation.
        """
        assert head.size(0) % 4 == 0
        
        head_layers = []
        for i in range(0, head.size(0), 4):
            layer_type_id = head[i].item()
            layer_type = LayerType(layer_type_id)

            parameters = []
            num_params = self._num_params_per_layer[layer_type_id]
            for j in range(num_params):
                param_id = head[i + j].item()
                parameters.append(self._parameter_candidates[layer_type_id][j][param_id])

            layer = Layer(layer_type, tuple(parameters))
            head_layers.append(layer)

            # Avoid generating uncalled for layers after we have declared end
            if layer_type == LayerType.End:
                break

        # Hard-coded input size for AvgPool layer given 224x224 RGB images
        head_wo_linear = HeadWoLinear([1, 2048, 7, 7], head_layers)
        head = Head(self.num_classes, head_wo_linear)
        return head

    def normalize_rnd_reward(self, unnormalized_reward: torch.Tensor) -> torch.Tensor:
        self.running_rnd_rew_std = self.rnd_gamma * self.running_rnd_rew_std \
                                 + (1 - self.rnd_gamma) * unnormalized_reward.std().item()
        normalized_reward = unnormalized_reward / self.running_rnd_rew_std
        return normalized_reward

    def get_reward(self, head_tensor: torch.LongTensor) -> float:
        self.logger.info(f"Trying head: f{head_tensor}...")
        # TODO: add reward for longer heads

        with torch.no_grad():
            in_tensor = torch.zeros((1, 2048, 7, 7))
            # If head is invalid, kill it
            try:
                head = self.form_head(head_tensor).share_memory()
                head(in_tensor)
            except Exception:
                self.logger.info(
                    f"Reward: 0 (invalid head), Time: {datetime.now().ctime()}"
                )
                return 0

            # If head is egregiously large, kill it
            flop_count = FlopCountAnalysis(head, in_tensor)
            total_flops = flop_count.total()
            # Hard-coded value for ResNet50 flop count (excluding default head)
            head_flops_percentage = total_flops / (4142706176 + total_flops)
            self.logger.info(f"Head flop count percentage: {head_flops_percentage}")
            if head_flops_percentage > 0.001:
                self.logger.info(
                    f"Reward: 0 (head too large), Time: {datetime.now().ctime()}"
                )
                return 0
        # If head is invalid, kill it
        in_tensor = torch.zeros((1, 2048, 7, 7)).cuda()
        try:
            head = self.form_head(head_tensor).cuda()
            head(in_tensor)
        except Exception:
            self.logger.info(
                f"Reward: 0 (invalid head), Time: {datetime.now().ctime()}"
            )
            return 0

        # If head is egregiously large, kill it
        flop_count = FlopCountAnalysis(head, in_tensor)
        total_flops = flop_count.total()
        # Hard-coded value for ResNet50 flop count (excluding default head)
        head_flops_percentage = total_flops / (4142706176 + total_flops)
        self.logger.info(f"Head flop count percentage: {head_flops_percentage}")
        if head_flops_percentage > 0.01:
            self.logger.info(
                f"Reward: 0 (head too large), Time: {datetime.now().ctime()}"
            )
            return 0

        # Delete tensor to save GPU memory
        head = head.cpu().share_memory()
        del in_tensor

        queue = multiprocessing.Queue()
        for _ in range(self.num_gpus):
            queue.put(head)
        training_start = time.time()
        utils.launch(
            _train_head,
            self.num_gpus,
            args=(
                self.train_dataset,
                self.eval_dataset,
                self.training_lr,
                self.training_batch_size,
                self.training_epochs,
                queue
            )
        )
        training_time = time.time() - training_start

        eval_acc = queue.get()
        queue.close()
        self.logger.info(
            f"Reward: {eval_acc}, Training time: {training_time:.4f} s, "
            f"Time: {datetime.now().ctime()}"
        )
        return eval_acc

    def train(self):
        total_steps = self.exploration_steps + self.exploitation_steps
        actor = self.exploration_actor
        self.logger.info(f"Exploring... Time: {datetime.now().ctime()}")

        for t in range(total_steps):
            if t == self.exploration_steps:
                self.logger.info(
                    f"Exploiting... Time: {datetime.now().ctime()}"
                )
                actor = self.exploitation_actor

            # Sample new heads from actor
            sampled_heads = actor.sample_heads(self.heads_batch_size)
            log_probs = actor.log_prob(sampled_heads).detach()
            rewards_lst = []

            for i in range(self.heads_batch_size):
                head = sampled_heads[:, i]
                reward = self.get_reward(head)
                rewards_lst.append(reward)

                # Store sampled heads into replay buffer
                head_cpu = head.cpu().clone()
                self.replay_buffer.store_sample(
                    head_cpu, reward, log_probs[i].item()
                )

            rewards = torch.tensor(rewards_lst).float().cuda()

            explore_bonus_dict = self.exploration_model.update(sampled_heads)
            explore_bonus = explore_bonus_dict["Exploration Reward"]
            normalized_explore_bonus = self.normalize_rnd_reward(explore_bonus)

            explore = t < self.exploration_steps
            explore_loss_dict = self.exploration_actor.update(
                sampled_heads, normalized_explore_bonus, explore, log_probs
            )
            exploit_loss_dict = self.exploitation_actor.update(
                sampled_heads, rewards, not explore, log_probs
            )

            # Logging
            self.logger.info(
                f"Step: {t+1}/{total_steps}, "
                f"Exploration reward: {explore_bonus_dict['Exploration Loss']}, "
                f"Exploration actor loss: {explore_loss_dict['LSTM Agent Loss']}, "
                f"Exploitation reward: {exploit_loss_dict['Average Reward']}, "
                f"Exploitation actor loss: {exploit_loss_dict['LSTM Agent Loss']}, "
                f"Time: {datetime.now().ctime()}"
            )
            self.writer.add_scalar(
                "Exploration reward", explore_bonus_dict["Exploration Loss"], t
            )
            self.writer.add_scalar(
                "Exploration actor loss", explore_loss_dict["LSTM Agent Loss"], t
            )
            self.writer.add_scalar(
                "Exploitation reward", exploit_loss_dict["Average Reward"], t
            )
            self.writer.add_scalar(
                "Exploitation actor loss", exploit_loss_dict["LSTM Agent Loss"], t
            )

            # Sample from replay buffer to train agents
            for i in range(self.sample_per_step):
                sampled_heads, rewards, log_probs = \
                    self.replay_buffer.sample(self.heads_batch_size)

                sampled_heads = sampled_heads.cuda()
                rewards = rewards.cuda()
                log_probs = log_probs.cuda()

                explore_bonus_dict = self.exploration_model.update(sampled_heads)
                explore_bonus = explore_bonus_dict["Exploration Reward"]
                normalized_explore_bonus = self.normalize_rnd_reward(explore_bonus)

                explore_loss_dict = self.exploration_actor.update(
                    sampled_heads, normalized_explore_bonus, False, log_probs
                )
                exploit_loss_dict = self.exploitation_actor.update(
                    sampled_heads, rewards, False, log_probs
                )
                self.logger.info(
                    f"Step: {t+1}/{total_steps}, "
                    f"Replay iteration: {i+1}/{self.sample_per_step}, "
                    f"Exploration reward: {explore_bonus_dict['Exploration Loss']}, "
                    f"Exploration actor loss: {explore_loss_dict['LSTM Agent Loss']}, "
                    f"Exploitation reward: {exploit_loss_dict['Average Reward']}, "
                    f"Exploitation actor loss: {exploit_loss_dict['LSTM Agent Loss']}, "
                    f"Time: {datetime.now().ctime()}"
                )
                
                time_step = t * self.sample_per_step + i
                self.writer.add_scalar(
                    "Replay buffer exploration reward",
                    explore_bonus_dict["Exploration Loss"],
                    time_step
                )
                self.writer.add_scalar(
                    "Replay buffer exploration actor loss",
                    explore_loss_dict["LSTM Agent Loss"],
                    time_step
                )
                self.writer.add_scalar(
                    "Replay buffer exploitation reward",
                    exploit_loss_dict["Average Reward"],
                    time_step
                )
                self.writer.add_scalar(
                    "Replay buffer exploitation actor loss",
                    exploit_loss_dict["LSTM Agent Loss"],
                    time_step
                )

        # Save models
        exploration_actor_path = os.path.join(self.logdir, "exploration_actor.pth")
        self.exploration_actor.save(exploration_actor_path)
        exploitation_actor_path = os.path.join(self.logdir, "exploitation_actor.pth")
        self.exploitation_actor.save(exploitation_actor_path)

    # See train/train_baseline.py:LayerType enumeration for more details
    _num_params_per_layer = [0, 2, 3, 0, 0, 2, 2, 2, 0, 0, 0, 1, 0]

    _parameter_candidates = [
        # Start
        [],
        # Linear: output_height, output_width
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        ],
        # Conv: out_channels, kernel_size, stride
        [
            [1, 2, 4, 8, 10, 16, 32, 64, 128, 256, 512, 1024, 2048],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        ],
        # BatchNorm
        [],
        # LayerNorm
        [],
        # AvgPool: output_height, output_width
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        ],
        # MaxPool: output_height, output_width
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        ],
        # Upsampling: scale_height, scale_width
        [
            [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4],
            [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
        ],
        # Relu
        [],
        # Tanh
        [],
        # Sigmoid
        [],
        # SkipConnection: starting_layer
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ],
        # End
        []
    ]

def _train_head(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    queue: multiprocessing.Queue
):
    device = torch.cuda.current_device()
    device = torch.device("cuda", device)
    head = queue.get().to(device)
    if utils.get_world_size() > 1:
        head = DistributedDataParallel(
            head,
            device_ids=[device],
            output_device=device
        )

    eval_acc = train(
        head,
        train_dataset,
        eval_dataset,
        nn.CrossEntropyLoss(),
        lr=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        do_logging=False
    )
    if utils.is_main_process():
        queue.put(eval_acc)

def main(args):
    # Force start method so that mp queues can properly work
    multiprocessing.set_start_method("spawn")

    num_classes = utils.dataset_num_classes(args.dataset)
    pretrained_model = load_model(
        num_classes,
        head_layers="AvgPool: 1, 1",
        from_checkpoint=args.pretrained_model_path,
        multi_gpu=False
    )
    train_dataset = load_extracted_features(
        args.dataset,
        pretrained_model, 
        train=True,
        use_percent=args.use_dataset_percent,
        path_to_store=args.path_to_data
    )
    eval_dataset = load_extracted_features(
        args.dataset,
        pretrained_model, 
        train=False,
        use_percent=args.use_dataset_percent,
        path_to_store=args.path_to_data
    )
    
    rl_trainer = RLTrainer(
        train_dataset,
        eval_dataset,
        num_classes,
        args.num_gpus,
        args.embedding_dim,
        args.lstm_hidden_dim,
        args.lstm_num_layers,
        args.rnd_output_dim,
        args.rnd_gamma,
        args.agent_initial_lr,
        args.rnd_initial_lr,
        args.baseline_reward,
        args.reward_scale,
        args.sample_per_step,
        args.head_max_seq_len,
        args.head_batch_size,
        args.exploration_steps,
        args.exploitation_steps,
        None, # TODO: add argument for replay buffer
        args.training_lr,
        args.training_batch_size,
        args.training_epochs,
        args.logdir
    )
    rl_trainer.train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="cifar-10", type=str.lower)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--pretrained-model-path", "-ptm",
                        default="./output/cifar-10-baseline.pth",
                        type=str)
    parser.add_argument("--use-dataset-percent", "-udp", default=0.01, type=float)
    parser.add_argument("--path-to-data", default="./data/", type=str)

    parser.add_argument("--embedding-dim", default=32, type=int)
    parser.add_argument("--lstm-hidden-dim", default=128, type=int)
    parser.add_argument("--lstm-num-layers", default=1, type=int)

    parser.add_argument("--rnd-output-dim", default=16, type=int)
    parser.add_argument("--rnd-gamma", default=0.99, type=float)

    parser.add_argument("--agent-initial-lr", default=1e-3, type=float)
    parser.add_argument("--rnd-initial-lr", default=1e-3, type=float)

    parser.add_argument("--baseline-reward", default=0, type=float)
    parser.add_argument("--reward-scale", default=1, type=float)

    parser.add_argument("--sample-per-step", default=10, type=int)
    parser.add_argument("--head-max-seq-len", default=128, type=int)
    parser.add_argument("--head-batch-size", default=128, type=int)
    parser.add_argument("--exploration-steps", default=100, type=int)
    parser.add_argument("--exploitation-steps", default=100, type=int)

    parser.add_argument("--training-lr", default=4e-3, type=float)
    parser.add_argument("--training-batch-size", default=32, type=int)
    parser.add_argument("--training-epochs", default=10, type=int)
    parser.add_argument("--logdir", default="./output/", type=str)

    args = parser.parse_args()
    main(args)
