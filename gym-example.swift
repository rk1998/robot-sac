// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import PythonKit
import TensorFlow

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @differentiable(wrt: self)
  public func dimensionGathering<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>
  ) -> Tensor {
    return _Raw.gatherNd(params: self, indices: indices)
  }

  /// Derivative of `_Raw.gatherNd`.
  ///
  /// Ported from TensorFlow Python reference implementation:
  /// https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/python/ops/array_grad.py#L691-L701
  @inlinable
  @derivative(of: dimensionGathering)
  func _vjpDimensionGathering<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let shapeTensor = Tensor<Index>(self.shapeTensor)
    let value = _Raw.gatherNd(params: self, indices: indices)
    return (
      value,
      { v in
        let dparams = _Raw.scatterNd(indices: indices, updates: v, shape: shapeTensor)
        return dparams
      }
    )
  }
}

/// Replay buffer to store the agent's experiences.
///
/// Vanilla Q-learning only trains on the latest experience. Deep Q-network uses
/// a technique called "experience replay", where all experience is stored into
/// a replay buffer. By storing experience, the agent can reuse the experiences
/// and also train in batches. For more information, check Human-level control
/// through deep reinforcement learning (Mnih et al., 2015).
class ReplayBuffer {
  /// The maximum size of the replay buffer. When the replay buffer is full,
  /// new elements replace the oldest element in the replay buffer.
  let capacity: Int
  /// If enabled, uses Combined Experience Replay (CER) sampling instead of the
  /// uniform random sampling in the original DQN paper. Original DQN samples
  /// batch uniformly randomly in the replay buffer. CER always includes the
  /// most recent element and samples the rest of the batch uniformly randomly.
  /// This makes the agent more robust to different replay buffer capacities.
  /// For more information about Combined Experience Replay, check A Deeper Look
  /// at Experience Replay (Zhang and Sutton, 2017).
  let combined: Bool

  /// The states that the agent observed.
  @noDerivative var states: [Tensor<Float>] = []
  /// The actions that the agent took.
  @noDerivative var actions: [Tensor<Int32>] = []
  /// The rewards that the agent received from the environment after taking
  /// an action.
  @noDerivative var rewards: [Tensor<Float>] = []
  /// The next states that the agent received from the environment after taking
  /// an action.
  @noDerivative var nextStates: [Tensor<Float>] = []
  /// The episode-terminal flag that the agent received after taking an action.
  @noDerivative var isDones: [Tensor<Bool>] = []
  /// The current size of the replay buffer.
  var count: Int { return states.count }

  init(capacity: Int, combined: Bool) {
    self.capacity = capacity
    self.combined = combined
  }

  func append(
    state: Tensor<Float>,
    action: Tensor<Int32>,
    reward: Tensor<Float>,
    nextState: Tensor<Float>,
    isDone: Tensor<Bool>
  ) {
    if count >= capacity {
      // Erase oldest SARS if the replay buffer is full
      states.removeFirst()
      actions.removeFirst()
      rewards.removeFirst()
      nextStates.removeFirst()
      isDones.removeFirst()
    }
    states.append(state)
    actions.append(action)
    rewards.append(reward)
    nextStates.append(nextState)
    isDones.append(isDone)
  }

  func sample(batchSize: Int) -> (
    stateBatch: Tensor<Float>,
    actionBatch: Tensor<Int32>,
    rewardBatch: Tensor<Float>,
    nextStateBatch: Tensor<Float>,
    isDoneBatch: Tensor<Bool>
  ) {
    let indices: Tensor<Int32>
    if self.combined == true {
      // Combined Experience Replay
      let sampledIndices = (0..<batchSize - 1).map { _ in Int32.random(in: 0..<Int32(count)) }
      indices = Tensor<Int32>(shape: [batchSize], scalars: sampledIndices + [Int32(count) - 1])
    } else {
      // Vanilla Experience Replay
      let sampledIndices = (0..<batchSize).map { _ in Int32.random(in: 0..<Int32(count)) }
      indices = Tensor<Int32>(shape: [batchSize], scalars: sampledIndices)
    }

    let stateBatch = Tensor(stacking: states).gathering(atIndices: indices, alongAxis: 0)
    let actionBatch = Tensor(stacking: actions).gathering(atIndices: indices, alongAxis: 0)
    let rewardBatch = Tensor(stacking: rewards).gathering(atIndices: indices, alongAxis: 0)
    let nextStateBatch = Tensor(stacking: nextStates).gathering(atIndices: indices, alongAxis: 0)
    let isDoneBatch = Tensor(stacking: isDones).gathering(atIndices: indices, alongAxis: 0)

    return (stateBatch, actionBatch, rewardBatch, nextStateBatch, isDoneBatch)
  }
}


// Initialize Python. This comment is a hook for internal use, do not remove.
// Force unwrapping with `!` does not provide source location when unwrapping `nil`, so we instead
// make a utility function for debuggability.
extension Optional {
  fileprivate func unwrapped(file: StaticString = #filePath, line: UInt = #line) -> Wrapped {
    guard let unwrapped = self else {
      fatalError("Value is nil", file: (file), line: line)
    }
    return unwrapped
  }
}

/// A Deep Q-Network.
///
/// A Q-network is a neural network that receives the observation (state) as input and estimates
/// the action values (Q values) of each action. For more information, check Human-level control
/// through deep reinforcement learning (Mnih et al., 2015).
struct DeepQNetwork: Layer {
  typealias Input = Tensor<Float>
  typealias Output = Tensor<Float>

  var l1, l2: Dense<Float>

  init(observationSize: Int, hiddenSize: Int, actionCount: Int) {
    l1 = Dense<Float>(inputSize: observationSize, outputSize: hiddenSize, activation: relu)
    l2 = Dense<Float>(inputSize: hiddenSize, outputSize: actionCount, activation: identity)
  }

  @differentiable
  func callAsFunction(_ input: Input) -> Output {
    return input.sequenced(through: l1, l2)
  }
}

/// Agent that uses the Deep Q-Network.
///
/// Deep Q-Network is an algorithm that trains a Q-network that estimates the action values of
/// each action given an observation (state). The Q-network is trained iteratively using the
/// Bellman equation. For more information, check Human-level control through deep reinforcement
/// learning (Mnih et al., 2015).
class DeepQNetworkAgent {
  /// The Q-network uses to estimate the action values.
  var qNet: DeepQNetwork
  /// The copy of the Q-network updated less frequently to stabilize the
  /// training process.
  var targetQNet: DeepQNetwork
  /// The optimizer used to train the Q-network.
  let optimizer: Adam<DeepQNetwork>
  /// The replay buffer that stores experiences of the interactions between the
  /// agent and the environment. The Q-network is trained from experiences
  /// sampled from the replay buffer.
  let replayBuffer: ReplayBuffer
  /// The discount factor that measures how much to weight to give to future
  /// rewards when calculating the action value.
  let discount: Float
  /// The minimum replay buffer size before the training starts.
  let minBufferSize: Int
  /// If enabled, uses the Double DQN update equation instead of the original
  /// DQN equation. This mitigates the overestimation problem of DQN. For more
  /// information about Double DQN, check Deep Reinforcement Learning with
  /// Double Q-learning (Hasselt, Guez, and Silver, 2015).
  let doubleDQN: Bool
  let device: Device

  init(
    qNet: DeepQNetwork,
    targetQNet: DeepQNetwork,
    optimizer: Adam<DeepQNetwork>,
    replayBuffer: ReplayBuffer,
    discount: Float,
    minBufferSize: Int,
    doubleDQN: Bool,
    device: Device
  ) {
    self.qNet = qNet
    self.targetQNet = targetQNet
    self.optimizer = optimizer
    self.replayBuffer = replayBuffer
    self.discount = discount
    self.minBufferSize = minBufferSize
    self.doubleDQN = doubleDQN
    self.device = device

    // Copy Q-network to Target Q-network before training
    updateTargetQNet(tau: 1)
  }

  func getAction(state: Tensor<Float>, epsilon: Float) -> Tensor<Int32> {
    if Float(np.random.uniform()).unwrapped() < epsilon {
      return Tensor<Int32>(numpy: np.array(np.random.randint(0, 2), dtype: np.int32))!
    } else {
      // Neural network input needs to be 2D
      let tfState = Tensor<Float>(numpy: np.expand_dims(state.makeNumpyArray(), axis: 0))!
      let qValues = qNet(tfState)[0]
      return Tensor<Int32>(qValues[1].scalarized() > qValues[0].scalarized() ? 1 : 0, on: device)
    }
  }

  func train(batchSize: Int) -> Float {
    // Don't train if replay buffer is too small
    if replayBuffer.count >= minBufferSize {
      let (tfStateBatch, tfActionBatch, tfRewardBatch, tfNextStateBatch, tfIsDoneBatch) =
        replayBuffer.sample(batchSize: batchSize)

      let (loss, gradients) = valueWithGradient(at: qNet) { qNet -> Tensor<Float> in
        // Compute prediction batch
        let npActionBatch = tfActionBatch.makeNumpyArray()
        let npFullIndices = np.stack(
          [np.arange(batchSize, dtype: np.int32), npActionBatch], axis: 1)
        let tfFullIndices = Tensor<Int32>(numpy: npFullIndices)!
        let stateQValueBatch = qNet(tfStateBatch)
        print("stateqvalue")
        print(stateQValueBatch.shape)
        let predictionBatch = stateQValueBatch.dimensionGathering(atIndices: tfFullIndices)
        print("predictions")
        print(predictionBatch.shape)

        // Compute target batch
        let nextStateQValueBatch: Tensor<Float>
        if self.doubleDQN == true {
          // Double DQN
          let npNextStateActionBatch = self.qNet(tfNextStateBatch).argmax(squeezingAxis: 1)
            .makeNumpyArray()
          let npNextStateFullIndices = np.stack(
            [np.arange(batchSize, dtype: np.int32), npNextStateActionBatch], axis: 1)
          let tfNextStateFullIndices = Tensor<Int32>(numpy: npNextStateFullIndices)!
          nextStateQValueBatch = self.targetQNet(tfNextStateBatch).dimensionGathering(
            atIndices: tfNextStateFullIndices)
        } else {
          // DQN
          nextStateQValueBatch = self.targetQNet(tfNextStateBatch).max(squeezingAxes: 1)
        }
        let targetBatch: Tensor<Float> =
          tfRewardBatch + self.discount * (1 - Tensor<Float>(tfIsDoneBatch)) * nextStateQValueBatch

        return huberLoss(
          predicted: predictionBatch,
          expected: targetBatch,
          delta: 1
        )
      }
      optimizer.update(&qNet, along: gradients)

      return loss.scalarized()
    }
    return 0
  }

  func updateTargetQNet(tau: Float) {
    self.targetQNet.l1.weight =
      tau * Tensor<Float>(self.qNet.l1.weight) + (1 - tau) * self.targetQNet.l1.weight
    self.targetQNet.l1.bias =
      tau * Tensor<Float>(self.qNet.l1.bias) + (1 - tau) * self.targetQNet.l1.bias
    self.targetQNet.l2.weight =
      tau * Tensor<Float>(self.qNet.l2.weight) + (1 - tau) * self.targetQNet.l2.weight
    self.targetQNet.l2.bias =
      tau * Tensor<Float>(self.qNet.l2.bias) + (1 - tau) * self.targetQNet.l2.bias
  }
}

let np = Python.import("numpy")
let gym = Python.import("gym")
let plt = Python.import("matplotlib.pyplot")

class TensorFlowEnvironmentWrapper {
  let originalEnv: PythonObject

  init(_ env: PythonObject) {
    self.originalEnv = env
  }

  func reset() -> Tensor<Float> {
    let state = self.originalEnv.reset()
    return Tensor<Float>(numpy: np.array(state, dtype: np.float32))!
  }

  func step(_ action: Tensor<Int32>) -> (
    state: Tensor<Float>, reward: Tensor<Float>, isDone: Tensor<Bool>, info: PythonObject
  ) {
    let (state, reward, isDone, info) = originalEnv.step(action.scalarized()).tuple4
    let tfState = Tensor<Float>(numpy: np.array(state, dtype: np.float32))!
    let tfReward = Tensor<Float>(numpy: np.array(reward, dtype: np.float32))!
    let tfIsDone = Tensor<Bool>(numpy: np.array(isDone, dtype: np.bool))!
    return (tfState, tfReward, tfIsDone, info)
  }
}

func evaluate(_ agent: DeepQNetworkAgent) -> Float {
  let evalEnv = TensorFlowEnvironmentWrapper(gym.make("CartPole-v0"))
  var evalEpisodeReturn: Float = 0
  var state: Tensor<Float> = evalEnv.reset()
  var reward: Tensor<Float>
  var evalIsDone: Tensor<Bool> = Tensor<Bool>(false)
  while evalIsDone.scalarized() == false {
    let action = agent.getAction(state: state, epsilon: 0)
    (state, reward, evalIsDone, _) = evalEnv.step(action)
    evalEpisodeReturn += reward.scalarized()
  }

  return evalEpisodeReturn
}

// Hyperparameters
/// The size of the hidden layer of the 2-layer Q-network. The network has the
/// shape observationSize - hiddenSize - actionCount.
let hiddenSize: Int = 100
/// Maximum number of episodes to train the agent. The training is terminated
/// early if maximum score is achieved during evaluation.
let maxEpisode: Int = 1000
/// The initial epsilon value. With probability epsilon, the agent chooses a
/// random action instead of the action that it thinks is the best.
let epsilonStart: Float = 1
/// The terminal epsilon value.
let epsilonEnd: Float = 0.01
/// The decay rate of epsilon.
let epsilonDecay: Float = 1000
/// The learning rate for the Q-network.
let learningRate: Float = 0.001
/// The discount factor. This measures how much to "discount" the future rewards
/// that the agent will receive. The discount factor must be from 0 to 1
/// (inclusive). Discount factor of 0 means that the agent only considers the
/// immediate reward and disregards all future rewards. Discount factor of 1
/// means that the agent values all rewards equally, no matter how distant
/// in the future they may be.
let discount: Float = 0.99
/// If enabled, uses the Double DQN update equation instead of the original DQN
/// equation. This mitigates the overestimation problem of DQN. For more
/// information about Double DQN, check Deep Reinforcement Learning with Double
/// Q-learning (Hasselt, Guez, and Silver, 2015).
let useDoubleDQN: Bool = true
/// The maximum size of the replay buffer. If the replay buffer is full, the new
/// element replaces the oldest element.
let replayBufferCapacity: Int = 100000
/// The minimum replay buffer size before the training starts. Must be at least
/// the training batch size.
let minBufferSize: Int = 64
/// The training batch size.
let batchSize: Int = 64
/// If enabled, uses Combined Experience Replay (CER) sampling instead of the
/// uniform random sampling in the original DQN paper. Original DQN samples
/// batch uniformly randomly in the replay buffer. CER always includes the most
/// recent element and samples the rest of the batch uniformly randomly. This
/// makes the agent more robust to different replay buffer capacities. For more
/// information about Combined Experience Replay, check A Deeper Look at
/// Experience Replay (Zhang and Sutton, 2017).
let useCombinedExperienceReplay: Bool = true
/// The number of steps between target network updates. The target network is
/// a copy of the Q-network that is updated less frequently to stabilize the
/// training process.
let targetNetUpdateRate: Int = 5
/// The update rate for target network. In the original DQN paper, the target
/// network is updated to be the same as the Q-network. Soft target network
/// only updates the target network slightly towards the direction of the
/// Q-network. The softTargetUpdateRate of 0 means that the target network is
/// not updated at all, and 1 means that soft target network update is disabled.
let softTargetUpdateRate: Float = 0.05

// Setup device
let device: Device = Device.default

// Initialize environment
let env = TensorFlowEnvironmentWrapper(gym.make("CartPole-v0"))

// Initialize agent
var qNet = DeepQNetwork(observationSize: 4, hiddenSize: hiddenSize, actionCount: 2)
var targetQNet = DeepQNetwork(observationSize: 4, hiddenSize: hiddenSize, actionCount: 2)
let optimizer = Adam(for: qNet, learningRate: learningRate)
var replayBuffer = ReplayBuffer(
  capacity: replayBufferCapacity,
  combined: useCombinedExperienceReplay
)
var agent = DeepQNetworkAgent(
  qNet: qNet,
  targetQNet: targetQNet,
  optimizer: optimizer,
  replayBuffer: replayBuffer,
  discount: discount,
  minBufferSize: minBufferSize,
  doubleDQN: useDoubleDQN,
  device: device
)

// RL Loop
var stepIndex = 0
var episodeIndex = 0
var episodeReturn: Float = 0
var episodeReturns: [Float] = []
var losses: [Float] = []
var state = env.reset()
var bestReturn: Float = 0
while episodeIndex < maxEpisode {
  stepIndex += 1

  // Interact with environment
  let epsilon: Float =
    epsilonEnd + (epsilonStart - epsilonEnd) * exp(-1.0 * Float(stepIndex) / epsilonDecay)
  let action = agent.getAction(state: state, epsilon: epsilon)
  let (nextState, reward, isDone, _) = env.step(action)
  episodeReturn += reward.scalarized()

  // Save interaction to replay buffer
  replayBuffer.append(
    state: state, action: action, reward: reward, nextState: nextState, isDone: isDone)

  // Train agent
  losses.append(agent.train(batchSize: batchSize))

  // Periodically update Target Net
  if stepIndex % targetNetUpdateRate == 0 {
    agent.updateTargetQNet(tau: softTargetUpdateRate)
  }

  // End-of-episode
  if isDone.scalarized() == true {
    state = env.reset()
    episodeIndex += 1
    let evalEpisodeReturn = evaluate(agent)
    episodeReturns.append(evalEpisodeReturn)
    if evalEpisodeReturn > bestReturn {
      print(
        String(
          format: "Episode: %4d | Step %6d | Epsilon: %.03f | Train: %3d | Eval: %3d", episodeIndex,
          stepIndex, epsilon, Int(episodeReturn), Int(evalEpisodeReturn)))
      bestReturn = evalEpisodeReturn
    }
    if evalEpisodeReturn > 399 {
      print("Solved in \(episodeIndex) episodes with \(stepIndex) steps!")
      break
    }
    episodeReturn = 0
  }

  // End-of-step
  state = nextState
}

// Save learning curve
plt.plot(episodeReturns)
plt.title("Deep Q-Network on CartPole-v0")
plt.xlabel("Episode")
plt.ylabel("Episode Return")
plt.savefig("dqnEpisodeReturns.png")
plt.clf()

// Save smoothed learning curve
let runningMeanWindow: Int = 10
let smoothedEpisodeReturns = np.convolve(
  episodeReturns, np.ones((runningMeanWindow)) / np.array(runningMeanWindow, dtype: np.int32),
  mode: "same")

plt.plot(episodeReturns)
plt.title("Deep Q-Network on CartPole-v0")
plt.xlabel("Episode")
plt.ylabel("Smoothed Episode Return")
plt.savefig("dqnSmoothedEpisodeReturns.png")
plt.clf()

// // Save TD loss curve
plt.plot(losses)
plt.title("Deep Q-Network on CartPole-v0")
plt.xlabel("Step")
plt.ylabel("TD Loss")
plt.savefig("/tmp/dqnTDLoss.png")
plt.clf()
