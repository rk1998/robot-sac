///Implementation of the DDPG (Deep Deterministic Policy Gradient) Algorithm
/// This implementation uses swift for Tensorflow and borrows ideas from
/// the swift-models repository: https://github.com/tensorflow/swift-models/
/// Original Paper:  http://arxiv.org/pdf/1509.02971v2.pdf
/// Swift for TensorFlow: https://github.com/tensorflow/swift
/// Author: Rohith Krishnan
import Foundation
import PythonKit
PythonLibrary.useVersion(3, 6)
import TensorFlow
let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
let gym = Python.import("gym")


//taken from https://github.com/tensorflow/swift-models/blob/master/Gym/DQN/
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


//Taken From https://github.com/tensorflow/swift-models/blob/master/Gym/DQN/ReplayBuffer.swift
// Replay buffer to store the agent's experiences.
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
  @noDerivative var actions: [Tensor<Float>] = []
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
    action: Tensor<Float>,
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
    actionBatch: Tensor<Float>,
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











//Taken from the DQN example in the tensorflow swift-models repo
//: https://github.com/tensorflow/swift-models/blob/master/Gym/DQN/main.swift
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

struct MultiLayeredNetwork : Layer {
  typealias Input = Tensor<Float>
  typealias Output = Tensor<Float>
  public var layers : [Dense<Float>] = []
  public var weight : [Tensor<Float>] = []
  public var bias : [Tensor<Float>] = []
  init(observationSize: Int, hiddenLayerSizes: [Int], outDimension: Int) {
        let inputSize = observationSize
        let inputLayer = Dense<Float>(inputSize: inputSize, outputSize: hiddenLayerSizes[0], activation: relu)
        self.layers.append(inputLayer)
        self.weight.append(inputLayer.weight)
        self.bias.append(inputLayer.bias)
        for i in 0..<hiddenLayerSizes.count - 1 {
          let layer_i = Dense<Float>(inputSize:hiddenLayerSizes[i], outputSize: hiddenLayerSizes[i + 1], activation: relu)
          self.layers.append(layer_i)
          self.weight.append(layer_i.weight)
          self.bias.append(layer_i.bias)
        }
        let outputLayer = Dense<Float>(inputSize: hiddenLayerSizes[hiddenLayerSizes.count - 1], outputSize: outDimension, activation: tanh)
        self.layers.append(outputLayer)
        self.weight.append(outputLayer.weight)
        self.bias.append(outputLayer.bias)
  }

  @differentiable
  func callAsFunction(_ input: Input) -> Output {
      return layers.differentiableReduce(input) { $1($0) }
  }

}


struct CriticNetwork: Layer {
  typealias Input = [Tensor<Float>]
  typealias Output = Tensor<Float>

  //var action_layer : Dense<Float>
  public var state_layer : Dense<Float>
  public var hidden_layers: MultiLayeredNetwork

  init(state_size: Int, action_size:Int, hiddenLayerSizes: [Int] = [400, 300], outDimension: Int) {
      self.state_layer = Dense<Float>(inputSize: state_size,
                            outputSize: hiddenLayerSizes[0], activation: relu)
      self.hidden_layers = MultiLayeredNetwork(observationSize: hiddenLayerSizes[0] + action_size,
                            hiddenLayerSizes: hiddenLayerSizes, outDimension: outDimension)
  }

  @differentiable
  func callAsFunction(_ input: Input) -> Output {
      let state: Tensor<Float> = input[0]
      let action: Tensor<Float> = input[1]
      let s = self.state_layer(state)
      let state_and_action = Tensor(concatenating: [s, action], alongAxis: 1)
      let q_value = self.hidden_layers(state_and_action)
      return q_value
  }

}


struct ActorNetwork: Layer {
  typealias Input = Tensor<Float>
  typealias Output = Tensor<Float>
  public var layer_1, layer_2, layer_3: Dense<Float>
  init(observationSize: Int, actionSize: Int, hiddenLayerSizes: [Int] = [400, 300]) {
      layer_1 = Dense<Float>(inputSize: observationSize, outputSize: hiddenLayerSizes[0], activation: relu)
      layer_2 = Dense<Float>(inputSize: hiddenLayerSizes[0], outputSize: hiddenLayerSizes[1], activation: relu)
      layer_3 = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: actionSize, activation: tanh)


  }

  @differentiable
  func callAsFunction(_ state: Input) -> Output {
      return state.sequenced(through: layer_1, layer_2, layer_3)
  }


}


class ActorCritic {


  public var actor_network: ActorNetwork

  public var critic_network: CriticNetwork

  public var target_critic_network: CriticNetwork

  public var target_actor_network: ActorNetwork

  let gamma: Float

  let state_size: Int

  let action_size: Int

  let lr : Float

  let actor_optimizer: Adam<ActorNetwork>

  let critic_optimizer: Adam<CriticNetwork>

  let replayBuffer: ReplayBuffer

  let min_buffer_size = 10

  init (
    actor: ActorNetwork,
    critic: CriticNetwork,
    stateSize: Int,
    actionSize: Int,
    learning_rate: Float = 0.0001,
    gamma: Float = 0.01) {
        self.actor_network = actor
        self.critic_network = critic
        self.target_critic_network = self.critic_network
        self.target_actor_network = self.actor_network
        self.gamma = gamma
        self.lr = learning_rate
        self.state_size = stateSize
        self.action_size = actionSize
        self.actor_optimizer = Adam(for: self.actor_network, learningRate: self.lr)
        self.critic_optimizer = Adam(for: self.critic_network, learningRate: self.lr)
        self.replayBuffer = ReplayBuffer(capacity: 1000, combined: false)

  }

  func remember(state: Tensor<Float>, action: Tensor<Float>, reward: Tensor<Float>, next_state: Tensor<Float>, dones: Tensor<Bool>) {
      self.replayBuffer.append(state:state, action:action, reward:reward, nextState:next_state, isDone:dones)
  }

  func get_action(state: Tensor<Float>) -> Tensor<Float> {
      return self.actor_network(state)
  }

  func train_actor_critic(batchSize: Int, iterationNum: Int) -> (Tensor<Float>, Tensor<Float>) {
      if self.replayBuffer.count >= self.min_buffer_size {
            let (states, actions, rewards, nextstates, dones) = self.replayBuffer.sample(batchSize: batchSize)
            //train critic
            let(critic_loss, critic_gradients) = valueWithGradient(at: critic_network) { critic_network -> Tensor<Float> in
              let npActionBatch = actions.makeNumpyArray()
              let npFullIndices = np.stack(
                [np.arange(batchSize, dtype: np.int32), npActionBatch], axis: 1)
              let tfFullIndices = Tensor<Int32>(numpy: npFullIndices)!
              let predicted_q_values = critic_network([states, actions])
              let predicted_q_values_batch = predicted_q_values.dimensionGathering(atIndices: tfFullIndices)

              let target_actions = self.target_actor_network(nextstates)

              let next_state_q_values = withoutDerivative(at: self.target_critic_network([nextstates, target_actions]))
              let target_q_values: Tensor<Float> = rewards + self.gamma * (1 - Tensor<Float>(dones)) * next_state_q_values
              return huberLoss(predicted: predicted_q_values_batch, expected: target_q_values, delta: 1)

            }
            self.critic_optimizer.update(&critic_network, along: critic_gradients)


            let(actor_loss, actor_gradients) = valueWithGradient(at: actor_network) { actor_network -> Tensor<Float> in
                let next_actions = actor_network(states)
                let actor_loss = 1 - self.critic_network([states, next_actions]).mean()
                return actor_loss
            }
            self.actor_optimizer.update(&actor_network, along: actor_gradients)
            return (actor_loss, critic_loss)
          } else {
            return (Tensor<Float>(0.0), Tensor<Float>(0.0))
          }

      }


      func updateCriticTargetNetwork(tau: Float) {
         self.target_critic_network.state_layer.weight =
                  tau * Tensor<Float>(self.critic_network.state_layer.weight) + (1 - tau) * self.target_critic_network.state_layer.weight
         self.target_critic_network.state_layer.bias =
                  tau * Tensor<Float>(self.critic_network.state_layer.bias) + (1 - tau) * self.target_critic_network.state_layer.bias
      }
 }
