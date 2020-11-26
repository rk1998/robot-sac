///Implementation of the DDPG (Deep Deterministic Policy Gradient) Algorithm with
///Hindsight Experience Replay
/// This implementation uses swift for Tensorflow and borrows ideas from
/// the swift-models repository: https://github.com/tensorflow/swift-models/
/// Original Paper:  https://arxiv.org/pdf/1707.01495.pdf
/// Swift for TensorFlow: https://github.com/tensorflow/swift
/// Author: Rohith Krishnan
import Foundation
import PythonKit
PythonLibrary.useVersion(3, 6)
import TensorFlow
let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
let gym = Python.import("gym")


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


//Taken from the DQN example in the tensorflow swift-models repo
//: https://github.com/tensorflow/swift-models/blob/master/Gym/DQN/main.swift
class MultiGoalEnvironmentWrapper {
  let originalEnv: PythonObject
  public let state_size: Int
  public let action_size: Int
  public let max_action_val: Float

  init(_ env: PythonObject) {
    self.originalEnv = env
    self.state_size = Int(env.observation_space.shape[0])!
    self.action_size = Int(env.action_space.shape[0])!
    self.max_action_val = Int(env.action_space.high[0])!
  }

  func reset() -> (state: Tensor<Float>, achieved_goal: Tensor<Float>, desired_goal: Tensor<Float> {
    let state = self.originalEnv.reset()
    let observation = state["observation"]
    let achieved_goal = state["achieved_goal"]
    let desired_goal = state["desired_goal"]
    return (Tensor<Float>(numpy: np.array(observation, dtype: np.float32))!, Tensor<Float>(numpy: np.array(achieved_goal, dtype: np.float32))!, Tensor<Float>(numpy: np.array(desired_goal, dtype: np.float32))!
  }

  func step(_ action: Tensor<Float>) -> (
    state: Tensor<Float>, achieved_goal: Tensor<Float>, desired_goal: Tensor<Float>, reward: Tensor<Float>, isDone: Tensor<Bool>, info: PythonObject
  ) {
    let (state, reward, isDone, info) = originalEnv.step(action.makeNumpyArray()).tuple4
    let observation = state["observation"]
    let achieved_goal = state["achieved_goal"]
    let desired_goal = state["desired_goal"]
    let tfState = Tensor<Float>(numpy: np.array(observation, dtype: np.float32))!
    let tfgoal = Tensor<Float>(numpy: np.array(achieved_goal, dtype: np.float32))!
    let tfgoal_desired = Tensor<Float>(numpy: np.array(desired_goal, dtype: np.float32))!
    let tfReward = Tensor<Float>(numpy: np.array(reward, dtype: np.float32))!
    let tfIsDone = Tensor<Bool>(numpy: np.array(isDone, dtype: np.bool))!
    return (tfState, tfgoal, tfgoal_desired, tfReward, tfIsDone, info)
  }

  func set_environment_seed(seed: Int) {
    self.originalEnv.seed(seed)
  }

  func compute_reward(achieved_goal: Tensor<Float>, goal: Tensor<Float>) -> Tensor<Float> {
    let achieved_goal_np = achieved_goal.makeNumpyArray()
    let goal_np = goal.makeNumpyArray()
    let reward = self.originalEnv.compute_reward(achieved_goal, goal, "sparse")
    let tfReward = Tensor<Float>(numpy: np.array(reward, dtype: np.float32))!
    return tfReward
  }

  func action_sample() -> Tensor<Float> {
    let action = originalEnv.action_space.sample()
    let tfAction = Tensor<Float>(numpy: np.array(action, dtype: np.float32))!
    return tfAction
  }
}


class HERSampler {

  let replay_length: Int
  let future_p: Float

  init(replay_length: Int) {
    self.replay_length = replay_length
    self.future_p = 1.0 - (1.0 /Float(1 + self.replay_length))
  }

  func sample_transitions(episode_batch: [Tensor<Float>] batch_size:Int) -> (
    stateBatch: Tensor<Float>,
    actionBatch: Tensor<Float>,
    rewardBatch: Tensor<Float>,
    nextStateBatch: Tensor<Float>,
    isDoneBatch: Tensor<Bool>
    ) {
  }


}



class HERReplayBuffer {
  let buffer_size: Int

  let max_timesteps: Int

  let max_size: Int

  let sample_size: Int

  var current_size: Int

  var num_transitions: Int

  let sampler : HERSampler



  @noDerivative var buffers : [String : PythonObject]

  init(max_timesteps: Int, max_buffer_size: Int, state_size: Int, goal_size: Int, action_size: Int) {
    self.max_timesteps = max_timesteps
    self.max_size = max_buffer_size
    self.current_size = 0
    self.num_transitions = 0
    self.sample_size = Int(max_buffer_size/max_timesteps)
    self.sampler = HERSampler(replay_length: 4)
    self.buffers = ["state" : [Tensor<Float>],
                    "achieved_goal" : [Tensor<Float>],
                    "desired_goal" : [Tensor<Float>],
                    "actions" : [Tensor<Float>]]
  }

  func remember(episode_batch: [[Tensor<Float>]]) {
    let states = episode_batch[0]
    let achieved_goals = episode_batch[1]
    let desired_goals = episode_batch[2]
    let actions = episode_batch[3]
    let count : Int = self.buffer["state"].count
    if count >= self.max_size {
      // Erase oldest SARS if the replay buffer is full
      self.buffer["state"].removeFirst()
      self.buffer["achieved_goal"].removeFirst()
      self.buffer["desired_goal"].removeFirst()
      self.buffer["actions"].removeFirst()
    }
    for i in 0..<episode_batch[0].count {
      self.buffer["state"].append(states[i])
      self.buffer["achieved_goal"].append(achieved_goals[i])
      self.buffer["desired_goal"].append(desired_goals[i])
      self.buffer["actions"].append(actions[i])
    }


  }

  func sample_batch(batch_size: Int) -> (
    stateBatch: Tensor<Float>,
    goalBatch: Tensor<Float>,
    actionBatch: Tensor<Float>,
    rewardBatch: Tensor<Float>,
    nextStateBatch: Tensor<Float>,
    isDoneBatch: Tensor<Bool>
  ) {

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

  @noDerivative var achieved_goals: [Tensor<Float>] = []

  @noDerivative var desired_goals: [Tensor<Float>] = []
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
    achieved_goal: Tensor<Float>,
    desired_goal: Tensor<Float>,
    action: Tensor<Float>,
    reward: Tensor<Float>,
    nextState: Tensor<Float>,
    isDone: Tensor<Bool>
  ) {
    if count >= capacity {
      // Erase oldest SARS if the replay buffer is full
      states.removeFirst()
      achieved_goals.removeFirst()
      desired_goals.removeFirst()
      actions.removeFirst()
      rewards.removeFirst()
      nextStates.removeFirst()
      isDones.removeFirst()
    }
    states.append(state)
    achieved_goals.append(achieved_goal)
    desired_goals.append(desired_goal)
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



struct CriticNetwork: Layer {
  typealias Input = [Tensor<Float>]
  typealias Output = Tensor<Float>


  public var layer_1, layer_2, layer_3 : Dense<Float>

  init(state_size: Int, action_size:Int, hiddenLayerSizes: [Int] = [400, 300], outDimension: Int) {
    self.layer_1 = Dense<Float>(inputSize: state_size + action_size, outputSize: hiddenLayerSizes[0], activation:relu)
    self.layer_2 = Dense<Float>(inputSize: hiddenLayerSizes[0], outputSize: hiddenLayerSizes[1], activation:relu)
    self.layer_3 = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: outDimension, activation: identity)

  }

  @differentiable
  func callAsFunction(_ input: Input) -> Output {
    let state: Tensor<Float> = input[0]
    let action: Tensor<Float> = input[1]
    let state_and_action = Tensor(concatenating: [state, action], alongAxis: 1)
    let h1 = layer_1(state_and_action)
    let h2 = layer_2(h1)
    let q_value = layer_3(h2)
    return q_value
  }

}


//Struct for the ActorNetwork
//This network directly maps environment states to the best action to take
// in that state
struct ActorNetwork: Layer {
  typealias Input = Tensor<Float>
  typealias Output = Tensor<Float>
  public var layer_1, layer_2, layer_3: Dense<Float>
  @noDerivative let max_action: Tensor<Float>
  init(observationSize: Int, actionSize: Int, hiddenLayerSizes: [Int] = [400, 300], maximum_action: Tensor<Float>) {
    layer_1 = Dense<Float>(inputSize: observationSize, outputSize: hiddenLayerSizes[0], activation: relu)
    layer_2 = Dense<Float>(inputSize: hiddenLayerSizes[0], outputSize: hiddenLayerSizes[1], activation: relu)
    layer_3 = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: actionSize, activation: tanh)
    self.max_action = maximum_action
  }

  @differentiable
  func callAsFunction(_ state: Input) -> Output {
    let layer_1_result = layer_1(state)
    let layer_2_result = layer_2(layer_1_result)
    let output_action = layer_3(layer_2_result)
    return self.max_action * output_action
  }

}

//Ornstein Uhlenbeck Noise - Gives Temporally correlated noise that provides better exploration of a physical space
class OUNoise {

  public var theta: Tensor<Float>

  public var mu: Tensor<Float>

  public var sigma: Tensor<Float>

  public var dt: Tensor<Float>

  public var x_init: Tensor<Float>

  public var x_prev: Tensor<Float>

  init(mu: Tensor<Float>, sigma: Tensor<Float>, x_init:Tensor<Float>, theta: Float = 0.25, dt: Float = 0.001) {
    self.mu = mu
    self.sigma = sigma
    self.x_init = x_init
    self.theta = Tensor<Float>(theta)
    self.dt = Tensor<Float>(dt)
    self.x_prev = self.x_init
    self.reset()
  }

  func getNoise() -> Tensor<Float> {
    let temp:Tensor<Float> = self.x_prev + self.theta*(self.mu - self.x_prev) * self.dt
    let x: Tensor<Float> = temp + self.sigma * sqrt(self.dt) * Tensor<Float>(randomNormal: TensorShape(self.mu.shape), mean:self.mu, standardDeviation: self.sigma)
    self.x_prev = x
    return x
  }

  func reset() {
    self.x_prev = self.x_init
  }


}



//Actor Critic Agent
class ActorCritic {


  public var actor_network: ActorNetwork

  public var critic_network: CriticNetwork

  public var target_critic_network: CriticNetwork

  public var target_actor_network: ActorNetwork

  public var replayBuffer: ReplayBuffer

  //let action_noise: GaussianNoise<Float>
  let action_noise: OUNoise

  let gamma: Float

  let state_size: Int

  let action_size: Int

  let actor_optimizer: Adam<ActorNetwork>

  let critic_optimizer: Adam<CriticNetwork>

  init (
    actor: ActorNetwork,
    actor_target: ActorNetwork,
    critic: CriticNetwork,
    critic_target: CriticNetwork,
    stateSize: Int,
    actionSize: Int,
    critic_lr: Float = 0.00009,
    actor_lr: Float = 0.0001,
    gamma: Float = 0.95) {
      self.actor_network = actor
      self.critic_network = critic
      self.target_critic_network = critic_target
      self.target_actor_network = actor_target
      self.gamma = gamma
      let mu : Tensor<Float> = Tensor<Float>(0.0)
      let sigma : Tensor<Float> = Tensor<Float>(0.3)
      let x_init: Tensor<Float> = Tensor<Float>(0.00)
      self.action_noise = OUNoise(mu: mu, sigma: sigma, x_init: x_init, theta: 0.15, dt: 0.05)
      //self.action_noise = GaussianNoise(standardDeviation: 0.20)
      self.state_size = stateSize
      self.action_size = actionSize
      self.actor_optimizer = Adam(for: self.actor_network, learningRate: actor_lr)
      self.critic_optimizer = Adam(for: self.critic_network, learningRate: critic_lr)
      self.replayBuffer = ReplayBuffer(capacity: 1500, combined: true)
  }

  func remember(state: Tensor<Float>, action: Tensor<Float>, reward: Tensor<Float>, next_state: Tensor<Float>, dones: Tensor<Bool>) {
    self.replayBuffer.append(state:state, action:action, reward:reward, nextState:next_state, isDone:dones)
  }

  func get_action(state: Tensor<Float>, env: TensorFlowEnvironmentWrapper, training: Bool) -> Tensor<Float> {

    let tfState = Tensor<Float>(numpy: np.expand_dims(state.makeNumpyArray(), axis: 0))!
    let net_action: Tensor<Float> = self.actor_network(tfState)
    if training {
      let noise = self.action_noise.getNoise()
      let noisy_action = net_action + noise
      //let noisy_action = self.action_noise(net_action)
      let action = noisy_action.clipped(min:-2.0, max:2.0)
      return action[0]
    } else {
      return net_action[0]
    }

  }

  func train_actor_critic(batchSize: Int, iterationNum: Int) -> (Float, Float) {

    let (states, actions, rewards, nextstates, dones) = self.replayBuffer.sample(batchSize: batchSize)
    //train critic
    let(critic_loss, critic_gradients) = valueWithGradient(at: self.critic_network) { critic_network -> Tensor<Float> in
      //get target q values from target critic network
      let next_state_q_values: Tensor<Float> = self.target_critic_network([nextstates, self.target_actor_network(nextstates)]).flattened()
      let target_q_values: Tensor<Float> =  rewards + self.gamma * (1 - Tensor<Float>(dones)) * next_state_q_values
      //get predicted q values from critic network
      let target_q_values_no_deriv : Tensor<Float> = withoutDerivative(at: target_q_values)
      let predicted_q_values: Tensor<Float> = critic_network([states, actions]).flattened()
      //let td_error: Tensor<Float> = target_q_values_no_deriv - predicted_q_values
      // let td_error: Tensor<Float> = squaredDifference(target_q_values_no_deriv, predicted_q_values)
      // let td_loss: Tensor<Float> = td_error.mean()
      //let td_loss: Tensor<Float> = 0.5*pow(td_error, 2).mean()
      //return td_loss
      return huberLoss(predicted: predicted_q_values, expected: target_q_values_no_deriv, delta: 5.0).mean()
    }
    self.critic_optimizer.update(&self.critic_network, along: critic_gradients)
    //train actor
    let(actor_loss, actor_gradients) = valueWithGradient(at: self.actor_network) { actor_network -> Tensor<Float> in
        //let next_actions = actor_network(states)
        let critic_q_values: Tensor<Float> = self.critic_network([states, actor_network(states)]).flattened()
        let loss: Tensor<Float> = Tensor<Float>(-1.0) * critic_q_values.mean()
        return loss
    }
    self.actor_optimizer.update(&self.actor_network, along: actor_gradients)
    return (actor_loss.scalarized(), critic_loss.scalarized())
  }

  func updateCriticTargetNetwork(tau: Float) {
    //update layer 1
    self.target_critic_network.layer_1.weight =
              tau * Tensor<Float>(self.critic_network.layer_1.weight) + (1 - tau) * self.target_critic_network.layer_1.weight
    self.target_critic_network.layer_1.bias =
              tau * Tensor<Float>(self.critic_network.layer_1.bias) + (1 - tau) * self.target_critic_network.layer_1.bias
    //update layer 2
    self.target_critic_network.layer_2.weight =
              tau * Tensor<Float>(self.critic_network.layer_2.weight) + (1 - tau) * self.target_critic_network.layer_2.weight
    self.target_critic_network.layer_2.bias =
              tau * Tensor<Float>(self.critic_network.layer_2.bias) + (1 - tau) * self.target_critic_network.layer_2.bias
    //update layer 3
    self.target_critic_network.layer_3.weight =
              tau * Tensor<Float>(self.critic_network.layer_3.weight) + (1 - tau) * self.target_critic_network.layer_3.weight
    self.target_critic_network.layer_3.bias =
              tau * Tensor<Float>(self.critic_network.layer_3.bias) + (1 - tau) * self.target_critic_network.layer_3.bias

  }

  func updateActorTargetNetwork(tau: Float) {
    //update layer 1
    self.target_actor_network.layer_1.weight =
              tau * Tensor<Float>(self.actor_network.layer_1.weight) + (1 - tau) * self.target_actor_network.layer_1.weight
    self.target_actor_network.layer_1.bias =
              tau * Tensor<Float>(self.actor_network.layer_1.bias) + (1 - tau) * self.target_actor_network.layer_1.bias
    //update layer 2
    self.target_actor_network.layer_2.weight =
              tau * Tensor<Float>(self.actor_network.layer_2.weight) + (1 - tau) * self.target_actor_network.layer_2.weight
    self.target_actor_network.layer_2.bias =
              tau * Tensor<Float>(self.actor_network.layer_2.bias) + (1 - tau) * self.target_actor_network.layer_2.bias
    //update layer 3
    self.target_actor_network.layer_3.weight =
              tau * Tensor<Float>(self.actor_network.layer_3.weight) + (1 - tau) * self.target_actor_network.layer_3.weight
    self.target_actor_network.layer_3.bias =
              tau * Tensor<Float>(self.actor_network.layer_3.bias) + (1 - tau) * self.target_actor_network.layer_3.bias
  }

 }