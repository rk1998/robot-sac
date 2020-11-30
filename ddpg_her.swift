///Implementation of the DDPG (Deep Deterministic Policy Gradient) Algorithm with
///Hindsight Experience Replay
/// This implementation uses swift for Tensorflow and borrows ideas from
/// the swift-models repository: https://github.com/tensorflow/swift-models/
/// This code is also based off of an implementation of the algorithm in python : https://github.com/TianhongDai/hindsight-experience-replay
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


//Environment wrapper for a multi goal environment from Open Ai Gym
//Based on env wrapper from the DQN example in the tensorflow swift-models repo
//: https://github.com/tensorflow/swift-models/blob/master/Gym/DQN/main.swift
class MultiGoalEnvironmentWrapper {
  let originalEnv: PythonObject
  public let state_size: Int
  public let action_size: Int
  public let max_action_val: Float
  public let goal_size: Int
  public let max_timesteps: Int

  init(_ env: PythonObject) {
    self.originalEnv = env
    let state = self.originalEnv.reset()
    let observation = state["observation"]
    let desired_goal = state["desired_goal"]
    self.state_size = Int(observation.shape[0])!
    self.action_size = Int(env.action_space.shape[0])!
    self.max_action_val = Float(env.action_space.high[0])!
    self.goal_size = Int(desired_goal.shape[0])!
    self.max_timesteps = Int(env._max_episode_steps)!
  }

  func reset() -> (state: Tensor<Float>, achieved_goal: Tensor<Float>, desired_goal: Tensor<Float>) {
    let state = self.originalEnv.reset()
    let observation = state["observation"]
    let achieved_goal = state["achieved_goal"]
    let desired_goal = state["desired_goal"]
    return (Tensor<Float>(numpy: np.array(observation, dtype: np.float32))!, Tensor<Float>(numpy: np.array(achieved_goal, dtype: np.float32))!, Tensor<Float>(numpy: np.array(desired_goal, dtype: np.float32))!)
  }

  func step(_ action: Tensor<Float>) -> (
    state: Tensor<Float>, achieved_goal: Tensor<Float>, desired_goal: Tensor<Float>, reward: Tensor<Float>, isDone: Tensor<Bool>, info: PythonObject) {
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
    let reward = self.originalEnv.compute_reward(achieved_goal_np, goal_np, "sparse")
    let tfReward = Tensor<Float>(numpy: np.array(reward, dtype: np.float32))!
    return tfReward
  }

  func compute_reward_np(achieved_goal: PythonObject, goal: PythonObject) -> PythonObject {
    let reward = self.originalEnv.compute_reward(achieved_goal, goal, "sparse")
    return reward
  }

  func action_sample() -> Tensor<Float> {
    let action = originalEnv.action_space.sample()
    let tfAction = Tensor<Float>(numpy: np.array(action, dtype: np.float32))!
    return tfAction
  }
}


//Sampling Method for Hindsight Experience Replay
//Based on Python implementation found here:
//https://github.com/TianhongDai/hindsight-experience-replay
class HERSampler {

  let replay_length: Int
  let future_p: Float

  init(replay_length: Int) {
    self.replay_length = replay_length
    self.future_p = 1.0 - (1.0/Float(1 + self.replay_length))
  }

  func sample_transitions(episode_batch: [String : Tensor<Float>], batch_size:Int, env: MultiGoalEnvironmentWrapper) -> [String: Tensor<Float>] {
    let timesteps = PythonObject(episode_batch["actions"]!.shape[1])
    let roll_out_batch_size = PythonObject(episode_batch["actions"]!.shape[0])
    let episode_idxs = np.random.randint(0, roll_out_batch_size, size: batch_size)
    let t_samples = np.random.randint(timesteps, size: batch_size)
    var transitions: [String : PythonObject] = [:]
    for (key, batch_list) in episode_batch {
      let np_list = batch_list.makeNumpyArray()
      let transition_sample = np_list[episode_idxs, t_samples].copy()
      transitions[key] = transition_sample
    }
    let batch_uniform  = np.random.uniform(size: PythonObject(batch_size))
    let batch_uniform_filter: PythonObject = batch_uniform < self.future_p
    let her_indexes = np.where(batch_uniform_filter)
    // var her_indexes_sf: [Int] = []
    // for i in 0..<batch_size{
    //   let val = batch_uniform[i]
    //   if Float(val)! < self.future_p {
    //     her_indexes_sf.append(i)
    //   }
    // }
    // let her_indexes = np.array(her_indexes_sf)
    let future_offset = (np.random.uniform(size: batch_size) * (timesteps - t_samples)).astype(np.int32)
    let future_t = (t_samples + 1 + future_offset)[her_indexes]
    let ag_numpy = episode_batch["achieved_goal"]!.makeNumpyArray()
    let future_ag = ag_numpy[episode_idxs[her_indexes], future_t]
    transitions["desired_goal"]![her_indexes] = future_ag
    transitions["reward"] = np.expand_dims(env.compute_reward_np(achieved_goal: transitions["achieved_goal_next"]!, goal: transitions["desired_goal"]!), 1)
    var transitions_tf : [String: Tensor<Float>] = [:]
    for (key, batch_list) in transitions {
      let reshaped_list = batch_list.reshape(batch_size, batch_list.shape[1])
      transitions_tf[key] = Tensor<Float>(numpy:reshaped_list)!
    }
    return transitions_tf

  }


}


//Replay Buffer for Hindsight Experience Replay
//Based off of python implementation found here:
//https://github.com/TianhongDai/hindsight-experience-replay
class HERReplayBuffer {

  let max_timesteps: Int

  let max_size: Int

  let sample_size: Int

  var current_size: Int

  var num_transitions: Int

  let sampler: HERSampler

  let state_size: Int

  let goal_size: Int

  let action_size: Int



  @noDerivative var buffers : [String : [Tensor<Float>]] = [:]

  init(max_timesteps: Int, max_buffer_size: Int, state_size: Int, goal_size: Int, action_size: Int) {
    self.max_timesteps = max_timesteps
    self.max_size = max_buffer_size
    self.current_size = 0
    self.num_transitions = 0
    self.state_size = state_size
    self.action_size = action_size
    self.goal_size = goal_size
    self.sample_size = Int(max_buffer_size/max_timesteps)
    self.sampler = HERSampler(replay_length: 4)

    self.buffers = ["state" : [],
                    "achieved_goal" : [],
                    "desired_goal" : [],
                    "actions" : []]
  }

  func remember(episode_batch: [[Tensor<Float>]]) {
    let states = episode_batch[0]
    let achieved_goals = episode_batch[1]
    let desired_goals = episode_batch[2]
    let actions = episode_batch[3]
    let count : Int = buffers["state"]!.count
    if count >= self.max_size {
      // Erase oldest SARS if the replay buffer is full
      buffers["state"]!.removeFirst()
      buffers["achieved_goal"]!.removeFirst()
      buffers["desired_goal"]!.removeFirst()
      buffers["actions"]!.removeFirst()
      current_size -= 1
    }
    for i in 0..<episode_batch[0].count {
      buffers["state"]!.append(states[i])
      buffers["achieved_goal"]!.append(achieved_goals[i])
      buffers["desired_goal"]!.append(desired_goals[i])
      buffers["actions"]!.append(actions[i])
      current_size += 1
    }


  }

  func sample_batch(batch_size: Int, env: MultiGoalEnvironmentWrapper) -> [String : Tensor<Float>] {
    var temp_buffer: [String : Tensor<Float>] = [:]
    for (batch_type, batch) in buffers {
      temp_buffer[batch_type] = Tensor<Float>(batch)
    }
    let state_count : Int = buffers["state"]!.count
    temp_buffer["state_next"] = temp_buffer["state"]![0..<state_count, 1..<self.max_timesteps + 1, 0..<self.state_size]
    temp_buffer["achieved_goal_next"] = temp_buffer["state"]![0..<state_count, 1..<self.max_timesteps + 1, 0..<self.goal_size]
    let batch = self.sampler.sample_transitions(episode_batch: temp_buffer, batch_size: batch_size, env: env)
    return batch

  }



}


struct CriticNetwork: Layer {
  typealias Input = [Tensor<Float>]
  typealias Output = Tensor<Float>


  public var layer_1, layer_2, layer_3, layer_4 : Dense<Float>

  init(state_size: Int, action_size:Int, hiddenLayerSizes: [Int] = [400, 300, 200], outDimension: Int) {
    self.layer_1 = Dense<Float>(inputSize: state_size + action_size, outputSize: hiddenLayerSizes[0], activation:relu)
    self.layer_2 = Dense<Float>(inputSize: hiddenLayerSizes[0], outputSize: hiddenLayerSizes[1], activation:relu)
    self.layer_3 = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: hiddenLayerSizes[2], activation: relu)
    self.layer_4 = Dense<Float>(inputSize: hiddenLayerSizes[2], outputSize: outDimension, activation: identity)
  }

  @differentiable
  func callAsFunction(_ input: Input) -> Output {
    let state: Tensor<Float> = input[0]
    let action: Tensor<Float> = input[1]
    let state_and_action = Tensor(concatenating: [state, action], alongAxis: 1)
    let h1 = layer_1(state_and_action)
    let h2 = layer_2(h1)
    let h3 = layer_3(h2)
    let q_value = layer_4(h3)
    return q_value
  }

}


//Struct for the ActorNetwork
//This network directly maps environment states to the best action to take
// in that state
struct ActorNetwork: Layer {
  typealias Input = Tensor<Float>
  typealias Output = Tensor<Float>
  public var layer_1, layer_2, layer_3, layer_4: Dense<Float>
  @noDerivative let max_action: Tensor<Float>
  init(observationSize: Int, actionSize: Int, hiddenLayerSizes: [Int] = [400, 300, 200], maximum_action: Tensor<Float>) {
    layer_1 = Dense<Float>(inputSize: observationSize, outputSize: hiddenLayerSizes[0], activation: relu)
    layer_2 = Dense<Float>(inputSize: hiddenLayerSizes[0], outputSize: hiddenLayerSizes[1], activation: relu)
    layer_3 = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: hiddenLayerSizes[2], activation: relu)
    layer_4 = Dense<Float>(inputSize: hiddenLayerSizes[2], outputSize: actionSize, activation: tanh)
    self.max_action = maximum_action
  }

  @differentiable
  func callAsFunction(_ state: Input) -> Output {
    let layer_1_result = layer_1(state)
    let layer_2_result = layer_2(layer_1_result)
    let layer_3_result = layer_3(layer_2_result)
    let output_action = layer_4(layer_3_result)
    return self.max_action * output_action
  }

}


class Normalizer {

  public let eps: Float
  public var total_sum: Tensor<Float>
  public var total_sumsq: Tensor<Float>
  public var total_count: Tensor<Float>
  public var mean: Tensor<Float>
  public var std: Tensor<Float>

  init(size: Int, epsilon: Float = 0.01) {
    self.eps = epsilon
    self.total_sum = Tensor<Float>(zeros: TensorShape(size))
    self.total_sumsq = Tensor<Float>(zeros: TensorShape(size))
    self.total_count = Tensor<Float>(1.0)
    self.mean = Tensor<Float>(zeros: TensorShape(size))
    self.std = Tensor<Float>(zeros: TensorShape(size))
  }

  func update(val: Tensor<Float>) {
    self.total_sum = self.total_sum + val.sum(alongAxes: 0)
    self.total_sumsq = self.total_sumsq + pow(val, 2).sum(alongAxes: 0)
    self.total_count = self.total_count + Tensor<Float>(Float(val.shape[0]))
  }

  func recompute_stats() {
    self.mean = self.total_sum/self.total_count
    let sum_sq: Tensor<Float> = (self.total_sumsq/self.total_count) - pow((self.total_sumsq/self.total_count), 2)
    self.std = sqrt(max(pow(Tensor<Float>(eps), 2), sum_sq))
  }

  func normalize(val: Tensor<Float>) -> Tensor<Float> {

    let result: Tensor<Float> = (val - self.mean)/(self.std + eps)
    return result
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

  public var replayBuffer: HERReplayBuffer


  public var state_normalizer: Normalizer

  public var goal_normalizer: Normalizer

  //let action_noise: GaussianNoise<Float>
  let action_noise: OUNoise

  let gamma: Float

  let state_size: Int

  let action_size: Int

  let goal_size: Int

  let actor_optimizer: Adam<ActorNetwork>

  let critic_optimizer: Adam<CriticNetwork>

  let max_action: Tensor<Float>

  init (
    actor: ActorNetwork,
    actor_target: ActorNetwork,
    critic: CriticNetwork,
    critic_target: CriticNetwork,
    replay_buff: HERReplayBuffer,
    stateSize: Int,
    actionSize: Int,
    goalSize: Int,
    maxAction: Tensor<Float>,
    critic_lr: Float = 0.0009,
    actor_lr: Float = 0.0009,
    gamma: Float = 0.99) {
      self.actor_network = actor
      self.critic_network = critic
      self.target_critic_network = critic_target
      self.target_actor_network = actor_target
      self.gamma = gamma
      let mu : Tensor<Float> = Tensor<Float>(0.0)
      let sigma : Tensor<Float> = Tensor<Float>(0.20)
      let x_init: Tensor<Float> = Tensor<Float>(0.00)
      self.action_noise = OUNoise(mu: mu, sigma: sigma, x_init: x_init, theta: 0.15, dt: 0.05)
      //self.action_noise = GaussianNoise(standardDeviation: 0.20)
      self.state_size = stateSize
      self.action_size = actionSize
      self.goal_size = goalSize

      self.state_normalizer = Normalizer(size: stateSize)
      self.goal_normalizer = Normalizer(size: goalSize)

      self.max_action = maxAction
      self.actor_optimizer = Adam(for: self.actor_network, learningRate: actor_lr)
      self.critic_optimizer = Adam(for: self.critic_network, learningRate: critic_lr)
      self.replayBuffer = replay_buff
  }

  func remember(episode_batch: [[Tensor<Float>]]) {
    self.replayBuffer.remember(episode_batch: episode_batch)
  }

  func update_normalizers(batchSize: Int, env: MultiGoalEnvironmentWrapper) {
    // var buffer: [String: [Tensor<Float>]] = ["state" : [],
    //                                             "achieved_goal" : [],
    //                                             "desired_goal" : [],
    //                                             "actions" : []]
    // let states = episode_batch[0]
    // let achieved_goals = episode_batch[1]
    // let desired_goals = episode_batch[2]
    // let actions = episode_batch[3]
    // for i in 0..<episode_batch[0].count {
    //   buffer["state"]!.append(states[i])
    //   buffer["achieved_goal"]!.append(achieved_goals[i])
    //   buffer["desired_goal"]!.append(desired_goals[i])
    //   buffer["actions"]!.append(actions[i])
    // }

    // var temp_buffer: [String : Tensor<Float>] = [:]
    // for (batch_type, batch) in buffer {
    //   temp_buffer[batch_type] = Tensor<Float>(batch)
    // }
    // let state_count : Int = buffer["state"]!.count
    // temp_buffer["state_next"] = temp_buffer["state"]![0..<state_count, 1..<50 + 1, 0..<self.state_size]
    // temp_buffer["achieved_goal_next"] = temp_buffer["state"]![0..<state_count, 1..<50 + 1, 0..<self.goal_size]

    let batch = self.replayBuffer.sample_batch(batch_size: batchSize, env: env)
    let states_sampled: Tensor<Float> = batch["state"]!
    let goals_sampled: Tensor<Float> = batch["desired_goal"]!
    self.state_normalizer.update(val: states_sampled)
    self.goal_normalizer.update(val: goals_sampled)

    self.state_normalizer.recompute_stats()
    self.goal_normalizer.recompute_stats()

  }

  func get_action(state: Tensor<Float>, goal: Tensor<Float>, env: MultiGoalEnvironmentWrapper, training: Bool) -> Tensor<Float> {
    let tfState = Tensor<Float>(numpy: np.expand_dims(state.makeNumpyArray(), axis: 0))!
    let tfGoal = Tensor<Float>(numpy: np.expand_dims(goal.makeNumpyArray(), axis: 0))!
    //normalize the state and the desired goal
    //let normed_state : Tensor<Float> = ((tfState - tfState.mean())/sqrt(tfState.standardDeviation() + 0.0001)).clipped(min: -5.0, max: 5.0)
    //let normed_goal : Tensor<Float> = ((tfGoal - tfGoal.mean())/sqrt(tfGoal.standardDeviation() + 0.0001)).clipped(min: -5.0, max: 5.0)
    let normed_state : Tensor<Float> = self.state_normalizer.normalize(val: tfState).clipped(min: -5.0, max: 5.0)
    let normed_goal : Tensor<Float> = self.goal_normalizer.normalize(val: tfGoal).clipped(min: -5.0, max: 5.0)



    let state_input : Tensor<Float> = Tensor<Float>(concatenating: [normed_state, normed_goal], alongAxis : 1)
    let net_action: Tensor<Float> = self.actor_network(state_input)
    if training {
      let noise = self.action_noise.getNoise()
      let noisy_action = net_action + noise
      //let noisy_action = self.action_noise(net_action)
      let action = noisy_action.clipped(min:-self.max_action, max:self.max_action)
      return action[0]
    } else {
      return net_action[0]
    }

  }

  func train_actor_critic(batchSize: Int, env: MultiGoalEnvironmentWrapper) -> (Float, Float) {

    let episode_batch: [String: Tensor<Float>] = self.replayBuffer.sample_batch(batch_size: batchSize, env: env)

    let states: Tensor<Float> = episode_batch["state"]!
    let nextstates: Tensor<Float> = episode_batch["state_next"]!
    let desired_goal: Tensor<Float> = episode_batch["desired_goal"]!
    let actions: Tensor<Float> = episode_batch["actions"]!
    let rewards: Tensor<Float> = episode_batch["reward"]!

    //normalize states and the desired goals
    // let norm_states: Tensor<Float> = ((states - states.mean())/sqrt(states.standardDeviation() + 0.00001)).clipped(min: -5.0, max: 5.0)
    // let norm_nextstates: Tensor<Float> = ((nextstates - nextstates.mean())/sqrt(nextstates.standardDeviation() + 0.00001)).clipped(min: -5.0, max: 5.0)
    // let norm_desired_goal: Tensor<Float> = ((desired_goal - desired_goal.mean())/sqrt(desired_goal.standardDeviation() + 0.00001)).clipped(min: -5.0, max: 5.0)

    let norm_states: Tensor<Float> = self.state_normalizer.normalize(val: states).clipped(min: -5.0, max: 5.0)
    let norm_nextstates: Tensor<Float> = self.state_normalizer.normalize(val: nextstates).clipped(min: -5.0, max: 5.0)
    let norm_desired_goal: Tensor<Float> = self.goal_normalizer.normalize(val:desired_goal).clipped(min: -5.0, max: 5.0)

    //concatenate states and goals for the network inputs
    let inputs: Tensor<Float> = Tensor<Float>(concatenating: [norm_states, norm_desired_goal], alongAxis: 1)
    let inputs_next: Tensor<Float> = Tensor<Float>(concatenating: [norm_nextstates, norm_desired_goal], alongAxis:1)


    //train critic
    let(critic_loss, critic_gradients) = valueWithGradient(at: self.critic_network) { critic_network -> Tensor<Float> in
      //get target q values from target critic network
      let next_state_q_values: Tensor<Float> = self.target_critic_network([inputs_next, self.target_actor_network(inputs_next)]).flattened()
      let target_q_values: Tensor<Float> =  rewards + self.gamma * next_state_q_values
      //get predicted q values from critic network
      let target_q_values_no_deriv : Tensor<Float> = withoutDerivative(at: target_q_values)
      let predicted_q_values: Tensor<Float> = critic_network([inputs, actions]).flattened()
      let td_error: Tensor<Float> = target_q_values_no_deriv - predicted_q_values
      let td_loss: Tensor<Float> = 0.5*pow(td_error, 2).mean()
      return td_loss
      //return huberLoss(predicted: predicted_q_values, expected: target_q_values_no_deriv, delta: 5.0).mean()
    }
    self.critic_optimizer.update(&self.critic_network, along: critic_gradients)

    //train actor
    let(actor_loss, actor_gradients) = valueWithGradient(at: self.actor_network) { actor_network -> Tensor<Float> in
        //let next_actions = actor_network(states)
        let next_actions = actor_network(inputs)
        let critic_q_values: Tensor<Float> = self.critic_network([inputs, next_actions]).flattened()
        let loss: Tensor<Float> = Tensor<Float>(-1.0) * critic_q_values.mean()
        //let reg_loss: Tensor<Float> = 0.5*pow((next_actions/self.max_action), 2).mean()
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
    //update layer 4
    self.target_critic_network.layer_4.weight =
              tau * Tensor<Float>(self.critic_network.layer_4.weight) + (1 - tau) * self.target_critic_network.layer_4.weight
    self.target_critic_network.layer_4.bias =
              tau * Tensor<Float>(self.critic_network.layer_4.bias) + (1 - tau) * self.target_critic_network.layer_4.bias

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
    //update layer 4
    self.target_actor_network.layer_4.weight =
              tau * Tensor<Float>(self.actor_network.layer_4.weight) + (1 - tau) * self.target_actor_network.layer_4.weight
    self.target_actor_network.layer_4.bias =
              tau * Tensor<Float>(self.actor_network.layer_4.bias) + (1 - tau) * self.target_actor_network.layer_4.bias
  }

 }



func evaluate_agent(agent: ActorCritic, env: MultiGoalEnvironmentWrapper, num_rollouts: Int = 3, num_steps: Int = 50) -> Float {
  var total_success_rate: [Tensor<Float>] = []
  var totalReward: Float = 0.0
  for _ in 0..<num_rollouts {
    var rollout_success_rate: [Float] = []
    var (observation, _, desired_g) = env.reset()
    for _ in 0..<num_steps {
      let action = agent.get_action(state: observation, goal: desired_g, env: env, training: false)
      let (next_state, _, desired_g_next, _, _, info) = env.step(action)
      observation = next_state
      desired_g = desired_g_next
      rollout_success_rate.append(Float(info["is_success"])!)

    }
    let rollouts_tf : Tensor<Float> = Tensor<Float>(rollout_success_rate)
    total_success_rate.append(rollouts_tf)
  }
  let total_success_rate_tf : Tensor<Float> = Tensor<Float>(total_success_rate)
  //print(total_success_rate_tf)
  let success_rate_tf : Tensor<Float> = total_success_rate_tf[0..<total_success_rate_tf.shape[0], total_success_rate_tf.shape[1] - 1]
  print(success_rate_tf)
  totalReward = success_rate_tf.mean().scalarized()
  return totalReward

}

func ddpg_her(actor_critic: ActorCritic, env: MultiGoalEnvironmentWrapper,
          epochs: Int = 50, episodes: Int = 40, rollouts: Int = 2,
          stepsPerEpisode: Int = 300, update_steps: Int = 30, batchSize: Int = 32, tau: Float = 0.001,
          update_every: Int = 1) ->([Float], [Float], [Float])  {

    // var totalRewards: [Float] = []
    // var movingAverageReward: [Float] = []
    var total_actor_losses: [Float] = []
    var total_critic_losses: [Float] = []
    var success_rates: [Float] = []
    //var bestReward: Float = -99999999.0
    //var sample_random_action: Bool = true
    //var training: Bool = false
    //let sampling_episodes: Int = 100
    actor_critic.updateCriticTargetNetwork(tau: 1.0)
    actor_critic.updateActorTargetNetwork(tau: 1.0)
    print("Starting Training\n")
    for epoch in 0..<epochs {
      print("\nEPOCH: \(epoch)")
      if epoch == 0 || epoch == 25 || epoch == 50 || epoch == 75 {
        let filename: String = "results/fetch_push_ddpg_her_4_ep" + String(epoch) + ".npy"
        test_agent(agent: actor_critic, env: env, num_steps: 60, filename: filename)
      }
      var actor_losses: [Float] = []
      var critic_losses: [Float] = []
      for i in 0..<episodes {
        print("\nEPISODE: \(i)")
        var mb_obs : [Tensor<Float>] = []
        var mb_ag : [Tensor<Float>] = []
        var mb_g : [Tensor<Float>] = []
        var mb_actions : [Tensor<Float>] = []
        for _ in 0..<rollouts {
          var ep_obs : [Tensor<Float>] = []
          var ep_ag : [Tensor<Float>] = []
          var ep_g : [Tensor<Float>] = []
          var ep_actions : [Tensor<Float>] = []

          var (observation, achieved_g, desired_g) = env.reset()
          var t : Int = 0
          while t < stepsPerEpisode {
            let action: Tensor<Float> = actor_critic.get_action(state: observation, goal: desired_g, env: env, training: true)
            let (obs_next, ag_next, _, _, _, _) = env.step(action)
            ep_obs.append(observation)
            ep_ag.append(achieved_g)
            ep_g.append(desired_g)
            ep_actions.append(action)
            observation = obs_next
            achieved_g = ag_next
            t += 1
          }
          ep_obs.append(observation)
          ep_ag.append(achieved_g)
          let episode_obs_tf : Tensor<Float> = Tensor<Float>(stacking: ep_obs)
          let episode_ag_tf : Tensor<Float> = Tensor<Float>(stacking: ep_ag)
          let episode_g_tf : Tensor<Float> = Tensor<Float>(stacking: ep_g)
          let episode_actions_tf : Tensor<Float> = Tensor<Float>(stacking: ep_actions)
          mb_obs.append(episode_obs_tf)
          mb_ag.append(episode_ag_tf)
          mb_g.append(episode_g_tf)
          mb_actions.append(episode_actions_tf)
        }
        actor_critic.remember(episode_batch: [mb_obs, mb_ag, mb_g, mb_actions])
        actor_critic.update_normalizers(batchSize: batchSize, env: env)
        var actor_loss_total : Float = 0.0
        var critic_loss_total : Float = 0.0
        for _ in 0..<update_steps {
          let(a_loss, c_loss) = actor_critic.train_actor_critic(batchSize: batchSize, env: env)
          actor_loss_total += a_loss
          critic_loss_total += c_loss
        }
        actor_losses.append(actor_loss_total/Float(update_steps))
        critic_losses.append(critic_loss_total/Float(update_steps))
        if i % update_every == 0 {
          actor_critic.updateActorTargetNetwork(tau: tau)
          actor_critic.updateCriticTargetNetwork(tau: tau)
        }

      }
      let avg_actor_loss = Tensor<Float>(actor_losses).mean().scalarized()
      let avg_critic_loss = Tensor<Float>(critic_losses).mean().scalarized()
      total_actor_losses.append(avg_actor_loss)
      total_critic_losses.append(avg_critic_loss)
      print("Evaluating\n")
      let eval_success_rate = evaluate_agent(agent: actor_critic, env: env, num_rollouts: 5, num_steps:stepsPerEpisode)
      print(String(format:"EPOCH %4d | Eval Success Rate: %.03f | Avg Actor Loss %.03f | Avg Critic Loss %.03f", epoch, eval_success_rate, avg_actor_loss, avg_critic_loss))

      success_rates.append(eval_success_rate)

    }

    return (total_actor_losses, total_critic_losses, success_rates)

 }



func test_agent(agent: ActorCritic, env: MultiGoalEnvironmentWrapper, num_steps: Int = 300, filename: String = "results/frames.npy") {
  var frames: [PythonObject] = []
  var (observation, _, desired_g) = env.reset()
  var success: [Float] = []
  for _ in 0..<num_steps {
    let frame = env.originalEnv.render(mode: "rgb_array")
    frames.append(frame)
    let action = agent.get_action(state: observation, goal: desired_g, env: env, training: false)
    let (next_state, _, desired_g_next, tfreward, _, info) = env.step(action)
    let reward : Float = tfreward.scalarized()
    print("Reward: \(reward)\n")
    success.append(Float(info["is_success"])!)
    observation = next_state
    desired_g = desired_g_next
  }
  env.originalEnv.close()
  let frame_np_array = np.array(frames)
  np.save(filename, frame_np_array)
  print("\n Is success: \(success[num_steps - 1])")
}


let env = MultiGoalEnvironmentWrapper(gym.make("FetchPush-v1"))
print(env.state_size)
print(env.action_size)
print(env.max_timesteps)
env.set_environment_seed(seed: 1001)
let max_action: Tensor<Float> = Tensor<Float>(env.max_action_val)
print(max_action)
let actor_net: ActorNetwork = ActorNetwork(observationSize: env.state_size + env.goal_size, actionSize: env.action_size, hiddenLayerSizes: [400, 300, 200], maximum_action:max_action)
let actor_target: ActorNetwork = ActorNetwork(observationSize: env.state_size + env.goal_size, actionSize: env.action_size, hiddenLayerSizes: [400, 300, 200], maximum_action:max_action)
let critic_net: CriticNetwork = CriticNetwork(state_size: env.state_size + env.goal_size,  action_size: env.action_size, hiddenLayerSizes: [400, 300, 200], outDimension: 1)
let critic_target: CriticNetwork = CriticNetwork(state_size: env.state_size + env.goal_size, action_size: env.action_size, hiddenLayerSizes: [400, 300, 200], outDimension: 1)
let replay_buffer: HERReplayBuffer = HERReplayBuffer(max_timesteps: env.max_timesteps, max_buffer_size: 9000, state_size: env.state_size, goal_size: env.goal_size, action_size: env.action_size)


let actor_critic: ActorCritic = ActorCritic(actor: actor_net,
                                            actor_target: actor_target,
                                            critic: critic_net,
                                            critic_target: critic_target,
                                            replay_buff: replay_buffer,
                                            stateSize: env.state_size,
                                            actionSize: env.action_size,
                                            goalSize: env.goal_size,
                                            maxAction: max_action)
let(actor_losses, critic_losses, success_rates) = ddpg_her(actor_critic: actor_critic,
                                                           env: env,
                                                           epochs: 60,
                                                           episodes: 50,
                                                           rollouts: 4,
                                                           stepsPerEpisode: env.max_timesteps,
                                                           update_steps: 50,
                                                           batchSize: 128,
                                                           tau: 0.005,
                                                           update_every: 5)

plt.plot(success_rates)
plt.title("DDPG+HER Success Rate on FetchPush-v1")
plt.xlabel("Epoch")
plt.ylabel("Eval. Success Rate")
plt.savefig("results/rewards/fetch_push_success_ddpg_her_4.png")
plt.clf()
let success_rates_np = np.array(success_rates)
np.save("results/rewards/fetch_push_success_ddpg_her_4.npy", success_rates_np)


plt.plot(actor_losses)
plt.title("DDPG+HER Avg. Actor Loss on FetchPush-v1")
plt.xlabel("Epoch")
plt.ylabel("Avg. Loss")
plt.savefig("results/losses/fetch_push_actor_loss_ddpg_her_4.png")
plt.clf()

plt.plot(critic_losses)
plt.title("DDPG+HER Avg. Critic Loss on FetchPush-v1")
plt.xlabel("Epoch")
plt.ylabel("Avg. Loss")
plt.savefig("results/losses/fetch_push_critic_loss_ddpg_her_4.png")
plt.clf()

test_agent(agent: actor_critic, env: env, num_steps: 60, filename: "results/fetch_push_ddpg_her_4.npy")




