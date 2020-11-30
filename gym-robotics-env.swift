/// Script that runs DDPG and Soft Actor Critic on Fetch Robotics Tasks
/// from Open Ai
/// This implementation uses swift for Tensorflow and borrows ideas from
/// the swift-models repository: https://github.com/tensorflow/swift-models/
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

  let max_action: Tensor<Float>

  init (
    actor: ActorNetwork,
    actor_target: ActorNetwork,
    critic: CriticNetwork,
    critic_target: CriticNetwork,
    stateSize: Int,
    actionSize: Int,
    maxAction: Tensor<Float>,
    critic_lr: Float = 0.0003,
    actor_lr: Float = 0.0001,
    gamma: Float = 0.95) {
      self.actor_network = actor
      self.critic_network = critic
      self.target_critic_network = critic_target
      self.target_actor_network = actor_target
      self.gamma = gamma
      let mu : Tensor<Float> = Tensor<Float>(0.0)
      let sigma : Tensor<Float> = Tensor<Float>(0.2)
      let x_init: Tensor<Float> = Tensor<Float>(0.00)
      self.action_noise = OUNoise(mu: mu, sigma: sigma, x_init: x_init, theta: 0.15, dt: 0.05)
      //self.action_noise = GaussianNoise(standardDeviation: 0.20)
      self.state_size = stateSize
      self.action_size = actionSize
      self.max_action = maxAction
      self.actor_optimizer = Adam(for: self.actor_network, learningRate: actor_lr)
      self.critic_optimizer = Adam(for: self.critic_network, learningRate: critic_lr)
      self.replayBuffer = ReplayBuffer(capacity: 5000, combined: true)
  }

  func remember(state: Tensor<Float>, action: Tensor<Float>, reward: Tensor<Float>, next_state: Tensor<Float>, dones: Tensor<Bool>) {
    self.replayBuffer.append(state:state, action:action, reward:reward, nextState:next_state, isDone:dones)
  }

  func get_action(state: Tensor<Float>, env: MultiGoalEnvironmentWrapper, training: Bool) -> Tensor<Float> {

    let tfState = Tensor<Float>(numpy: np.expand_dims(state.makeNumpyArray(), axis: 0))!
    let normed_state : Tensor<Float> = ((tfState - tfState.mean())/sqrt(tfState.standardDeviation() + 0.0001)).clipped(min: -5.0, max: 5.0)
    let net_action: Tensor<Float> = self.actor_network(normed_state)
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

  func train_actor_critic(batchSize: Int) -> (Float, Float) {

    let (states, actions, rewards, nextstates, dones) = self.replayBuffer.sample(batchSize: batchSize)

    let norm_states: Tensor<Float> = ((states - states.mean(alongAxes: 0))/sqrt(states.standardDeviation(alongAxes: 0) + 0.00001)).clipped(min: -5.0, max: 5.0)
    let norm_nextstates: Tensor<Float> = ((nextstates - nextstates.mean(alongAxes: 0))/sqrt(nextstates.standardDeviation(alongAxes: 0) + 0.00001)).clipped(min: -5.0, max: 5.0)
    //train critic
    let(critic_loss, critic_gradients) = valueWithGradient(at: self.critic_network) { critic_network -> Tensor<Float> in
      //get target q values from target critic network
      let next_state_q_values: Tensor<Float> = self.target_critic_network([norm_nextstates, self.target_actor_network(norm_nextstates)]).flattened()
      let target_q_values: Tensor<Float> =  rewards + self.gamma * (1 - Tensor<Float>(dones)) * next_state_q_values
      //get predicted q values from critic network
      let target_q_values_no_deriv : Tensor<Float> = withoutDerivative(at: target_q_values)
      let predicted_q_values: Tensor<Float> = critic_network([norm_states, actions]).flattened()
      let td_error: Tensor<Float> = target_q_values_no_deriv - predicted_q_values
      // let td_error: Tensor<Float> = squaredDifference(target_q_values_no_deriv, predicted_q_values)
      // let td_loss: Tensor<Float> = td_error.mean()
      let td_loss: Tensor<Float> = 0.5*pow(td_error, 2).mean()
      return td_loss
      //return huberLoss(predicted: predicted_q_values, expected: target_q_values_no_deriv, delta: 5.0).mean()
    }
    self.critic_optimizer.update(&self.critic_network, along: critic_gradients)
    //train actor
    let(actor_loss, actor_gradients) = valueWithGradient(at: self.actor_network) { actor_network -> Tensor<Float> in
        let next_actions = actor_network(norm_states)
        let critic_q_values: Tensor<Float> = self.critic_network([norm_states, next_actions]).flattened()
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
    var (observation, _, _) = env.reset()
    for _ in 0..<num_steps {
      let action = agent.get_action(state: observation, env: env, training: false)
      let (next_state, _, _, _, _, info) = env.step(action)
      observation = next_state
      //desired_g = desired_g_next
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

func ddpg(actor_critic: ActorCritic, env: MultiGoalEnvironmentWrapper,
          epochs: Int = 50, episodes: Int = 40, rollouts: Int = 2,
          stepsPerEpisode: Int = 300, update_steps: Int = 30, batchSize: Int = 32, tau: Float = 0.001,
          update_every: Int = 1) ->([Float], [Float], [Float])  {

    // var totalRewards: [Float] = []
    // var movingAverageReward: [Float] = []
    var total_actor_losses: [Float] = []
    var total_critic_losses: [Float] = []
    var success_rates: [Float] = []
    actor_critic.updateCriticTargetNetwork(tau: 1.0)
    actor_critic.updateActorTargetNetwork(tau: 1.0)
    print("Starting Training\n")
    for epoch in 0..<epochs {
      print("\nEPOCH: \(epoch)")
      if epoch == 0 || epoch == 25 || epoch == 50 || epoch == 75 {
        let filename: String = "results/fetch_push_ddpg_ep" + String(epoch) + ".npy"
        test_agent(agent: actor_critic, env: env, num_steps: 60, filename: filename)
      }
      var actor_losses: [Float] = []
      var critic_losses: [Float] = []
      for i in 0..<episodes {
        print("\nEPISODE: \(i)")

        for _ in 0..<rollouts {

          var (observation, _, _) = env.reset()
          var t : Int = 0
          while t < stepsPerEpisode {
            let action: Tensor<Float> = actor_critic.get_action(state: observation, env: env, training: true)
            let (obs_next, _, _, reward, isDone, _) = env.step(action)
            actor_critic.remember(state:observation, action:action, reward:reward, next_state:obs_next, dones:isDone)
            observation = obs_next
            t += 1
          }
        }
        // actor_critic.remember(episode_batch: [mb_obs, mb_ag, mb_g, mb_actions])
        var actor_loss_total : Float = 0.0
        var critic_loss_total : Float = 0.0
        for _ in 0..<update_steps {
          let(a_loss, c_loss) = actor_critic.train_actor_critic(batchSize: batchSize)
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
  var (observation, _, _) = env.reset()
  var success: [Float] = []
  for _ in 0..<num_steps {
    let frame = env.originalEnv.render(mode: "rgb_array")
    frames.append(frame)
    let action = agent.get_action(state: observation,env: env, training: false)
    let (next_state, _, _, tfreward, _, info) = env.step(action)
    let reward : Float = tfreward.scalarized()
    print("Reward: \(reward)\n")
    success.append(Float(info["is_success"])!)
    observation = next_state
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
let actor_net: ActorNetwork = ActorNetwork(observationSize: env.state_size, actionSize: env.action_size, hiddenLayerSizes: [256, 256, 256], maximum_action:max_action)
let actor_target: ActorNetwork = ActorNetwork(observationSize: env.state_size, actionSize: env.action_size, hiddenLayerSizes: [256, 256, 256], maximum_action:max_action)
let critic_net: CriticNetwork = CriticNetwork(state_size: env.state_size,  action_size: env.action_size, hiddenLayerSizes: [256, 256, 256], outDimension: 1)
let critic_target: CriticNetwork = CriticNetwork(state_size: env.state_size, action_size: env.action_size, hiddenLayerSizes: [256, 256, 256], outDimension: 1)


let actor_critic: ActorCritic = ActorCritic(actor: actor_net,
                                            actor_target: actor_target,
                                            critic: critic_net,
                                            critic_target: critic_target,
                                            stateSize: env.state_size,
                                            actionSize: env.action_size,
                                            maxAction: max_action)
let(actor_losses, critic_losses, success_rates) = ddpg(actor_critic: actor_critic,
                                                           env: env,
                                                           epochs: 60,
                                                           episodes: 50,
                                                           rollouts: 3,
                                                           stepsPerEpisode: env.max_timesteps,
                                                           update_steps: 50,
                                                           batchSize: 128,
                                                           tau: 0.005,
                                                           update_every: 5)

plt.plot(success_rates)
plt.title("DDPG Success Rate on FetchPush-v1")
plt.xlabel("Epoch")
plt.ylabel("Eval. Success Rate")
plt.savefig("results/rewards/fetch_push_success_ddpg.png")
plt.clf()
let success_rates_np = np.array(success_rates)
np.save("results/rewards/fetch_push_success_ddpg.npy", success_rates_np)


plt.plot(actor_losses)
plt.title("DDPG Avg. Actor Loss on FetchPush-v1")
plt.xlabel("Epoch")
plt.ylabel("Avg. Loss")
plt.savefig("results/losses/fetch_push_actor_loss_ddpg.png")
plt.clf()

plt.plot(critic_losses)
plt.title("DDPG Avg. Critic Loss on FetchPush-v1")
plt.xlabel("Epoch")
plt.ylabel("Avg. Loss")
plt.savefig("results/losses/fetch_push_critic_loss_ddpg.png")
plt.clf()

test_agent(agent: actor_critic, env: env, num_steps: 60, filename: "results/fetch_push_ddpg.npy")
