///Implementation of the Soft Actor Critic Algorithm
/// This implementation uses swift for Tensorflow and borrows ideas from
/// the swift-models repository: https://github.com/tensorflow/swift-models/
/// Original Paper:  https://arxiv.org/pdf/1801.01290.pdf
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


//Taken from the DQN example in the tensorflow swift-models repo
//: https://github.com/tensorflow/swift-models/blob/master/Gym/DQN/main.swift
class TensorFlowEnvironmentWrapper {
  let originalEnv: PythonObject
  public let state_size: Int
  public let action_size: Int

  init(_ env: PythonObject) {
    self.originalEnv = env
    self.state_size = Int(env.observation_space.shape[0])!
    self.action_size = Int(env.action_space.shape[0])!
  }

  func reset() -> Tensor<Float> {
    let state = self.originalEnv.reset()
    return Tensor<Float>(numpy: np.array(state, dtype: np.float32))!
  }

  func step(_ action: Tensor<Float>) -> (
    state: Tensor<Float>, reward: Tensor<Float>, isDone: Tensor<Bool>, info: PythonObject
  ) {
    let (state, reward, isDone, info) = originalEnv.step([action.scalarized()]).tuple4
    let tfState = Tensor<Float>(numpy: np.array(state, dtype: np.float32))!
    let tfReward = Tensor<Float>(numpy: np.array(reward, dtype: np.float32))!
    let tfIsDone = Tensor<Bool>(numpy: np.array(isDone, dtype: np.bool))!
    return (tfState, tfReward, tfIsDone, info)
  }

  func set_environment_seed(seed: Int) {
    self.originalEnv.seed(seed)
  }

  func action_sample() -> Tensor<Float> {
    let action = originalEnv.action_space.sample()
    let tfAction = Tensor<Float>(numpy: np.array(action, dtype: np.float32))!
    return tfAction
  }
}



//Diagonal Gaussian Distribution
struct DiagonalGaussian {

    public var dim: Int

    init(dimension: Int ){
        self.dim = dimension
    }

    func KLDivergence(old_mean: Tensor<Float>, old_log_std:Tensor<Float>, new_mean: Tensor<Float>, new_log_std: Tensor<Float>) -> Tensor<Float> {
        let old_std: Tensor<Float> = exp(old_log_std)
        let new_std: Tensor<Float> = exp(new_log_std)
        let numerator: Tensor<Float> = pow(old_mean - new_mean, 2) + pow(old_std, 2) - pow(new_std, 2)
        let denominator: Tensor<Float> = 2*pow(new_std, 2) + 0.0000000001
        let result = (numerator/denominator) + new_log_std - old_log_std
        return result.sum()
    }

    func log_likelihood(x: Tensor<Float>, means: Tensor<Float>, log_stds: Tensor<Float>) -> Tensor<Float> {
        let z_s: Tensor<Float> = (x - means) / exp(log_stds)
        var result: Tensor<Float> = -log_stds.sum(alongAxes: -1) - (0.5*pow(z_s, 2)).sum(alongAxes: -1)
        let pi: Float = Float(np.pi)!
        result = result - 0.5 * Float(self.dim) * log(2 * Tensor<Float>(pi))
        return result
    }

    func sample(means: Tensor<Float>, log_stds: Tensor<Float>) -> Tensor<Float> {
        //let rand_normal: Tensor<Float> = Tensor<Float>(randomNormal:means.shape, mean: Tensor<Float>(0.0), standardDeviation: Tensor<Float>(0.0))
        let rand_normal: Tensor<Float> = Tensor<Float>(randomNormal: means.shape, mean: Tensor<Float>(0.0), standardDeviation: Tensor<Float>(1.0))
        let result = means + rand_normal * exp(log_stds)
        return result
    }


    func entropy(log_stds: Tensor<Float>) -> Tensor<Float> {
        let value: Float = Float(2 * np.pi * np.e)!
        let result: Tensor<Float> = log_stds + log(sqrt(Tensor<Float>(value)))
        return result.sum(alongAxes: -1)
    }

}

//Actor Network that produces the mean and standard deviation for
//a Gaussian Distribution. This distribution is used to sample the actions
// for a given state during training
//During testing, the mean of the distribution is treated like the action we
//want to take for a given state
struct GaussianActorNetwork: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    public var batch_norm : BatchNorm<Float>

    public var layer_1, layer_2, out_mean, out_log_std: Dense<Float>

    public var dim: Tensor<Float>

    @noDerivative let log_sig_max: Float = 2.0

    @noDerivative let log_sig_min: Float = -20.0

    @noDerivative let eps: Float = 0.000001
    @noDerivative public var dist: DiagonalGaussian

    @noDerivative public var max_action: Tensor<Float>

    init(state_size: Int, action_size: Int, hiddenLayerSizes: [Int] = [400, 300], maximum_action: Tensor<Float>) {
        self.dim = Tensor<Float>(Float(action_size))
        self.batch_norm = BatchNorm<Float>(featureCount: state_size, axis: -1, momentum:0.95)
        self.layer_1 = Dense<Float>(inputSize: state_size, outputSize: hiddenLayerSizes[0], activation: relu)
        self.layer_2 = Dense<Float>(inputSize: hiddenLayerSizes[0], outputSize: hiddenLayerSizes[1], activation: relu)
        self.out_mean = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: action_size, activation:identity)
        self.out_log_std = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: action_size, activation:identity)
        self.dist = DiagonalGaussian(dimension: action_size)
        self.max_action = maximum_action
    }

    @differentiable
    func sample(means: Tensor<Float>, log_stds: Tensor<Float>) -> Tensor<Float> {

        let rand_normal: Tensor<Float> = Tensor<Float>(randomNormal:means.shape, mean: Tensor<Float>(zeros: means.shape), standardDeviation: Tensor<Float>(ones: log_stds.shape))
        let result = means + rand_normal * exp(log_stds)
        return result
    }

    @differentiable
    func log_likelihood(x: Tensor<Float>, means: Tensor<Float>, log_stds: Tensor<Float>) -> Tensor<Float> {
        let z_s: Tensor<Float> = (x - means) / exp(log_stds)
        var result: Tensor<Float> = -log_stds.sum(alongAxes: -1) - (0.5*pow(z_s, 2)).sum(alongAxes: -1)
        let pi: Float = 3.141592653589793
        result = result - 0.5 * self.dim * log(Tensor<Float>(2*pi))
        return result
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        let norm_input = batch_norm(input)
        let h1 = layer_1(norm_input)
        let h2 = layer_2(h1)
        let mu = out_mean(h2)
        let log_std = out_log_std(h2)
        let clipped_log_std: Tensor<Float> = log_std.clipped(min: self.log_sig_min, max:self.log_sig_max)
        //During training we sample from a Diagonal Gaussian.
        //During testing we can just take our mean as our raw actions
        let raw_actions_testing: Tensor<Float> = mu
        let raw_actions_training: Tensor<Float> = self.sample(means:mu, log_stds: clipped_log_std)

        // let raw_actions_training: Tensor<Float> = self.dist.sample(means: mu, log_stds: clipped_log_std)
        // var logp_pis: Tensor<Float> = self.dist.log_likelihood(x: raw_actions_training, means: mu, log_stds: clipped_log_std)
        //var logp_pis_test: Tensor<Float> = self.dist.log_likelihood(x: raw_actions_testing, means: mu, log_stds: clipped_log_std)

        var logp_pis: Tensor<Float> = self.log_likelihood(x: raw_actions_training, means: mu, log_stds: clipped_log_std)
        //var logp_pis_test: Tensor<Float> = self.log_likelihood(x: raw_actions_testing, means: mu, log_stds: clipped_log_std)

        //apply a squashing function to the raw_actions
        let actions_training: Tensor<Float> = tanh(raw_actions_training)
        let actions_testing: Tensor<Float> = tanh(raw_actions_testing)

        //squash correction
        let diff_train: Tensor<Float> = (log(1.0 - pow(actions_training, 2) + Tensor<Float>(eps))).sum(alongAxes: 1)
        //let diff_test = (log(1.0 - pow(actions_testing, 2) + Tensor<Float>(eps))).sum(alongAxes: 1)
        logp_pis = logp_pis - diff_train
        //logp_pis_test = logp_pis -  diff_test
        return Tensor<Float>([actions_training * self.max_action, actions_testing * self.max_action, logp_pis, mu, log_std])

    }

}

struct CriticQNetwork: Layer {
  typealias Input = [Tensor<Float>]
  typealias Output = [Tensor<Float>]

  public var batch_norm : BatchNorm<Float>
  public var layer_1, layer_2, layer_3 : Dense<Float>
  public var layer_4, layer_5, layer_6 : Dense<Float>

  init(state_size: Int, action_size:Int, hiddenLayerSizes: [Int] = [400, 300], outDimension: Int) {
    self.batch_norm = BatchNorm<Float>(featureCount: state_size, axis:-1, momentum: 0.95)
    self.layer_1 = Dense<Float>(inputSize: state_size + action_size, outputSize: hiddenLayerSizes[0], activation:relu)
    self.layer_2 = Dense<Float>(inputSize: hiddenLayerSizes[0], outputSize: hiddenLayerSizes[1], activation:relu)
    self.layer_3 = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: outDimension, activation: identity)

    self.layer_4 = Dense<Float>(inputSize: state_size + action_size, outputSize: hiddenLayerSizes[0], activation:relu)
    self.layer_5 = Dense<Float>(inputSize: hiddenLayerSizes[0], outputSize: hiddenLayerSizes[1], activation:relu)
    self.layer_6 = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: outDimension, activation: identity)

  }

  @differentiable
  func callAsFunction(_ input: Input) -> Output{
    let state: Tensor<Float> = input[0]
    let normed_state = batch_norm(state)
    let action: Tensor<Float> = input[1]
    let state_and_action = Tensor(concatenating: [normed_state, action], alongAxis: 1)
    let h1 = layer_1(state_and_action)
    let h2 = layer_2(h1)
    let q_value_1 = layer_3(h2)

    let h3 = layer_4(state_and_action)
    let h4 = layer_5(h3)
    let q_value_2 = layer_6(h4)
    return [q_value_1, q_value_2]
  }

}

struct CriticVNetwork: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    public var batch_norm : BatchNorm<Float>
    public var layer_1, layer_2, layer_3 : Dense<Float>

    init(state_size: Int, hiddenLayerSizes: [Int] = [256, 256], outDimension: Int) {
        self.batch_norm = BatchNorm<Float>(featureCount: state_size, axis: -1, momentum: 0.95)
        self.layer_1 = Dense<Float>(inputSize: state_size, outputSize: hiddenLayerSizes[0], activation:relu)
        self.layer_2 = Dense<Float>(inputSize: hiddenLayerSizes[0], outputSize:hiddenLayerSizes[1], activation:relu)
        self.layer_3 = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: outDimension, activation: identity)
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        let normed_input = batch_norm(input)
        let h1 = layer_1(normed_input)
        let h2 = layer_2(h1)
        let output = layer_3(h2)
        return output
    }

}


struct AlphaLayer : Layer {
    typealias Output = Tensor<Float>

    @differentiable
    public var log_alpha : Tensor<Float>

    init(log_alpha_init: Tensor<Float>) {
      self.log_alpha = log_alpha_init
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Output {
      //let alpha: Tensor<Float> = exp(self.log_alpha)
      let log_result: Tensor<Float> = self.log_alpha*input
      return log_result
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

//Soft Actor Critic Agent
class SoftActorCritic {


  public var actor_network: GaussianActorNetwork

  public var critic_q1: CriticQNetwork

  public var critic_v_network: CriticVNetwork

  public var target_critic_v_network: CriticVNetwork

  public var replayBuffer: ReplayBuffer

  public var alpha_net : AlphaLayer

  //let action_noise: GaussianNoise<Float>
  let action_noise: OUNoise

  let gamma: Float

  let state_size: Int

  let action_size: Int

  public var alpha: Tensor<Float>

  public var log_alpha: Tensor<Float>

  let target_alpha: Tensor<Float>

  let actor_optimizer: Adam<GaussianActorNetwork>

  let critic_q1_optimizer: Adam<CriticQNetwork>

  let critic_v_optimizer: Adam<CriticVNetwork>

  let alpha_optimizer: Adam<AlphaLayer>

  let alpha_lr : Float

  let train_alpha: Bool

  //let alpha_optimizer: Adam<Tensor<Float>>

  init (
    actor: GaussianActorNetwork,
    critic_1: CriticQNetwork,
    critic_v: CriticVNetwork,
    critic_v_target: CriticVNetwork,
    stateSize: Int,
    actionSize: Int,
    critic_lr: Float = 0.0001,
    actor_lr: Float = 0.0001,
    critic_v_lr: Float = 0.0001,
    alpha: Float = 0.2,
    trainAlpha: Bool = false,
    gamma: Float = 0.95) {
      self.actor_network = actor
      self.critic_q1 = critic_1
      self.critic_v_network = critic_v
      self.target_critic_v_network = critic_v_target
      // self.target_actor_network = actor_target
      //Init OUNoise (not really used)
      self.gamma = gamma
      let mu : Tensor<Float> = Tensor<Float>(0.0)
      let sigma : Tensor<Float> = Tensor<Float>(0.15)
      let x_init: Tensor<Float> = Tensor<Float>(-0.01)
      self.action_noise = OUNoise(mu: mu, sigma: sigma, x_init: x_init, theta: 0.15, dt: 0.05)
      self.state_size = stateSize
      self.action_size = actionSize

      //init alpha
      self.alpha = Tensor<Float>(alpha)
      self.log_alpha = log(self.alpha)
      self.alpha_net = AlphaLayer(log_alpha_init: self.log_alpha)
      self.target_alpha = Tensor<Float>(Float(-self.action_size))
      self.alpha_lr = critic_lr
      self.train_alpha = trainAlpha

      //init optimizers
      self.actor_optimizer = Adam(for: self.actor_network, learningRate: actor_lr)
      self.critic_q1_optimizer = Adam(for: self.critic_q1, learningRate: critic_lr)
      self.critic_v_optimizer = Adam(for: self.critic_v_network, learningRate: critic_v_lr)
      self.alpha_optimizer = Adam(for: self.alpha_net, learningRate:actor_lr)

      self.replayBuffer = ReplayBuffer(capacity: 2000, combined: true)
  }

  func remember(state: Tensor<Float>, action: Tensor<Float>, reward: Tensor<Float>, next_state: Tensor<Float>, dones: Tensor<Bool>) {
    self.replayBuffer.append(state:state, action:action, reward:reward, nextState:next_state, isDone:dones)
  }

  func get_action(state: Tensor<Float>, env: TensorFlowEnvironmentWrapper, training: Bool) -> Tensor<Float> {

    let tfState = Tensor<Float>(numpy: np.expand_dims(state.makeNumpyArray(), axis: 0))!
    let net_output: Tensor<Float> = self.actor_network(tfState)
    let actions_training: Tensor<Float> = net_output[0]
    let actions_testing: Tensor<Float> = net_output[1]
    if training {
      // let noise = self.action_noise.getNoise()
      // let noisy_action = actions_training + noise
      // let action = noisy_action.clipped(min:-2.0, max:2.0)
      let action = actions_training
      return action[0]
    } else {
      return actions_testing[0]
    }

  }

  func train_actor_critic(batchSize: Int, iterationNum: Int) -> (Float, Float, Float) {

    let (states, actions, rewards, nextstates, dones) = self.replayBuffer.sample(batchSize: batchSize)

    //train critic_q1
    let(critic_q1_loss, critic_q1_gradients) = valueWithGradient(at: self.critic_q1) { critic_q1 -> Tensor<Float> in
      //get target q values from target critic network
      let value_next_target: Tensor<Float> = self.target_critic_v_network(nextstates).flattened()
      let target_q_values: Tensor<Float> =  rewards + self.gamma * (1 - Tensor<Float>(dones)) * value_next_target

      //get predicted q values from critic network
      let target_q_values_no_deriv : Tensor<Float> = withoutDerivative(at: target_q_values)
      let output = critic_q1([states, actions])
      let predicted_q_values_1 : Tensor<Float> = output[0].flattened()
      let predicted_q_values_2 : Tensor<Float> = output[1].flattened()
      let huber_loss_1 = huberLoss(predicted: predicted_q_values_1, expected: target_q_values_no_deriv, delta: 5.0).mean()
      let huber_loss_2 = huberLoss(predicted: predicted_q_values_2, expected: target_q_values_no_deriv, delta: 5.0).mean()
      let td_error_1: Tensor<Float> = target_q_values_no_deriv - predicted_q_values_1
      let td_error_2: Tensor<Float> = target_q_values_no_deriv - predicted_q_values_2
      let td_loss_1: Tensor<Float> = 0.5*(pow(td_error_1, 2)).mean()
      let td_loss_2: Tensor<Float> = 0.5*(pow(td_error_2, 2)).mean()
      let td_loss = td_loss_1 + td_loss_2
      return td_loss
      //return huber_loss_1 + huber_loss_2
      //return huberLoss(predicted: predicted_q_values, expected: target_q_values_no_deriv, delta: 5.0).mean()
    }
    self.critic_q1_optimizer.update(&self.critic_q1, along: critic_q1_gradients)

    //train critic_q2
    // let(critic_q2_loss, critic_q2_gradients) = valueWithGradient(at: self.critic_q2) { critic_q2 -> Tensor<Float> in
    //   //get target q values from target critic network
    //   // let value_next_target: Tensor<Float> = self.target_critic_v_network(nextstates).flattened()
    //   // let target_q_values: Tensor<Float> =  rewards + self.gamma * (1 - Tensor<Float>(dones)) * value_next_target
    //   //get predicted q values from critic network
    //   let target_q_values_no_deriv : Tensor<Float> = withoutDerivative(at: target_q_values)
    //   let predicted_q_values: Tensor<Float> = critic_q2([states, actions]).flattened()
    //   let td_error: Tensor<Float> = target_q_values_no_deriv - predicted_q_values
    //   let td_loss: Tensor<Float> = 0.5*(pow(td_error, 2)).mean()
    //   return td_loss
    //   //return huberLoss(predicted: predicted_q_values, expected: target_q_values_no_deriv, delta: 5.0).mean()
    // }
    // self.critic_q2_optimizer.update(&self.critic_q2, along: critic_q2_gradients)
    //train value network
    let(critic_v_loss, critic_v_gradients) = valueWithGradient(at: self.critic_v_network) { critic_v_network -> Tensor<Float> in
        let current_v: Tensor<Float> = critic_v_network(states).flattened()
        let actor_output = self.actor_network(states)
        let sample_actions: Tensor<Float> = actor_output[0]
        let logp_pi: Tensor<Float> = actor_output[2].flattened()

        let critic_out = self.critic_q1([states, sample_actions])
        let current_q1: Tensor<Float> = critic_out[0].flattened()
        let current_q2: Tensor<Float> = critic_out[1].flattened()

        // let current_q1: Tensor<Float> = self.critic_q1([states, sample_actions])
        // let current_q2: Tensor<Float> = self.critic_q2([states, sample_actions])
        let minimum_q: Tensor<Float> = min(current_q1, current_q2)

        //let target_values: Tensor<Float> = minimum_q - self.alpha*logp_pi
        let target_values_no_deriv: Tensor<Float> = withoutDerivative(at: minimum_q - self.alpha*logp_pi)
        let td_error: Tensor<Float> = target_values_no_deriv - current_v
        let td_loss: Tensor<Float> = 0.5*(pow(td_error, 2)).mean()
        return td_loss
        //return huberLoss(predicted: current_v, expected: target_values_no_deriv, delta: 3.5).mean()
    }
    //print(critic_v_gradients)
    self.critic_v_optimizer.update(&self.critic_v_network, along: critic_v_gradients)


    //train actor
    let(actor_loss, actor_gradients) = valueWithGradient(at: self.actor_network) { actor_network -> Tensor<Float> in
        //let next_actions = actor_network(states)
        let actor_output = actor_network(states)
        let sample_actions: Tensor<Float> = actor_output[0]
        let logp_pi: Tensor<Float> = actor_output[2].flattened()

        // let current_q1: Tensor<Float> = self.critic_q1([states, sample_actions])
        // let current_q2: Tensor<Float> = self.critic_q2([states, sample_actions])
        let critic_out = self.critic_q1([states, sample_actions])
        let current_q1: Tensor<Float> = critic_out[0].flattened()
        let current_q2: Tensor<Float> = critic_out[1].flattened()
        let minimum_q: Tensor<Float> = withoutDerivative(at:min(current_q1, current_q2))

        let error: Tensor<Float> = self.alpha*logp_pi - minimum_q
        let loss: Tensor<Float> = error.mean()
        return loss
    }
    self.actor_optimizer.update(&self.actor_network, along: actor_gradients)


    if self.train_alpha {
      //train_alpha
      let(_, alpha_gradients) = valueWithGradient(at: self.alpha_net) { alpha_net -> Tensor<Float> in
        let actor_output = self.actor_network(states)
        let logp_pi : Tensor<Float> = actor_output[2].flattened()
        let target_value = withoutDerivative(at: self.target_alpha - logp_pi)
        let output : Tensor<Float> = alpha_net(target_value)
        let loss = -1.0 * output.mean()
        return loss

      }
      self.alpha_optimizer.update(&self.alpha_net, along:alpha_gradients)
      self.alpha = exp(self.alpha_net.log_alpha).clipped(min: 0.0,  max: 1.0)

    }

    return (actor_loss.scalarized(), critic_q1_loss.scalarized(), critic_v_loss.scalarized())
  }

  func updateValueTargetNetwork(tau: Float) {
    //update layer 1
    self.target_critic_v_network.layer_1.weight =
              tau * Tensor<Float>(self.critic_v_network.layer_1.weight) + (1 - tau) * self.target_critic_v_network.layer_1.weight
    self.target_critic_v_network.layer_1.bias =
              tau * Tensor<Float>(self.critic_v_network.layer_1.bias) + (1 - tau) * self.target_critic_v_network.layer_1.bias
    //update layer 2
    self.target_critic_v_network.layer_2.weight =
              tau * Tensor<Float>(self.critic_v_network.layer_2.weight) + (1 - tau) * self.target_critic_v_network.layer_2.weight
    self.target_critic_v_network.layer_2.bias =
              tau * Tensor<Float>(self.critic_v_network.layer_2.bias) + (1 - tau) * self.target_critic_v_network.layer_2.bias
    //update layer 3
    self.target_critic_v_network.layer_3.weight =
              tau * Tensor<Float>(self.critic_v_network.layer_3.weight) + (1 - tau) * self.target_critic_v_network.layer_3.weight
    self.target_critic_v_network.layer_3.bias =
              tau * Tensor<Float>(self.critic_v_network.layer_3.bias) + (1 - tau) * self.target_critic_v_network.layer_3.bias

  }


 }



//training algorithm for SoftActorCritic
func sac_train(actor_critic: SoftActorCritic, env: TensorFlowEnvironmentWrapper,
          maxEpisodes: Int = 1000, batchSize: Int = 32,
          stepsPerEpisode: Int = 300, tau: Float = 0.001,
          update_every: Int = 1, epsilonStart: Float = 0.99,
          epsilonEnd:Float = 0.01, epsilonDecay: Float = 1000) ->([Float], [Float], [Float], [Float], [Float]) {
    var totalRewards: [Float] = []
    var movingAverageReward: [Float] = []
    var actor_losses: [Float] = []
    var critic_1_losses: [Float] = []
    //var critic_2_losses: [Float] = []
    var value_losses: [Float] = []
    var bestReward: Float = -99999999.0
    var training: Bool = false
    let sampling_episodes: Int = 5
    actor_critic.updateValueTargetNetwork(tau: 1.0)
    for i in 0..<maxEpisodes {
      print("\nEpisode: \(i)")
      var state = env.reset()
      print(state)
      //sample random actions for the first few episodes, then start using actor network w/ noise
      if i == sampling_episodes {
        print("Finished Warmup Episodes")
        print("Starting Training")
        training = true
      }
      var totalReward: Float = 0
      var totalActorLoss: Float = 0
      var totalCriticQ1Loss: Float = 0
      var totalValueLoss: Float = 0
      var totalTrainingSteps: Int = 0
      for j in 0..<stepsPerEpisode {

        var action: Tensor<Float>
        //Sample random action or take action from actor depending on epsilon
        if i < sampling_episodes {
          action = env.action_sample()
        } else {
          action = actor_critic.get_action(state: state , env: env , training: true)
        }
        let(nextState, reward, isDone, _) = env.step(action)
        totalReward += reward.scalarized()
        //add (s, a, r, s') to actor_critic's replay buffer
        actor_critic.remember(state:state, action:action, reward:reward, next_state:nextState, dones:isDone)

        if actor_critic.replayBuffer.count > batchSize && training{
          totalTrainingSteps += 1
          //Train Actor and Critic Networks
          let(actor_loss, critic_q1_loss, value_loss) = actor_critic.train_actor_critic(batchSize: batchSize, iterationNum: j)
          totalActorLoss += actor_loss
          totalCriticQ1Loss += critic_q1_loss
          //totalCriticQ2Loss += critic_q2_loss
          totalValueLoss += value_loss
        }

        if j % update_every == 0 {
            actor_critic.updateValueTargetNetwork(tau: tau)
        }

        state = nextState
      }
      totalRewards.append(totalReward)

      if totalRewards.count < 10 {
        var sum: Float = 0.0
        for k in 0..<totalRewards.count{
          let reward_k = totalRewards[k]
          sum += reward_k
        }
        let avgTotal = sum/Float(totalRewards.count)
        if avgTotal > bestReward {
          bestReward = avgTotal
        }
      }

      if totalRewards.count >= 10 {
        var sum: Float = 0.0
        for k in totalRewards.count - 10..<totalRewards.count {
          let reward_k: Float  = totalRewards[k]
          sum += reward_k
        }
        let avgTotal: Float = sum/10
        movingAverageReward.append(avgTotal)
        if avgTotal > bestReward {
          bestReward = avgTotal
        }
      }
      if training {
        let avgActorLoss: Float = totalActorLoss/Float(totalTrainingSteps)
        let avgCriticQ1Loss: Float = totalCriticQ1Loss/Float(totalTrainingSteps)
        //let avgCriticQ2Loss: Float = totalCriticQ2Loss/Float(totalTrainingSteps)
        let avgValueLoss: Float = totalValueLoss/Float(totalTrainingSteps)
        actor_losses.append(avgActorLoss)
        critic_1_losses.append(avgCriticQ1Loss)
        //critic_2_losses.append(avgCriticQ2Loss)
        value_losses.append(avgValueLoss)
        print(String(format: "Episode: %4d | Total Reward %.03f | Best Avg. Reward: %.03f | Avg. Actor Loss: %.03f | Avg. Critic 1 Loss: %.03f | Avg. Value Loss: %.03f",
        i, totalReward, bestReward, avgActorLoss, avgCriticQ1Loss, avgValueLoss))
        print(actor_critic.alpha)
        print(actor_critic.alpha_net.log_alpha)
      } else {
        print(String(format: "Episode: %4d | Total Reward %.03f | Best Avg. Reward: %.03f",
        i, totalReward, bestReward))
      }
    }
    print("Finished Training")
    return (totalRewards, movingAverageReward, actor_losses, critic_1_losses, value_losses)
}



func evaluate_agent(agent: SoftActorCritic, env: TensorFlowEnvironmentWrapper, num_steps: Int = 300) {
  var frames: [PythonObject] = []
  var state = env.reset()
  var totalReward: Float = 0.0
  for _ in 0..<num_steps {
    let frame = env.originalEnv.render(mode: "rgb_array")
    frames.append(frame)
    let action = agent.get_action(state: state, env: env, training: false)
    let (next_state, reward, _, _) = env.step(action)
    let scalar_reward = reward.scalarized()
    print("\nStep Reward: \(scalar_reward)")
    totalReward += scalar_reward
    state = next_state
  }
  env.originalEnv.close()
  let frame_np_array = np.array(frames)
  np.save("results/sac_pendulum_frames_5.npy", frame_np_array)
  print("\n Total Reward: \(totalReward)")
}


//train actor critic on pendulum environment
let env = TensorFlowEnvironmentWrapper(gym.make("Pendulum-v0"))
env.set_environment_seed(seed: 1001)
let max_action: Tensor<Float> = Tensor<Float>(2.0)
let actor_net: GaussianActorNetwork = GaussianActorNetwork(state_size: env.state_size, action_size: env.action_size, hiddenLayerSizes: [300, 200], maximum_action:max_action)
let critic_q1: CriticQNetwork = CriticQNetwork(state_size: env.state_size, action_size: env.action_size, hiddenLayerSizes: [300, 300], outDimension: 1)
//let critic_q2: CriticQNetwork = CriticQNetwork(state_size: 3, action_size: 1, hiddenLayerSizes: [300, 200], outDimension: 1)
let critic_v: CriticVNetwork = CriticVNetwork(state_size: env.state_size, hiddenLayerSizes:[300, 200], outDimension: 1)
let critic_v_target: CriticVNetwork = CriticVNetwork(state_size: env.state_size, hiddenLayerSizes:[300, 200], outDimension: 1)

let actor_critic: SoftActorCritic = SoftActorCritic(actor: actor_net,
                                                    critic_1: critic_q1,
                                                    critic_v: critic_v,
                                                    critic_v_target: critic_v_target,
                                                    stateSize: 3, actionSize: 1, alpha:1.0, train_alpha: true, gamma: 0.99)
Context.local.learningPhase = .training
let(totalRewards, movingAvgReward, actor_losses, critic_1_losses, value_losses)
  = sac_train(actor_critic: actor_critic,
        env: env,
        maxEpisodes: 1500,
        batchSize: 32,
        stepsPerEpisode: 200,
        tau: 0.005,
        update_every: 75,
        epsilonStart: 0.99,
        epsilonDecay: 150)


//plot results
plt.plot(totalRewards)
plt.title("SAC on Pendulum-v0 Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.savefig("results/rewards/pendulum-sacreward-5.png")
plt.clf()
let totalRewards_arr = np.array(totalRewards)
np.save("results/rewards/pendulum-sacreward-5.npy", totalRewards)

// Save smoothed learning curve
let runningMeanWindow: Int = 10
let smoothedEpisodeReturns = np.convolve(
  totalRewards, np.ones((runningMeanWindow)) / np.array(runningMeanWindow, dtype: np.int32),
  mode: "same")
plt.plot(movingAvgReward)
plt.title("SAC on Pendulum-v0 Average Rewards")
plt.xlabel("Episode")
plt.ylabel("Smoothed Episode Reward")
plt.savefig("results/rewards/pendulum-sacsmoothedreward-5.png")
plt.clf()

let avgRewards_arr = np.array(movingAvgReward)
np.save("results/rewards/pendulum-sacavgreward-5.npy", avgRewards_arr)

//save actor and critic losses
plt.plot(critic_1_losses)
plt.title("SAC on Pendulum-v0 critic losses")
plt.xlabel("Episode")
plt.ylabel("TD Loss")
plt.savefig("results/losses/sac-critic-losses-5.png")
plt.clf()


plt.plot(actor_losses)
plt.title("SAC on Pendulum-v0 actor losses")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.savefig("results/losses/sac-actor-losses-5.png")
plt.clf()

plt.plot(value_losses)
plt.title("SAC on Pendulum-v0 Value Network losses")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.savefig("results/losses/sac-value-losses-5.png")
plt.clf()

Context.local.learningPhase = .inference
evaluate_agent(agent: actor_critic, env: env, num_steps: 200)







