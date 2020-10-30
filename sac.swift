///Implementation of the Soft Actor Critic Algorithm
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

  init(_ env: PythonObject) {
    self.originalEnv = env
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


struct DiagonalGaussian {

    public var dim: Int

    init(dimension: Int ){
        self.dim = dimension
    }

    func KLDivergence(old_mean: Tensor<Float>, old_log_std:Tensor<Float>, new_mean: Tensor<Float>, new_log_std: Tensor<Float>) -> Tensor<Float> {
        let old_std: Tensor<Float> = exp(old_log_std)
        let new_std: Tensor<Float> = exp(new_log_std)
        let numerator: Tensor<Float> = pow(old_mean - new_mean, 2) + pow(old_std, 2) - pow(new_std, 2)
        let denominator: Tensor<Float> = 2* pow(new_std, 2) + 1e-8
        let result = (numerator/denominator) + new_log_std - old_log_std
        return result.sum()
    }

    func log_likelihood(x: Tensor<Float>, means: Tensor<Float>, log_stds: Tensor<Float>) -> Tensor<Float> {
        let z_s: Tensor<Float> = (x - means) / exp(log_stds)
        let result = -log_stds.sum(alongAxes: -1) - 0.5* pow(z_s, 2).sum(alongAxes: -1) - 0.5 * self.dim * log(2 * Tensor<Float>(3.14159))
        return result
    }

    func sample(means: Tensor<Float>, log_stds: Tensor<Float>) -> Tensor<Float> {
        let normal_dist = NormalDistribution(mean: Tensor<Float>(zerosLike: means))
        let result = means + normal_dist.next() + exp(log_stds)
        return result
    }

    func entropy(log_stds: Tensor<Float>) -> Tensor<Float> {
        let result: Tensor<Float> = log_stds + log(sqrt(Tensor<Float>(2 * np.pi * np.e)))
        return result.sum(alongAxes: -1)
    }

}


struct GaussianActorNetwork: Layer {
    typealias Input = Tensor<Float>
    typealias Output = [Tensor<Float>]

    let log_sig_max: Float = 2.0
    let log_sig_min: Float = -20.0
    let eps = 0.000001

    public var layer_1, layer_2: Dense<Float>
    public var out_mean: Dense<Float>
    public var out_log_std: Dense<Float>

    @noDerivative
    public var dist: DiagonalGaussian

    @noDerivative
    public var max_action: Tensor<Float>

    init(state_size: Int, action_size: Int, hiddenLayerSizes: [Int] =[400, 300], maximum_action: Tensor<Float>) {
        self.layer_1 = Dense<Float>(inputSize: state_size, outputSize: hiddenLayerSizes[0], activation: relu)
        self.layer_2 = Dense<Float>(inputSize: hiddenLayerSizes[0], outputSize: hiddenLayerSizes[1], activation: relu)
        self.out_mean = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: action_size, activation:identity)
        self.out_log_std = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: action_size)
        self.dist = DiagonalGaussian(dimension: action_size)
        self.max_action = max_action
    }


    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        let h1 = layer_1(input)
        let h2 = layer_2(input)
        let mu = out_mean(h2)
        let log_std = out_log_std(h2)
        let clipped_log_std = log_std.clipped(min: self.log_sig_min, max:self.log_sig_max)
        let raw_actions: Tensor<Float> = tanh(mu)
        var logp_pis: Tensor<Float> = self.dist.log_likelihood(x: raw_actions, means: mu, log_stds: log_std)
        let diff = (log(1.0 - pow(raw_actions, 2) + eps)).sum(alongAxes: 1)
        logp_pis -= diff
        return [raw_actions * self.max_action, logp_pis, mu, log_std]

    }

}


struct CriticQNetwork: Layer {
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

struct CriticVNetwork: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    public var layer_1, layer_2, layer_3 : Dense<Float>

    init(state_size: Int, hiddenLayerSizes: [Int] = [256, 256], outDimension: Int) {
        self.layer_1 = Dense<Float>(inputSize: state_size, outputSize: hiddenLayerSizes[0], activation:relu)
        self.layer_2 = Dense<Float>(inputSize: hiddenLayerSizes[0], outputSize:hiddenLayerSizes[1], activation:relu)
        self.layer_3 = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: outDimension, activation: identity)
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        let h1 = layer_1(input)
        let h2 = layer_2(h1)
        let output = layer_3(h2)
        return output
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


