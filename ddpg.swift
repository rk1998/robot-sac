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

	func action_sample() -> Tensor<Float> {
		let action = originalEnv.action_space.sample()
		let tfAction = Tensor<Float>(numpy: np.array(action, dtype: np.float32))!
		return tfAction
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
  //public var state_layer : Dense<Float>
  public var layer_1, layer_2, layer_3 : Dense<Float>
  //public var hidden_layers: MultiLayeredNetwork

  init(state_size: Int, action_size:Int, hiddenLayerSizes: [Int] = [400, 300], outDimension: Int) {
      // self.state_layer = Dense<Float>(inputSize: state_size,
      //                       outputSize: hiddenLayerSizes[0], activation: relu)
      self.layer_1 = Dense<Float>(inputSize: state_size + action_size, outputSize: hiddenLayerSizes[0], activation:relu)
      self.layer_2 = Dense<Float>(inputSize: hiddenLayerSizes[0], outputSize: hiddenLayerSizes[1], activation:relu)
      self.layer_3 = Dense<Float>(inputSize: hiddenLayerSizes[1], outputSize: outDimension, activation: identity)

      // self.hidden_layers = MultiLayeredNetwork(observationSize: hiddenLayerSizes[0] + action_size,
      //                       hiddenLayerSizes: hiddenLayerSizes, outDimension: outDimension)
  }

  @differentiable
  func callAsFunction(_ input: Input) -> Output {
      let state: Tensor<Float> = input[0]
      let action: Tensor<Float> = input[1]
      // let s = self.state_layer(state)
      let state_and_action = Tensor(concatenating: [state, action], alongAxis: 1)
      let h1 = layer_1(state_and_action)
      let h2 = layer_2(h1)
      let q_value = layer_3(h2)
      // let q_value = self.hidden_layers(state_and_action)
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
    let layer_1_result = layer_1(state)
    let layer_2_result = layer_2(layer_1_result)
    let output_action = layer_3(layer_2_result)
    return output_action
    //return state.sequenced(through: layer_1, layer_2, layer_3)
  }

}

//Actor Critic Agent
class ActorCritic {

  public var actor_network: ActorNetwork

  public var critic_network: CriticNetwork

  public var target_critic_network: CriticNetwork

  public var target_actor_network: ActorNetwork

	public var replayBuffer: ReplayBuffer

  let gamma: Float

  let state_size: Int

  let action_size: Int

  let lr : Float

  let actor_optimizer: Adam<ActorNetwork>

  let critic_optimizer: Adam<CriticNetwork>

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

  func get_action(state: Tensor<Float>, epsilon: Float, env: TensorFlowEnvironmentWrapper) -> Tensor<Float> {
			if Float(np.random.uniform()).unwrapped() < epsilon {
				let action = env.action_sample()
				return action
			} else {
        let tfState = Tensor<Float>(numpy: np.expand_dims(state.makeNumpyArray(), axis: 0))!
				let action: Tensor<Float> = self.actor_network(tfState)
				return action[0]
			}
  }

  func train_actor_critic(batchSize: Int, iterationNum: Int) -> (Float, Float) {
      if self.replayBuffer.count >= self.min_buffer_size {
        let (states, actions, rewards, nextstates, dones) = self.replayBuffer.sample(batchSize: batchSize)
        //train critic
        let(critic_loss, critic_gradients) = valueWithGradient(at: critic_network) { critic_network -> Tensor<Float> in
        	// let npActionBatch = actions.makeNumpyArray()
					let npFullIndices = np.stack(
            [np.arange(batchSize, dtype: np.int32)], axis: 1)
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
        return (actor_loss.scalarized(), critic_loss.scalarized())
      } else {
        return (0.0, 0.0)
      }

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


func ddpg(actor_critic: ActorCritic, env: TensorFlowEnvironmentWrapper,
          maxEpisodes: Int = 1000, batchSize: Int = 32,
          stepsPerEpisode: Int = 300, tau: Float = 0.01,
          update_every: Int = 50, epsilonStart: Float = 0.99,
          epsilonEnd:Float = 0.01, epsilonDecay: Float = 1000) ->([Float], [Float], [Float]) {
    var totalRewards: [Float] = []
		var actor_losses: [Float] = []
		var critic_losses: [Float] = []
		var bestReward: Float = -99999999.0
    for i in 0..<maxEpisodes {

			print("\nEpisode: \(i)")
			var state = env.reset()
      print("\nStart State: \(state)")

			var totalReward: Float = 0
      var totalActorLoss: Float = 0
      var totalCriticLoss: Float = 0
			for j in 0..<stepsPerEpisode {
				//epsilon decay
				let epsilon: Float =
					epsilonEnd + (epsilonStart - epsilonEnd) * exp(-1.0 * Float(j) / epsilonDecay)
					//Sample random action or take action from actor depending on epsilon
					let action = actor_critic.get_action(state: state, epsilon: epsilon, env: env)
					let(nextState, reward, isDone, _) = env.step(action)
					totalReward += reward.scalarized()
					//add (s, a, r, s') to actor_critic's replay buffer
					actor_critic.remember(state:state, action:action, reward:reward, next_state:nextState, dones:isDone)
					if actor_critic.replayBuffer.count > batchSize {
							let(actor_loss, critic_loss) = actor_critic.train_actor_critic(batchSize: batchSize, iterationNum: j)
              totalActorLoss += actor_loss
              totalCriticLoss += critic_loss
							// actor_losses.append(actor_loss)
							// critic_losses.append(critic_loss)
					}

          if j % update_every == 0 {
            actor_critic.updateCriticTargetNetwork(tau: tau)
            actor_critic.updateActorTargetNetwork(tau: tau)
          }

					state = nextState
			}
			if totalReward > bestReward {
				bestReward = totalReward
			}
			print(String(format: "Episode: %4d | Total Reward %.03f | Best Reward: %.03f", i, totalReward, bestReward))
			totalRewards.append(totalReward)
      let avgActorLoss: Float = totalActorLoss/Float(maxEpisodes)
      let avgCriticLoss: Float = totalCriticLoss/Float(maxEpisodes)
      actor_losses.append(avgActorLoss)
      critic_losses.append(avgCriticLoss)
			if bestReward > 50 {
				print("Solved in \(i) episodes")
				break
			}
			// print("\n Total Reward: \(totalReward) ")
    }
		print("Finished Training")
    return (totalRewards, actor_losses, critic_losses)
}


func evaluate_agent(agent: ActorCritic, env: TensorFlowEnvironmentWrapper, num_steps: Int = 200) {
  var state = env.reset()
  var totalReward: Float = 0.0
  for _ in 0..<num_steps {
    env.originalEnv.render()
    let action = agent.get_action(state: state, epsilon: 0.0, env: env)
    let (next_state, reward, _, _) = env.step(action)
    totalReward += reward.scalarized()
    state = next_state
  }
  env.originalEnv.close()
  print("\n Total Reward: \(totalReward)")
}


//train actor critic on pendulum environment
let actor_net: ActorNetwork = ActorNetwork(observationSize: 3, actionSize: 1)
let critic_net: CriticNetwork = CriticNetwork(state_size: 3, action_size: 1, outDimension: 1)
let actor_critic: ActorCritic = ActorCritic(actor: actor_net, critic: critic_net, stateSize: 3, actionSize: 1, learning_rate: 0.0001, gamma: 0.99)
let env = TensorFlowEnvironmentWrapper(gym.make("Pendulum-v0"))
let(totalRewards, actor_losses, critic_losses) = ddpg(actor_critic: actor_critic, env: env, maxEpisodes: 500, stepsPerEpisode: 200, tau: 0.0001, epsilonStart: 0.95)
evaluate_agent(agent: actor_critic, env: env, num_steps: 300)

//plot results
plt.plot(totalRewards)
plt.title("DDPG on Pendulum-v0 Rewardss")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.savefig("results/pendulum-ddpgreward.png")
plt.clf()

// Save smoothed learning curve
let runningMeanWindow: Int = 10
let smoothedEpisodeReturns = np.convolve(
  totalRewards, np.ones((runningMeanWindow)) / np.array(runningMeanWindow, dtype: np.int32),
  mode: "same")
plt.plot(smoothedEpisodeReturns)
plt.title("DDPG on Pendulum-v0 Smoothed Rewards")
plt.xlabel("Episode")
plt.ylabel("Smoothed Episode Reward")
plt.savefig("results/pendulum-ddpgsmoothedreward.png")
plt.clf()

//save actor and critic losses
plt.plot(critic_losses)
plt.title("DDPG on Pendulum-v0 critic losses")
plt.xlabel("Episode")
plt.ylabel("TD Loss")
plt.savefig("results/ddpg-critic-losses.png")
plt.clf()


plt.plot(actor_losses)
plt.title("DDPG on Pendulum-v0 actor losses")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.savefig("results/ddpg-actor-losses.png")
plt.clf()
