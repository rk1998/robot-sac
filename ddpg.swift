import Foundation
import PythonKit
PythonLibrary.useVersion(3, 6)
import TensorFlow
let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
let gym = Python.import("gym")

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
  var layers : [Dense<Float>] = []

  init(observationSize: Int, hiddenLayerSizes: [Int], outDimension: Int) {
        let inputSize = observationSize
        let inputLayer = Dense<Float>(inputSize: inputSize, outputSize: hiddenLayerSizes[0], activation: relu)
        self.layers.append(inputLayer)
        for i in 0..<hiddenLayerSizes.count - 1 {
          let layer_i = Dense<Float>(inputSize:hiddenLayerSizes[i], outputSize: hiddenLayerSizes[i + 1], activation: relu)
          self.layers.append(layer_i)
        }
        let outputLayer = Dense<Float>(inputSize: hiddenLayerSizes[hiddenLayerSizes.count - 1], outputSize: outDimension, activation: tanh)
        self.layers.append(outputLayer)
  }

  @differentiable
  func callAsFunction(_ input: Input) -> Output {
      return layers.differentiableReduce(input) { $1($0) }
  }

}
