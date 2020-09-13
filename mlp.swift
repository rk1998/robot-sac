import PythonKit
PythonLibrary.useVersion(3, 6)
import TensorFlow
let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
let gym = Python.import("gym")
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

let state: Tensor<Float> = Tensor<Float>([0.1, 0.3, 0.4])
print(state.shape)
let tfState = Tensor<Float>(numpy: np.expand_dims(state.makeNumpyArray(), axis: 0))!
let network = MultiLayeredNetwork(observationSize: 3, hiddenLayerSizes: [400, 300], outDimension: 1)
let output = network(tfState)
print(output)
let weights: [Tensor<Float>] = network.weight
for i in 0..<network.weight.count {
    print(network.weight[i].shape)
    print(network.bias[i].shape)
}
