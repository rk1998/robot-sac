import TensorFlow
import PythonKit
PythonLibrary.useVersion(3, 6)
print(Python.version)
let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
let gym = Python.import("gym")

func randomStep(numSteps: Int) -> Int {
    let env = gym.make("FetchPush-v1")
    env.reset()
    for _ in 0..<numSteps {
      env.render()
      let (state, reward, isDone, info) = env.step(env.action_space.sample()).tuple4
      print(state) //take a random action

    }
    env.close()
    return 1
}
randomStep(numSteps: 500)
