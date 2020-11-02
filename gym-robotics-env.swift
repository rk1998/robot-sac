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
      let action = env.action_space.sample()
      print("Action Size")
      print(action.shape)
      let (state, reward, isDone, _) = env.step(action).tuple4
      print(state)
      print("Reward")
      print(reward)

    }
    env.close()
    return 1
}
randomStep(numSteps: 500)
