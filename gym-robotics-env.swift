import TensorFlow
import PythonKit
PythonLibrary.useVersion(3, 6)
print(Python.version)
let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
let gym = Python.import("gym")

func randomStep(numSteps: Int) -> Int {
    let env = gym.make("Pendulum-v0")
    env.reset()
    for _ in 0..<numSteps {
      env.render()
      let action = env.action_space.sample()
      print(action)
      let (state, reward, isDone, _) = env.step(action).tuple4
      print(state)
      print(reward)

    }
    env.close()
    return 1
}
randomStep(numSteps: 500)
