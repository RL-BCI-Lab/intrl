
from stable_baselines3.common.callbacks import BaseCallback

class Render(BaseCallback):
    def __init__(self, verbose=0):
        super(Render, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.training_env.render()
        
    def _on_training_end(self) -> None:
        self.training_env.close()