import wandb


class WandbLogger:
    def __init__(self, project_name: str, config: dict):
        wandb.init(
            project=project_name,
            config=config
        )

    def log_metrics(self, metrics: dict, step: int = None):
        wandb.log(metrics, step=step)

    def log_image(self, image_name: str, image, step: int = None):
        wandb.log({image_name: wandb.Image(image)}, step=step)

    def finish(self):
        wandb.finish()