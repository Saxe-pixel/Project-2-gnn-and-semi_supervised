from itertools import chain
import inspect
import hydra
import torch
from omegaconf import OmegaConf

from utils import seed_everything


@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    # print out the full config
    print(OmegaConf.to_yaml(cfg))

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    seed_everything(cfg.seed, cfg.force_deterministic)

    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)

    dm = hydra.utils.instantiate(cfg.dataset.init)

    # We must pass dataset-specific dimensions to the model.
    # Get them from the datamodule after it has been set up.
    num_node_features = dm.data_train_labeled.num_node_features
    edge_dim = getattr(dm.data_train_labeled, "num_edge_features", None)

    # Build model kwargs based on the model's __init__ signature
    target_cls = hydra.utils.get_class(cfg.model.init._target_)
    sig = inspect.signature(target_cls.__init__)
    model_kwargs = {"num_node_features": num_node_features}
    if "edge_dim" in sig.parameters and edge_dim is not None:
        model_kwargs["edge_dim"] = edge_dim

    model = hydra.utils.instantiate(cfg.model.init, **model_kwargs).to(device)

    if cfg.compile_model:
        model = torch.compile(model)
    models = [model]
    trainer = hydra.utils.instantiate(cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device)

    results = trainer.train(**cfg.trainer.train)
    # results = torch.Tensor(results)
    print("Training finished. Results:", results)



if __name__ == "__main__":
    main()
