import dataclasses
import json
import os
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

from util import check_create_folder


@dataclass
class RunSettings:
    run_number: int = None
    run_name: str = None
    datasets: List[str] = field(default_factory=list)
    model_name: str = None

    epochs: int = 30
    batch: int = 512
    learning_rate: float = 0.001
    ema_momentum: Optional[float] = 0.999
    weight_decay: Optional[float] = None
    noise_multiplier: Optional[float] = None
    delta: Optional[float] = None
    l2_norm_clip: float = 1.0
    privacy_epsilon: float = 1.0
    num_microbatches: int = 512
    num_shadow_models: int = 32
    adam_epsilon: float = 1e-7

    model_input_shape: Tuple[int, int, int] = (32, 32, 3)
    random_seed: int = 42

    is_train_model: bool = False
    is_evaluating_model: bool = False
    is_compiling_evalulation: bool = False
    is_running_amia_attack: bool = False
    is_running_mia_attack: bool = False
    is_compiling_attack_results: bool = False
    is_force_model_retrain: bool = False
    is_force_stat_recalculation: bool = False
    is_generating_ds_info: bool = False
    is_force_regenerate_ds_info: bool = False
    is_generating_privacy_report: bool = False

    def __post_init__(self):
        """Post init function to restore the field's default values when initialized with None."""

        if self.num_microbatches is None:
            self.num_microbatches = self.batch

        # if the passed field was None, then apply the dataclasses' default field value
        for f in fields(self):
            if (not isinstance(f.default, dataclasses._MISSING_TYPE)
                    and getattr(self, f.name) is None):
                setattr(self, f.name, f.default)

    def dict(self) -> Dict[str, Any]:
        """Convert all dataclass fields to a dict."""
        return asdict(self)

    def save_settings_as_json(self, filepath: str):
        check_create_folder(filepath)
        filename = os.path.join(filepath, "parameter.json")
        print(f"Saving settings to: {filename}")
        with open(filename, "w") as f:
            json.dump(self.dict(), f, indent=2)

    def print_values(self):
        print(self.dict())


def create_settings_from_args(args):
    return RunSettings(
        run_number=args.run_number,
        run_name=args.run_name,
        datasets=args.datasets,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch_size,
        learning_rate=args.learning_rate,
        ema_momentum=args.momentum,
        weight_decay=args.weight_decay,
        noise_multiplier=args.noise_multiplier,
        l2_norm_clip=args.l2_norm_clip,
        privacy_epsilon=args.epsilon,
        num_microbatches=args.microbatches,
        num_shadow_models=args.shadow_models,
        adam_epsilon=args.adam_epsilon,
        is_train_model=args.train_model,
        is_evaluating_model=args.evaluate_model,
        is_compiling_evalulation=args.compile_evaluation,
        is_running_amia_attack=args.run_amia_attack,
        is_running_mia_attack=args.include_mia,
        is_compiling_attack_results=args.compile_attack_results,
        is_force_model_retrain=args.force_model_retrain,
        is_force_stat_recalculation=args.force_stat_recalculation,
        is_generating_ds_info=args.generate_ds_info,
        is_force_regenerate_ds_info=args.force_ds_info_regeneration,
        is_generating_privacy_report=args.generate_privacy_report,
    )


def create_arg_parse_instance() -> ArgumentParser:
    parser = ArgumentParser(
        prog="Dataset Analysis for Privacy-Preserving-Machine-Learning",
        description=
        "A toolbox to analyse the influence of dataset characteristics on the performance of algorithm pertubation in PPML.",
    )

    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        type=str,
        help=
        "Which datasets to load before running the other steps. Multiple datasets can be specified, but at least one needs to be passed here. Available datasets are: mnist, fmnist, cifar10, cifar100, svhn, emnist-(large|medium|letters|digits|mnist)-(unbalanced|balanced). With modifications _cX (class size), _i[L/N]Y (imbalance), _nX (number of classes), _gray.",
        metavar="D",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["cnn", "private_cnn"],
        help=
        "Specify which model should be used for training/ attacking. Only one can be selected!",
        metavar="M",
    )
    parser.add_argument(
        "-r",
        "--run-number",
        type=int,
        help=
        "The run number to be used for training models, loading or saving results. This flag is theoretically not needed if you only want to generate ds-info results.",
        metavar="R",
    )
    parser.add_argument(
        "-n",
        "--run-name",
        type=str,
        help=
        "The run name to be used for training models, loading or saving results. This flag is theoretically not needed if you only want to generate ds-info results. The naming hierarchy here is: model_name/run_name/run_number.",
        metavar="N",
    )

    parser.add_argument(
        "-s",
        "--shadow-models",
        type=int,
        help="The number of shadow models to be trained.",
        metavar="S",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="The number of epochs the model should be trained on.",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        help="The learning rate used for training models.",
        metavar="L",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        help=
        "The desired epsilon value for DP-SGD learning. Can be any value: 0.1, 1, 10, ...",
        metavar="E",
    )
    parser.add_argument(
        "-ae",
        "--adam-epsilon",
        type=float,
        help="The epsilon hat value for the Adam optimizer.",
    )
    parser.add_argument("--batch-size",
                        type=int,
                        help="Size of batch used for training.")
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=float,
        help="The weight decay used in the Adam optimizer.",
    )
    parser.add_argument(
        "-ema",
        "--momentum",
        type=float,
        help=
        "Momentum value used for Adam's EMA when training the models. If set, EMA in Adam is activated.",
    )
    parser.add_argument(
        "-c",
        "--l2-norm-clip",
        type=float,
        help="The L2 norm clip value set for private training models.",
        metavar="C",
    )
    parser.add_argument(
        "-b",
        "--microbatches",
        type=int,
        help="Number of microbatches used for private training.",
        metavar="B",
    )
    parser.add_argument(
        "-np",
        "--noise-multiplier",
        type=float,
        help="A fix set noise multiplier for DP-SGD.",
    )

    parser.add_argument(
        "-tm",
        "--train-model",
        action="store_true",
        help=
        "If this flag is set, a single model is trained on the given datasets (respecting train_ds, val_ds & test_ds). This always overrides a previously trained model on the same dataset name and run number.",
    )
    parser.add_argument(
        "-em",
        "--evaluate-model",
        action="store_true",
        help=
        "If this flag is set, a single model is loaded based on run number, run name, model name and dataset name. Then predictions are run on the test and train dataset to evaluate the model.",
    )
    parser.add_argument(
        "-ce",
        "--compile-evaluation",
        help=
        "If this flag is set, the program compiles all single model evaluations from different run numbers to a single file.",
        action="store_true",
    )
    parser.add_argument(
        "--run-amia-attack",
        action="store_true",
        help=
        "If this flag is set, an Advanced MIA attack is run on the trained shadow models and the results are saved.",
    )
    parser.add_argument(
        "-ca",
        "--compile-attack-results",
        action="store_true",
        help=
        "If this flag is set, all saved attack results are compiled and compared with each other, allowing dataset comparison.",
    )
    parser.add_argument(
        "--force-model-retrain",
        action="store_true",
        help=
        "If this flag is set, the shadow models, even if they already exist.",
    )
    parser.add_argument(
        "--force-stat-recalculation",
        action="store_true",
        help=
        "If this flag is set, the statistics are recalucated on the shadow models.",
    )
    parser.add_argument(
        "--generate-ds-info",
        action="store_true",
        help="If this flag is set, dataset infos are generated and saved.",
    )
    parser.add_argument(
        "--force-ds-info-regeneration",
        action="store_true",
        help=
        "If this flag is set, the whole ds-info dict is not loaded from a json file but regenerated from scratch.",
    )
    parser.add_argument(
        "--include-mia",
        action="store_true",
        help=
        "If this flag is set, then the mia attack is also used during attacking and mia related results/ graphics are produced during result generation.",
    )
    parser.add_argument(
        "-p",
        "--generate-privacy-report",
        help=
        "Dont train/load anything, just generate a privacy report for the given values.",
        action="store_true",
    )
    return parser