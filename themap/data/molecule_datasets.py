from typing import Dict, List, Optional, Union

from dpu_utils.utils import RichPath

from themap.data.molecule_dataset import get_task_name_from_path
from themap.utils.logging import get_logger

logger = get_logger(__name__)


class DataFold:
    """Enum for data fold types."""

    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class MoleculeDatasets:
    """Dataset of related tasks, provided as individual files split into meta-train, meta-valid and
    meta-test sets."""

    def __init__(
        self,
        train_data_paths: List[RichPath] = [],
        valid_data_paths: List[RichPath] = [],
        test_data_paths: List[RichPath] = [],
        num_workers: Optional[int] = None,
    ) -> None:
        """Initialize MoleculeDatasets.

        Args:
            train_data_paths (List[RichPath]): List of paths to training data files.
            valid_data_paths (List[RichPath]): List of paths to validation data files.
            test_data_paths (List[RichPath]): List of paths to test data files.
            num_workers (Optional[int]): Number of workers for data loading.
        """
        logger.info("Initializing MoleculeDatasets")
        self._fold_to_data_paths: Dict[DataFold, List[RichPath]] = {
            DataFold.TRAIN: train_data_paths,
            DataFold.VALIDATION: valid_data_paths,
            DataFold.TEST: test_data_paths,
        }
        self._num_workers = num_workers
        logger.info(
            f"Initialized with {len(train_data_paths)} training, {len(valid_data_paths)} validation, and {len(test_data_paths)} test paths"
        )

    def __repr__(self) -> str:
        return f"MoleculeDatasets(train={len(self._fold_to_data_paths[DataFold.TRAIN])}, valid={len(self._fold_to_data_paths[DataFold.VALIDATION])}, test={len(self._fold_to_data_paths[DataFold.TEST])})"

    def get_num_fold_tasks(self, fold: DataFold) -> int:
        """Get number of tasks in a specific fold.

        Args:
            fold (DataFold): The fold to get number of tasks for.

        Returns:
            int: Number of tasks in the fold.
        """
        return len(self._fold_to_data_paths[fold])

    @staticmethod
    def from_directory(
        directory: Union[str, RichPath],
        task_list_file: Optional[Union[str, RichPath]] = None,
        **kwargs,
    ) -> "MoleculeDatasets":
        """Create MoleculeDatasets from a directory.

        Args:
            directory (Union[str, RichPath]): Directory containing train/valid/test subdirectories.
            task_list_file (Optional[Union[str, RichPath]]): File containing list of tasks to include.
            **kwargs (any): Additional arguments to pass to MoleculeDatasets constructor.

        Returns:
            MoleculeDatasets: Created dataset.
        """
        logger.info(f"Loading datasets from directory {directory}")
        if isinstance(directory, str):
            directory = RichPath.create(directory)
        else:
            directory = directory

        if task_list_file is not None:
            if isinstance(task_list_file, str):
                task_list_file = RichPath.create(task_list_file)
            logger.info(f"Using task list file: {task_list_file}")
            with open(task_list_file, "r") as f:
                task_list = [line.strip() for line in f.readlines()]
        else:
            task_list = None

        def get_fold_file_names(data_fold_name: str):
            fold_dir = directory.join(data_fold_name)
            if not fold_dir.exists():
                logger.warning(f"Directory {fold_dir} does not exist")
                return []
            return [
                f
                for f in fold_dir.iterate_filtered(glob_pattern="*.jsonl.gz")
                if task_list is None or get_task_name_from_path(f) in task_list
            ]

        train_data_paths = get_fold_file_names("train")
        valid_data_paths = get_fold_file_names("valid")
        test_data_paths = get_fold_file_names("test")

        logger.info(
            f"Found {len(train_data_paths)} training, {len(valid_data_paths)} validation, and {len(test_data_paths)} test tasks"
        )
        return MoleculeDatasets(
            train_data_paths=train_data_paths,
            valid_data_paths=valid_data_paths,
            test_data_paths=test_data_paths,
            **kwargs,
        )

    def get_task_names(self, data_fold: DataFold) -> List[str]:
        """Get list of task names in a specific fold.

        Args:
            data_fold (DataFold): The fold to get task names for.

        Returns:
            List[str]: List of task names in the fold.
        """
        return [get_task_name_from_path(path) for path in self._fold_to_data_paths[data_fold]]
