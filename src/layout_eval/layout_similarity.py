from dataclasses import dataclass
import multiprocessing
from .measures.maximum_iou import compute_maximum_iou_for_layout_set
from .measures.mean_iou import compute_mean_iou_for_layout_pair
from .measures.docsim import compute_doc_sim_for_layout_pair
from .measures.docemd import compute_doc_emd_for_layout_pair
from .measures.ltsim import compute_lt_sim_for_layout_pair

@dataclass
class LayoutSimilarity:
    method: str = 'ltsim'
    kwargs: dict = None

    def __post_init__(self):
        # Set default value for kwargs if None
        self.kwargs = self.kwargs or {}
        self.method_function, self.input_type = self._get_method_function(self.method, **self.kwargs)

    def compare(self, layouts1, layouts2, disable_parallel=False, n_jobs=None) -> list[float]:
        # Automatically switch behavior based on input type
        if self.input_type == 'pair':
            if len(layouts1) != len(layouts2):
                raise ValueError("The size of layouts1 and layouts2 should be the same")
            args = list(zip(layouts1, layouts2))
        elif self.input_type == 'list_pair':
            args = [(layouts1, layouts2)]
        else:
            raise ValueError(f"Unknown input type: {self.input_type}")

        if disable_parallel or self.input_type == 'list_pair':
            # Sequential processing
            scores = [self.method_function(*arg) for arg in args]
        else:
            # Parallel processing
            with multiprocessing.Pool(n_jobs) as p:
                scores = p.starmap(self.method_function, args)
        return scores

    def set_method(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs or {}
        self.method_function, self.input_type = self._get_method_function(method, **self.kwargs)

    def _get_method_function(self, method, **kwargs):
        if method == 'lt_sim':
            return compute_lt_sim_for_layout_pair, 'pair'
        elif method == 'doc_sim':
            return compute_doc_sim_for_layout_pair, 'pair'
        elif method == 'maximum_iou':
            return compute_maximum_iou_for_layout_set, 'list_pair'
        elif method == 'mean_iou':
            return compute_mean_iou_for_layout_pair, 'pair'
        elif method == 'doc_emd':
            return compute_doc_emd_for_layout_pair, 'pair'
        else:
            raise ValueError(f"Unknown method: {method}")