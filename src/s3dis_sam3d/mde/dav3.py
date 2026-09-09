import torch
from depth_anything_3.api import DepthAnything3


class DA3Predictor:
    def __init__(
        self,
        *,
        device=None,
        local_files_only=False,
    ):
        self.device_name = device
        self.local_files_only = bool(
            local_files_only
        )
        self.model = None

    def _load(self):
        if self.model is not None:
            return


        if self.device_name is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            device = torch.device(
                self.device_name
            )

        print(
            f"Loading DA3 on {device}...",
            flush=True,
        )

        self.model = (
            DepthAnything3.from_pretrained(
                DA3_MODEL,
                revision=DA3_REVISION,
                local_files_only=(
                    self.local_files_only
                ),
            )
            .to(device)
            .eval()
        )

        loaded_revision = getattr(
            self.model,
            "_commit_hash",
            None,
        )

        if loaded_revision not in {
            None,
            DA3_REVISION,
        }:
            raise RuntimeError(
                "Unexpected DA3 revision: "
                f"{loaded_revision}, "
                f"expected {DA3_REVISION}"
            )

    def predict_raw(
        self,
        image_path: Path,
        target_shape=TARGET_SHAPE,
    ):
        """
        Return the raw DA3 metric-model prediction.

        Important:
        focal correction is deliberately NOT applied here.
        This preserves the old PriorBIMDA convention exactly.
        """

        self._load()

        height, width = target_shape

        result = self.model.inference(
            [str(image_path)],
            process_res=DA3_PROCESS_RES,
            export_dir=None,
        )

        depth = np.asarray(
            result.depth[0],
            dtype=np.float32,
        )

        if depth.shape != target_shape:
            depth = cv2.resize(
                depth,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        if (
            not np.isfinite(depth).all()
            or np.any(depth <= 0)
        ):
            raise ValueError(
                f"DA3 produced invalid depth: "
                f"{image_path}"
            )

        return depth