from __future__ import annotations

import warnings

import numpy as np


def patch_numpy_aliases() -> None:
    aliases = {
        "int": int,
        "float": float,
        "complex": complex,
        "bool": bool,
        "float_": np.float64,
        "int_": np.int64,
        "complex_": np.complex128,
        "bool_": np.bool_,
        "str_": np.str_,
    }
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def patch_reflection_for_bcr_kmer_run() -> None:
    from immuneML.util.ReflectionHandler import ReflectionHandler

    original_basic_names = ReflectionHandler.all_nonabstract_subclass_basic_names

    def limited_basic_names(cls, drop_part: str, subdirectory: str = ""):
        if subdirectory == "IO/dataset_import/":
            return ["AIRR"]
        if subdirectory == "encodings":
            return ["KmerFrequency", "DeepRC"]
        if subdirectory == "ml_methods/classifiers/":
            return ["LogisticRegression", "DeepRC"]
        if subdirectory in {
            "ml_methods/generative_models/",
            "ml_methods/clustering/",
            "ml_methods/dim_reduction/",
        }:
            return []
        return original_basic_names(cls, drop_part, subdirectory)

    ReflectionHandler.all_nonabstract_subclass_basic_names = staticmethod(limited_basic_names)


def patch_deeprc_metadata_ids() -> None:
    from immuneML.encodings.deeprc.DeepRCEncoder import DeepRCEncoder

    original_export_metadata_file = DeepRCEncoder.export_metadata_file

    def export_metadata_file_with_extensions(self, dataset, labels, output_folder):
        metadata_filepath = original_export_metadata_file(self, dataset, labels, output_folder)

        import pandas as pd

        metadata = pd.read_csv(metadata_filepath)
        metadata[DeepRCEncoder.ID_COLUMN] = metadata[DeepRCEncoder.ID_COLUMN].astype(str).map(
            lambda sample_id: sample_id
            if sample_id.endswith(f".{DeepRCEncoder.EXTENSION}")
            else f"{sample_id}.{DeepRCEncoder.EXTENSION}"
        )
        metadata.to_csv(metadata_filepath, sep=DeepRCEncoder.METADATA_SEP, index=False)
        return metadata_filepath

    DeepRCEncoder.export_metadata_file = export_metadata_file_with_extensions


def patch_deeprc_prediction_metadata_separator() -> None:
    from immuneML.data_model.EncodedData import EncodedData
    from immuneML.encodings.deeprc.DeepRCEncoder import DeepRCEncoder
    from immuneML.ml_methods.classifiers.DeepRC import DeepRC

    def predict_proba_with_metadata_separator(self, encoded_data: EncodedData):
        from deeprc.dataset_readers import RepertoireDataset as DeepRCRepDataset

        self.check_is_fitted(self.label.name)

        hdf5_filepath, _ = self._convert_dataset_to_hdf5(encoded_data, self.label)
        task_definition = self._make_task_definition(self.label)

        test_dataset = DeepRCRepDataset(
            metadata_filepath=encoded_data.info["metadata_filepath"],
            hdf5_filepath=str(hdf5_filepath),
            sample_id_column=DeepRCEncoder.ID_COLUMN,
            metadata_file_column_sep=DeepRCEncoder.METADATA_SEP,
            task_definition=task_definition,
            keep_in_ram=self.keep_dataset_in_ram,
            inputformat="NCL",
            sequence_counts_scaling_fn=self.sequence_counts_scaling_fn,
        )

        test_dataloader = self.make_data_loader(
            test_dataset,
            indices=range(len(test_dataset.target_features)),
            label_name=self.label.name,
            eval_only=True,
            is_train=False,
        )

        probs_pos_class = self._model_predict(self.model, test_dataloader)
        return {
            self.label.name: {
                self.label.positive_class: probs_pos_class,
                self.label.get_binary_negative_class(): 1 - probs_pos_class,
            }
        }

    DeepRC._predict_proba = predict_proba_with_metadata_separator


def patch_deeprc_model_predict_signature() -> None:
    from immuneML.ml_methods.classifiers.DeepRC import DeepRC

    def model_predict_with_current_deeprc_signature(self, model, dataloader):
        import torch
        from tqdm import tqdm

        with torch.no_grad():
            model.to(device=self.pytorch_device)
            scoring_predictions = []
            for scoring_data in tqdm(
                dataloader, total=len(dataloader), desc="Evaluating model", disable=True, position=1
            ):
                labels, inputs, sequence_lengths, counts_per_sequence, sample_ids = scoring_data
                labels, inputs, sequence_lengths, n_sequences = model.reduce_and_stack_minibatch(
                    labels, inputs, sequence_lengths, counts_per_sequence
                )
                logit_outputs = model(
                    inputs_flat=inputs,
                    sequence_lengths_flat=sequence_lengths,
                    n_sequences_per_bag=n_sequences,
                )
                scoring_predictions.append(torch.sigmoid(logit_outputs))

            scoring_predictions = torch.cat(scoring_predictions, dim=0).float().cpu().numpy()

        return scoring_predictions

    DeepRC._model_predict = model_predict_with_current_deeprc_signature


def patch_deeprc_serial_hdf5_conversion() -> None:
    import deeprc.dataset_converters as dataset_converters

    class SerialPool:
        def __init__(self, processes=None):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap(self, func, iterable):
            return map(func, iterable)

    dataset_converters.multiprocessing.Pool = SerialPool


def patch_template_parser_utf8() -> None:
    from immuneML.presentation.TemplateParser import TemplateParser

    def parse_with_utf8(template_path, template_map: dict, result_path):
        import pystache

        with template_path.open("r", encoding="utf-8") as template_file:
            template = template_file.read()

        rendered_template = pystache.render(template, template_map)

        with result_path.open("w", encoding="utf-8") as file:
            file.writelines(rendered_template)

        return result_path

    TemplateParser.parse = staticmethod(parse_with_utf8)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"immuneML\.encodings\.deeprc\..*")
    patch_numpy_aliases()
    patch_reflection_for_bcr_kmer_run()
    patch_deeprc_metadata_ids()
    patch_deeprc_prediction_metadata_separator()
    patch_deeprc_model_predict_signature()
    patch_deeprc_serial_hdf5_conversion()
    patch_template_parser_utf8()
    from immuneML.app.ImmuneMLApp import main as immuneml_main

    immuneml_main()


if __name__ == "__main__":
    main()
