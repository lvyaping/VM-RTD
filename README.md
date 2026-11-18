VM-RTD

Relation and Transfer Distillation for Visual Mamba in Medical Image Segmentation

VM-RTD transfers spatial knowledge from a Swin Transformer teacher to a VM-UNet student through two complementary forms of supervision.

Scan branch relation distillation matches local feature relations between the teacher and each of the four scan branches at encoder stages 3 and 4.

Internal scan transfer supervision combines source write-in, intervening state transitions, and target readout to construct transfer distributions at encoder stage 3. Teacher feature relations guide these distributions over the same candidate positions.

Both objectives are optimized jointly with the segmentation loss. Inference uses the student network alone.

Experiments

The framework is evaluated on ISIC2017, ISIC2018, BUSI, and BUS using Dice and IoU. The repository includes experiments for the baseline student, individual supervision components, output relation and endpoint controls, and the full method.

Repository Structure

vmdmd/ — Models, losses, data loading, training, and evaluation.

configs/ — Dataset configurations.

scripts/ — Experiment launchers, evaluation, visualization, and efficiency measurement.

tests/ — Implementation checks.

weights/ — Pretrained student weights.

licenses/ — Third-party licenses.

Usage

Install dependencies from requirements.txt and configure dataset and checkpoint paths before running the scripts. The full method uses scripts/train_B_scan_transfer.sh, which retains its original experiment naming.
