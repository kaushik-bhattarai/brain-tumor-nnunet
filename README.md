# 3D nnU-Net for Brain Tumour Segmentation (multimodal MRI)

End-to-end pipeline built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) (`3d_fullres`) that segments glioma
sub-regions from 4-channel brain MRI, with a leakage-free held-out test set and BraTS-style evaluation.

- **Data:** Medical Segmentation Decathlon Task01_BrainTumour (484 labelled BraTS-derived volumes: FLAIR, T1w, T1gd, T2w)
- **Model:** nnU-Net v2, 3D full-resolution U-Net, region-based training on WT / TC / ET
- **Split:** 10% of cases held out as a test set (seeded, case level); nnU-Net trains/validates on the remaining 90% (fold 0)
- **Evaluation:** 3D Dice and HD95 (mm, using real voxel spacing) for whole tumour (WT), tumour core (TC), enhancing tumour (ET)
- **Training:** Colab notebook included

## Results

Evaluated on 48 held-out patients never used for training or validation.

| Region | Dice (mean ± std) | HD95 mm (mean) |
|--------|-------------------|----------------|
| Whole tumour (WT) | 0.903 ± 0.083 | 7.9 |
| Tumour core (TC) | 0.842 ± 0.123 | 8.2 |
| Enhancing tumour (ET) | 0.759 ± 0.251 | 7.9 |

Test cases: N = 48 | Trainer: `nnUNetTrainer` (default 1000-epoch schedule), stopped at about epoch 100, best checkpoint | Fold: 0 | GPU: Colab

Median HD95 is 3.5 / 4.9 / 2.2 mm (WT / TC / ET); the means are pulled up by a few bad cases.

## Run it

### Option A: Google Colab
1. Put this folder in your Google Drive root.
2. Open `notebooks/colab_train.ipynb` 
3. Set `TRAINER = 'nnUNetTrainer'` , or `nnUNetTrainer_100epochs` is a shorter full schedule.

### Option B: local machine
```bash
pip install -r requirements.txt           
source scripts/env.sh                     
# get Task01_BrainTumour (http://medicaldecathlon.com), then:
python scripts/prepare_dataset.py --msd-dir /path/to/Task01_BrainTumour --dataset-id 501
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity
nnUNetv2_train 501 3d_fullres 0 --npz      # default trainer; the results above come from stopping at about epoch 100
nnUNetv2_predict -i $nnUNet_raw/Dataset501_BrainTumour/imagesTs -o preds_test -d 501 -c 3d_fullres -f 0 -chk checkpoint_best.pth
cd scripts && python evaluate.py --pred ../preds_test --gt $nnUNet_raw/Dataset501_BrainTumour/labelsTs \
    --images $nnUNet_raw/Dataset501_BrainTumour/imagesTs --out ../results
```

## Data, license and credits

MSD Task01_BrainTumour is released under CC-BY-SA 4.0 (check the terms at medicaldecathlon.com before redistributing anything derived from it).
Please cite nnU-Net (Isensee et al., Nature Methods 2021) and the Medical Segmentation Decathlon (Antonelli et al., Nature Communications 2022).