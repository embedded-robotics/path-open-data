# path-open-data
This repository will deal with the code for assessment of the Open-Ended Pathology VQA

For Data Augmentation

1. git clone git@github.com:DIAGNijmegen/pathology-he-autoaugmetation.git or git clone git@github.com:DIAGNijmegen/pathology-he-auto-augment.git
2. pip install scikit-image
3. pip install albumentations==1.1.0
4. Change the following lines in pathology-he-autoaugmetation/randaugment/train/randaugement_new_ranges.py
_REPLACE = 225 -> This will replace the background to a semi-white color to match the slides background
available_ops = ['Scaling','TranslateX', 'TranslateY','ShearX', 'ShearY','Brightness', 'Sharpness','Color', 'Contrast','Rotate','Equalize', 'HsvH','HsvS','HsvV','HedH','HedE','HedD'] -> We are not doing Elastic, Gaussian Blur or Gaussian Noise operation since that tends to disturb the data to an extent which is not identifiable

