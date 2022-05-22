import os
os.system('wget -N http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip')

import zipfile
with zipfile.ZipFile('./DIV2K_train_HR.zip', 'r') as zip_ref:
    zip_ref.extractall('./')