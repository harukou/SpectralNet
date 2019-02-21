**To** use SpectralNet on MNIST, Reuters, the nested 'C' dataset (as seen above), or the semi-supervised and noisy nested 'C' dataset, please run

```
cd path_to_spectralnet/src/applications; 
python run.py --gpu=gpu_num --dset=mnist|reuters|cc|cc_semisup
```
**To** use SpectralNet on a new dataset: 
open `SpectralNet/tree/master/src/applications` and run `script.py`, and you will see the 2 centrical circle clustering results.

If you want to design and change your 2-d dataset, please turn to the `SpectralNet/tree/master/src/new_dset` and write the code like `concentric2.py`