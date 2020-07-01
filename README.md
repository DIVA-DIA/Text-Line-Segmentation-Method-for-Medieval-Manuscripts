**&rarr; Link to paper: [https://arxiv.org/abs/1906.11894](https://arxiv.org/abs/1906.11894)**

# Text Line Segmentation Method for Medieval Manuscripts
Image and Text Segmentation pipeline for the paper ["Labeling, Cutting, Grouping: an Efficient Text Line Segmentation Method for Medieval Manuscripts"](https://arxiv.org/abs/1906.11894), published at the 15th IAPR International Conference on Document Analysis and Recognition (ICDAR) in 2019.

## Getting started

In order to get the pipeline up and running it is only necessary to clone the latest version of the repository:

``` shell
git clone https://github.com/DIVA-DIA/Text-Line-Segmentation-Method-for-Medieval-Manuscripts
```

Run the install the conda environment:

``` shell
conda env create -f environment.yml
```

and activate the environment:
```
conda actiavte image_text_segmentation
```

## First run
To see if the code works properly you can call the algorithm from the root folder with the following command
```
python python src/line_segmentation/line_segmentation.py
```
This will run the code on the image *test1.png* from the folder ```src/data/```.

You can change the input image, output folder and som other parameters for a single run with this input parameters
```
optional arguments:
  -h, --help            show this help message and exit
  --input-path INPUT_PATH
                        Path to the input file
  --output-path OUTPUT_PATH
                        Path to the output folder
  --seam-every-x-pxl SEAM_EVERY_X_PXL
                        After how many pixel a new seam should be casted
  --penalty_reduction PENALTY_REDUCTION
                        Punishment reduction for the seam leaving the y axis
  --testing TESTING     Are you running on a testing file provided bz us?
  --console_log CONSOLE_LOG
                        Console logging
  --vertical VERTICAL   Is the text orientation vertical?
```

## Running on own images
If you want to run the algorithm on need to segment the images beforehand.
This segmented image is then the input for the algorithm. The format of the image *needs* to be as described [here](#Ground-Truth-Format)

Then you can either run it on a single file as described [above](#First-run) or you can run it on a folder of files including evaluation of the algorithm.
```
python src/line_segmentation/evaluation/evaluate_algorithm.py
```

Which has the following parameters:
```
optional arguments:
  -h, --help            show this help message and exit
  --input-folders-pxl INPUT_FOLDERS_PXL [INPUT_FOLDERS_PXL ...]
                        path to folders containing pixel-gt (e.g.
                        /dataset/CB55/output-m /dataset/CSG18/output-m
                        /dataset/CSG863/output-m)
  --gt-folders-xml GT_FOLDERS_XML [GT_FOLDERS_XML ...]
                        path to folders containing xml-gt (e.g.
                        /dataset/CB55/test-page /dataset/CSG18/test-page
                        /dataset/CSG863/test-page)
  --gt-folders-pxl GT_FOLDERS_PXL [GT_FOLDERS_PXL ...]
                        path to folders containing xml-gt (e.g.
                        /dataset/CB55/test-m /dataset/CSG18/test-m
                        /dataset/CSG863/test-m)
  --output-path DIR     path to store output files
  --penalty-reduction PENALTY_REDUCTION
                        path to store output files
  --small-component-ratio SMALL_COMPONENT_RATIO
                        The percentage a connected component needs to be
                        considered. If he is smaller then x * avg_area it will
                        get deleted from the list of CC
  --seam-every-x-pxl SEAM_EVERY_X_PXL
                        how many pixels between the seams
  --vertical            assume text has vertical orientation (e.g. Chinese
  --eval-tool DIR       path to folder containing
                        DIVA_Line_Segmentation_Evaluator
  -j J                  number of thread to use for parallel search. If set to
                        0 #cores will be used instead
```

## Ground Truth Format
The ground truth format is described [here](https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator#ground-truth-format)

## Citing us

If you use our software, please cite our paper as:

``` latex
@inproceedings{alberti2019linesegmentaton,
    address = {Sydney, Australia},
    author = {Alberti, Michele and Voegtlin, Lars and  Pondenkandath, Vinaychandran and Seuret, Mathias and Ingold, Rolf and Liwicki, Marcus},
    title = {{Labeling, Cutting, Grouping: an Efficient Text Line Segmentation Method for Medieval Manuscripts}},
    booktitle = {2019 15th IAPR International Conference on Document Analysis and Recognition (ICDAR)},
    year = {2019},
    month = {sep},
}
```

## License

Our work is on GNU Lesser General Public License v3.0
