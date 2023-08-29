# Single-Channel RF Challenge Starter Code

[Click here for details on the challenge setup](https://rfchallenge.mit.edu/wp-content/uploads/2021/08/Challenge1_pdf_detailed_description.pdf)

For those eager to dive in, we have prepared a concise guide to get you started.

Check out [notebook/RFC_QuickStart_Guide.ipynb](https://github.com/RFChallenge/rfchallenge_singlechannel_starter_grandchallenge2023/blob/0.2.0/notebook/RFC_QuickStart_Guide.ipynb) for practical code snippets. You will find steps to create a small but representative training set and steps for inference to generate your submission outputs.
For a broader understanding and other helpful resources in the starter kit integral to the competition, please see the details and references provided below.

## TestSet for Evaluation

This starter kit equips you with essential resources to develop signal separation and interference rejection solutions. In this competition, the crux of the evaluation hinges on your ability to handle provided signal mixtures. Your task will be twofold:

1.  Estimate the Signal of Interest (SOI) component within the two-component mixture.
    
2.  Deduce the best possible estimate of the underlying information bits encapsulated in the SOI.

Delve into the specifics below for comprehensive details.

### TestSet1:

[Click here for TestSet1A files](https://www.dropbox.com/scl/fi/d3wynqylml9mxctvt72eu/TestSet1A.zip?rlkey=izbum6lvw7hdz575lpetjdzf3&dl=0)

50 frames of each interference type have been reserved to form TestSet1. These will be released alongside the main dataset, and the mixtures from TestSet1A are generated from this collection. Please note that although TestSet1 is available for examination, the final evaluation for participants will be based on a hidden, unreleased set (TestSet2).

#### File Descriptions:

***`TestSet1A_testmixture_[SOI Type]_[Interference Type].npy`:*** This is a numpy array of 1,100 x 40,960 np.complex64 floats; each row represents a mixture signal (40,960 complex values); the mixture signals are organized in increasing SINR, spanning11 SINR levels with 100 mixtures per SINR level.

***`TestSet1A_testmixture_[SOI Type]_[Interference Type]_metadata.npy`:*** This is a numpy array of 1,100 x 5 objects containing metadata information. The first column is the scalar value (k) by which the interference component is scaled; the second column is the corresponding target SINR level in dB (calculated as $10 \log_{10}(k)$), the third column is the actual SINR computed based on the components present in the mixture (represented as $10 \log_{10}(P_{SOI}/P_{Interference}))$, and the fourth and fifth column are strings denoting the SOI type and interference type respectively

### TestSet2:
***Expected November 25, 2023 (Not yet released)***

50 frames of each interference type have been designated for TestSet2. Please note that this set will not be made available during the competition.

The format for test mixtures in TestSet2A will be consistent with that of TestSet1A. However, any changes or modifications to the format will be communicated to the participants as the competition progresses.

 
### Submission Specifications:

For every configuration defined by a specific SOI Type and Interference Type, participants are required to provide:

1.  SOI Component Estimate:
-   A numpy array of dimensions 1,100 x 40,960.
-   This should contain complex values representing the estimated SOI component present.
-   Filename: `[ID String]_[TestSet Identifier]_estimated_soi_[SOI Type]_[Interference Type].npy`
    (where ID String will be a unique identifier, e.g., your team name)
    
2.  Information Bits Estimate:
-   A numpy array of dimensions 1,100 x B.    
-   The value of B depends on the SOI type:
    -   B = 5,120 for QPSK SOI
    -   B = 57,344 for OFDMQPSK SOI
-   The array should exclusively contain values of 1’s and 0’s, corresponding to the estimated information bits carried by the SOI.
-   Filename: `[ID String]_[TestSet Identifier]_estimated_bits_[SOI Type]_[Interference Type].npy`
    (where ID String will be a unique identifier, e.g., your team name)
    
For guidance on mapping the SOI signal to the information bits, participants are advised to consult the provided demodulation helper functions (e.g., as used in [notebook/RFC_Demo.ipynb](https://github.com/RFChallenge/rfchallenge_singlechannel_starter_grandchallenge2023/blob/0.2.0/notebook/RFC_EvalSet_Demo.ipynb)).


## Starter Code Setup:
Relevant bash commands to set up the starter code:
```bash
git clone https://github.com/RFChallenge/rfchallenge_singlechannel_starter_grandchallenge2023.git rfchallenge
cd rfchallenge

# To obtain the dataset
wget -O  dataset.zip "https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0"
unzip  dataset.zip 
rm dataset.zip 

# To obtain TestSet1A
wget -O  TestSet1A.zip  "https://www.dropbox.com/scl/fi/d3wynqylml9mxctvt72eu/TestSet1A.zip?rlkey=izbum6lvw7hdz575lpetjdzf3&dl=0"
unzip TestSet1A.zip -d dataset
rm TestSet1A.zip
```

Dependencies: The organizers have used the following libraries to generate the signal mixtures and train the relevant baseline models
* python==3.7.13
* numpy==1.21.6
* tensorflow==2.8.2
* sionna==0.10.0
* tqdm==4.64.0
* h5py==3.7.0

For a complete overview of the dependencies within our Anaconda environment, please refer [here (rfsionna)](https://github.com/RFChallenge/rfchallenge_singlechannel_starter_grandchallenge2023/blob/0.2.0/rfsionna_env.yml). Additionally, if you're interested in the Torch-based baseline, you can find the respective Anaconda environment dependencies that the organizers used [here (rftorch)](https://github.com/RFChallenge/rfchallenge_singlechannel_starter_grandchallenge2023/blob/0.2.0/rftorch_env.yml).

Since participants are tasked with running their own inference, we are currently not imposing restrictions on the libraries for training and inference. However, the submissions are expected to be in the form numpy arrays (`.npy` files) that are compatible with our system (`numpy==1.21.6`).

> Note: Diverging from the versions of the dependencies listed above might result in varied behaviors of the starter code. Participants are advised to check for version compatibility in their implementations and solutions.


## Helper Functions for Testing:

To assist participants during testing, we provide several example scripts designed to create and test with evaluation sets analogous to TestSet1A.

`python sampletest_testmixture_generator.py [SOI Type] [Interference Type]`

This script generates a new evaluation set (default name: SampleEvalSetA) based on the raw interference dataset of TestSet1. Participants can employ this for cross-checking. The produced outputs include a mixture numpy array, a metadata numpy array (similar to what's given in TestSet1A), and a ground truth file.

(Example generated SampleEvalSetA can be found [here](https://drive.google.com/file/d/1trKDjQ2QmIj8jOa3xAObyeURbcZsN2LR/view?usp=drive_link).)


`python sampletest_tf_unet_inference.py [SOI Type] [Interference Type] [TestSet Identifier]`

`python sampletest_torch_wavenet_inference.py [SOI Type] [Interference Type] [TestSet Identifier]`

(Default: Use SampleEvalSetA for [TestSet Identifier])
Scripts that leverage the supplied baseline methods (Modified U-Net on Tensorflow or WaveNet on PyTorch) for inference.

`python sampletest_evaluationscript.py [SOI Type] [Interference Type] [TestSet Identifier] [Method ID String]`

[Method ID String] is your submission's unique identifier---refer to submission specifications.
Utilize this script to assess the outputs generated from the inference script.

 
## Helper Functions for Training:

For a grasp of the basic functionalities concerning the communication signals (the SOI) and code snippets relating to how we load and extract interference signal windows to create signal mixtures, participants are referred to the RFC_Demo.ipynb in our starter code.

We also provide some reference codes used by the organizers to train the baseline methods. These files include:

1.  Training Dataset Scripts: Used for creating an extensive training set. The shell script file with the relevant commands is included: sampletrain_gendataset_script.sh. Participants can refer to and modify (comment/uncomment) the relevant commands in the shell script. The corresponding python files used can be found in the `dataset_utils` directory and include:
    -   `example_generate_rfc_mixtures.py`: Creates 240,000 sample mixtures with varying random target SINR levels (ranging between -33 dB and 3 dB). The output targets 60 H5DF files, each containing 4,000 mixtures.    
    -   `tfds_scripts/Dataset_[SOI Type]_[Interference Type]_Mixture.py`: Used in conjunction with the Tensorflow UNet training scripts; the H5DF files are processed into Tensorflow Datasets (TFDS) for training.
    -  ` example_preprocess_npy_dataset.py`: Used in conjunction with the Torch WaveNet training scripts; the H5DF files are processed into separate npy files (one file per mixture). An associated dataloader is supplied within the PyTorch baseline code.
    -   `example_generate_competition_trainmixture.py`: Another python script for generating example mixtures for training; this script creates a training set that is more aligned with the TestSet’s specifications (e.g., focusing solely on the 11 discrete target SINR levels).
    

2.  Model Training Scripts: The competition organizers have curated two implementations:
    -   UNet on Tensoflow: `train_unet_model.py`, accompanied with neural network specification in `src/unet_model.py`
    -   WaveNet on Torch: `train_torchwavenet.py`, accompanied with dependencies including `supervised_config.yml` and `src/configs`, `src/torchdataset.py`, `src/learner_torchwavenet.py`, `src/config_torchwavenet.py` and `src/torchwavenet.py`  

While the provided scripts serve as a starting point, participants have no obligations to utilize them. These files are provided as references to aid those wishing to expand upon or employ the baseline methods.

Trained model weights for the UNet and WaveNet can be obtained here: [reference_modes.zip](https://www.dropbox.com/scl/fi/890vztq67krephwyr0whb/reference_models.zip?rlkey=6yct3w8rx183f0l3ok2my6rej&dl=0).

Relevant bash commands:
```bash
wget -O  reference_models.zip "https://www.dropbox.com/scl/fi/890vztq67krephwyr0whb/reference_models.zip?rlkey=6yct3w8rx183f0l3ok2my6rej&dl=0"
unzip  reference_models.zip 
rm reference_models.zip 
```

---
## Available Support Channels:
*(For the Grand Challenge: September to December 2023)*

As you embark on this challenge, we would like to offer avenues for assistance.
Below are several channels through which you can reach out to us for help. Our commitment is to foster an environment that aids understanding and collaboration. Your questions, feedback, and concerns are instrumental in ensuring a seamless competition.
* Discord: [To be provided]
    
* Github (under the Issues tab): https://github.com/RFChallenge/rfchallenge_singlechannel_starter_grandchallenge2023/issues
    
* Email: rfchallenge@mit.edu
    >Note: Please be aware that the organizers reserve the right to publicly share email exchanges on any of the above channels. This is done to promote information dissemination and provide clarifications to commonly asked questions.

While we endeavor to offer robust support and timely communication, please understand that our assistance is provided on a "best-effort" basis. We are committed to addressing as many queries and issues as possible, but we may not have solutions to all problems.

Participants are encouraged to utilize the provided channels and collaborate with peers. By participating, you acknowledge and agree that the organizers are not responsible for resolving all issues or ensuring uninterrupted functionality of any tools or platforms. Your understanding and patience are greatly appreciated.

---
### Acknowledgements
The efforts of the organizers are supported by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government.

The organizers acknowledge the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing HPC resources that have contributed to the development of this work.
