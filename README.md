# Chess evaluation using machine learning model
 Model created by Demid Efremov as part of PMLDL course
 
 [d.efremov@innopolis.university](d.efremov@innopolis.university)

 The model aims to evaluate different chess positions using convolutional neural network
 
 The project is using Lichess positions dataset, and comes together with a [chess bot](https://lichess.org/@/AI_Chess_Bot) created on the same site. Check out bot's code in [this repository](https://github.com/Demid65/lichess-bot).

 ## Requirements
This projects uses pytorch as the machine learning network.

Requirements can be installed with:

```$ pip install -r requirements.txt```

 ## Usage
 1. Install the requirements
 2. Download the model [here](https://drive.google.com/file/d/1qQbfcsSICvmeQe_70bux0Mnvgd5L1J1D/view?usp=sharing), and unzip it into the project folder.
 3. Run the inference script with ```$ python evaluate.py```

 Command line parameters can be seen with
 
 ```$ python evaluate.py -h```

 ## Training
 1. Install the requirements
 2. Run the dataset downloader with ```$ python download_dataset.py```
 3. Run the training with ```$ python train.py```

 Use -h flag with any script to see configurable parameters.
 Check out the notebooks for extra info and code snippets.
 
 