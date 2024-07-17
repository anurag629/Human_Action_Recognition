# Part-based GCN for Action Recognition

This project implements a Part-based Graph Convolutional Network (PB-GCN) for action recognition. The model is trained and evaluated on the UCF50 dataset. Three different CNN models are built and compared for performance.

## Project Structure

   ```
   part_based_gcn_action_recognition/
   ├── data/
   │   ├── raw/
   │   │   └── UCF50/
   │   ├── processed/
   │   └── loaders/
   │       └── ucf50_loader.py
   ├── docs/
   │   ├── api/
   │   └── usage/
   ├── notebooks/
   │   └── data_exploration.ipynb
   ├── src/
   │   ├── models/
   │   │   ├── __init__.py
   │   │   └── simple_cnn.py
   │   │   └── medium_cnn.py
   │   │   └── complex_cnn.py
   │   │   └── part_based_graph.py
   │   │   └── spatio_temporal_graph.py
   │   │   └──part_based_graph_convolutions.py
   │   ├── utils/
   │   │   ├── __init__.py
   │   │   ├── data_utils.py
   │   │   ├── training_utils.py
   │   │   └── logger.py
   │   ├── main.py
   │   └── config.py
   ├── tests/
   │   ├── test_data_loader.py
   │   ├── test_models.py
   │   └── test_utils.py
   ├── experiments/
   │   ├── experiment_1/
   │   │   ├── logs/
   │   │   ├── checkpoints/
   │   │   └── results/
   │   ├── experiment_2/
   │   │   ├── logs/
   │   │   ├── checkpoints/
   │   │   └── results/
   │   └── ...
   ├── configs/
   │   └── config.json
   ├── requirements.txt
   ├── README.md
   └── setup.py
   ```



## Setup

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd part_based_gcn_action_recognition
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the UCF50 dataset from [here](https://www.crcv.ucf.edu/data/UCF50.rar) and extract it to `data/raw/UCF50/`.

## Running the Project

### Encode Data

To encode the data, run:
   ```bash
   python scripts/encode_data.py
   ```


Before training the model, set the path:

   ```bash
   set PYTHONPATH=.
   ```

### Load Data and Train the Model

To train the model, run:
   ```bash
   python src/main.py train --config configs/config.json
   ```

### Test a Trained Model

   To test a trained model, run:
   ```bash
   python src/main.py test --config configs/config.json
   ```

   Real-Time Testing with Laptop Camera:
   ```bash
   python src/main.py realtime --config configs/config.json
   ```

   Testing with Uploaded Video:
   ```bash
   python src/main.py upload --config configs/config.json --video path/to/your/video.mp4
   ```

