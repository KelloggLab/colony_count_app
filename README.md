# colony_count_app

A lightweight web application for automated microbial colony counting.

🚀 Overview

The colony_count_app streamlines microbiology workflows by automating colony counting from plate images. It provides:

* A simple browser-based interface (app.py)

* Automated plate image processing and colony detection (colony_count.py)

* Optional model training for training sets, which requires plate images and manually picked points (i.e. positive labeled features) (train_all.py)

* Tools to test and refine detection accuracy

This reduces manual counting time, minimizes user bias, and increases experimental throughput.

🔧 Features

* Upload a plate image and receive an instant colony count

* Visualization of detected colonies

* Exportable results

* Customizable training workflow for atypical plate images

* Minimal dependencies for easy installation in lab or HPC environments

🛠️ Installation

Clone the repository:

```
git clone https://github.com/KelloggLab/colony_count_app.git
cd colony_count_app
```

Create the conda environment:

```
conda env create -f environment.yml
conda activate colony_count_app
```

Launch the app:

```
streamlit run app.py
```

🎯 Usage

Launch the app and upload an image (JPG/PNG).

The system detects colonies and reports a count.

View the annotated output showing colony detections.

Download or record results.

(Optional) Train the model using your labels and the training_set/ directory for improved accuracy on non-standard plates.

📁 Repository Structure
/
├── app.py                 # Web application entry point
├── colony_count.py        # Core colony detection logic
├── guiHelper.py           # Helper functions for UI and processing
├── train_all.py           # Model training script for batches of images
├── patch_gallery_demo.py  # Test your feature extraction patch size
├── training_set/          # Training images and labels
├── example.jpg/.png       # Example images for quick testing
├── environment.yml        # Conda environment file
└── README.md              # This README

✅ Requirements & Compatibility

* Python 3.8+

* macOS or Linux recommended (Windows untested)

* Standard modern browser (Chrome, Firefox)

* Best performance with high-contrast plate images

🚧 Current Limitations

* Dense/overlapping colonies may affect accuracy

* Assumes approximately circular plates

* Lighting variation may require retraining or preprocessing

Planned improvements:

* Development of Pre-trained models. 

👥 Contributing

Submit issues for bugs or feature requests

Use pull requests for code changes

Update environment.yml if dependencies change

📄 License

This project is released under the MIT License.

📬 Contact

For questions or collaboration, please contact ekellogg: Elizabeth.Kellogg@stjude.org

