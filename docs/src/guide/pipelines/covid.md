# COVID Image Classification

## Background
This pipeline is based on the Jupyter Notebook [COVID Image Classification](../notebooks/covid.md) but converted to a modular format to support kubeflow components. Initially the idea was to use [minio](../minio.md) as a storage mechanism for the data files but had to skip that effort to continue using the volumes available via the Kubeflow dashboard 

## Prerequisites

The prerequisites are still the same, we need the [kfp](https://pypi.org/project/kfp/) module to be present for compiling the python source into .yaml file for uploading to the Kubeflow cluster

We also require a module called [kfp-kubernetes](https://kfp-kubernetes.readthedocs.io/en/kfp-kubernetes-1.2.0/) to read any secrets / mount volumes from our Kubeflow Pipeline

Please use the below command to install the module - I used version 1.2.0

```
pip install kfp-kubernetes
```

## Pipeline Source

I had to rewrite the original notebook into multiple functions as shown below to support the multiple steps in the Kubeflow pipeline. Helps to productionize a notebook

Please find the sources [here](https://github.com/madhusudanabburu/pgp-aiml/blob/master/CV%20-%20COVID19%20Image%20Classification/covid_tensorflow.py)

### Load Data

```
@dsl.component(packages_to_install=['pandas', 'numpy', 'minio'])
def load_data(trainImages_url:str, trainLabel_url:str, trainImages:Output[Dataset], trainLabels:Output[Dataset]):
```
The above component loads the data files and just saves them to the volume - It is currently redundant but may be needed in the future to extend the loading of data from another pipeline or datasource

- Input Parameters : Images and labels to train the model
- Output Parameters : Images and labels to train the model 

### Pre Process Data

```
@dsl.component(packages_to_install=['pandas', 'numpy', 'opencv-python-headless', 'matplotlib', 'scikit-learn'])
def pre_process_data(trainImages: Dataset, trainLabels: Dataset, train_out: Output[Dataset], y_out: Output[Dataset]):
```
The above component helps in pre-processing the data by using the cv2 libraries to remove any blurring on the images to reduce the noise. It also converts the images to HSV (Better colors). I've chosen the gaussian blurred images as my choice as the HSV colored ones and masked ones were darker.

- Input Parameters : Images and labels to train the model
- Output Parameters : Preprocessed Images and labels to train the model

### Train

```
@dsl.component(packages_to_install=['pandas', 'numpy', 'opencv-python-headless', 'scikit-learn', 'seaborn', 'matplotlib', 'joblib', 'tensorflow==2.11.0', 'tensorflow-io==0.29.0'])
def train(new_train_in: Dataset, y_in: Dataset, cnn_model: Output[Model], X_test_out: Output[Dataset], y_test_out: Output[Dataset]):  
```

This component does the actual training of the model by splitting the input dataset into train and test parts. It builds a Sequential Convolutional Neural Network Classifier. The images are of 128x128 size and colored. I'm using three sets of Convolutional and Pooling layers 

This CNN classifier is compiled with **RMSprop optimizer** with **categorical_crossentropy** as loss function and **accuracy** as metrics to monitor 

The model is then fit with batch sizes of 5 and epoch sizes of 5. Batch size is the number of samples processed before the model is updated whereas epoch is the number of complete passes through the training dataset. I had a very low number for both the batch and epoch as there were memory issues with the pipeline 

- Input Parameters : Preprocessed Images and Labels to train the model
- Output Parameters : CNN Model and test set output

### Train with Learning Rate Reduction

```
@dsl.component(packages_to_install=['pandas', 'numpy', 'opencv-python-headless', 'scikit-learn', 'seaborn', 'matplotlib', 'joblib', 'tensorflow==2.11.0', 'tensorflow-io==0.29.0'])
def train_lr_reduction(new_train_in: Dataset, y_in: Dataset, cnn_lr_reduction_model: Output[Model], X_test_out: Output[Dataset], y_test_out: Output[Dataset]):  
```
This component also does the actual training of the model by splitting the input dataset into train and test parts. It builds a Sequential Convolutional Neural Network Classifier. The images are of 128x128 size and colored. I'm using three sets of Convolutional and Pooling layers 

The only difference with the [Train](#train) component is this uses the **ReduceLROnPlateau** method as a callback while fitting the model and uses the **val_accuracy** to compare the validation data for accuracy

This classifier also uses **RMSprop Optimizer** with **categorical_crossentropy** as loss function and **accuracy** as metrics to monitor

- Input Parameters : Preprocessed Images and Labels to train the model
- Output Parameters : CNN Model with Learning Rate Reduction technique and test set output

### Compare CNN Models

```
@dsl.component(packages_to_install=['pandas', 'numpy', 'opencv-python-headless', 'scikit-learn', 'seaborn', 'matplotlib', 'joblib', 'tensorflow==2.11.0', 'tensorflow-io==0.29.0'])
def compare_cnn_models(cnn_model: Input[Model], cnn_lr_reduction_model: Input[Model], X_test_in: Input[Dataset], y_test_in: Input[Dataset]): 
```

This component is used to compare the models that were created earlier viz [Train](#train) and [Train with LearningRate Reduction](#train-with-learning-rate-reduction). All it does is to evaluate the test loss and accuracy and also predict the values and compare them with the test labels. It also prints the confusion matrix

- Input Parameters : Input models, Test values
- Output Parameters : None

## Pipeline Compilation

Use the below command to compile the pipeline into a target format like .yaml to upload to the Kubeflow Dashboard

```
kfp dsl compile --py covid_tensorflow.py --output covid_tensorflow.yaml
```

![Image from images folder](~@source/images/pipelines/covid/kfp_covid_pipeline_compile.png)

Let's upload the generated yaml file on our Kubeflow cluster

## Upload Pipeline to Kubeflow

Let's go to the Kubeflow dashboard on our localhost and try to click on the Pipelines link on the left navigation bar and upload our pipeline file - I've created another version of the pipeline here as shown in the screenshot

![Image from images folder](~@source/images/pipelines/covid/kfp_covid_yaml_upload.png)

Uploading the yaml file should show a graph view of the pipeline with its tasks as shown below

![Image from images folder](~@source/images/pipelines/covid/kfp_covid_pipeline_graph.png)

As you can see, there are several methods/functions (in pipeline terminology, they are called as components) that are shown in the graph with the sequence of operations and the dependencies

## Run the Pipeline

Click the **Create Run** shown on the top right of the Dashboard to create a run for this pipeline. It should show the following screen

![Image from images folder](~@source/images/pipelines/covid/kfp_covid_pipeline_run.png)

Let's click on Start at the bottom of the screen and see what happens (this pipeline ran for approx 30 minutes as the data was minimal but the CNN model was taking time)

![Image from images folder](~@source/images/pipelines/covid/kfp_covid_pipeline_output.png)

I had to collect the prediction from logs of the **compare_cnn_models** step

![Image from images folder](~@source/images/pipelines/covid/kfp_covid_pipeline_model_prediction.png)
![Image from images folder](~@source/images/pipelines/covid/kfp_covid_pipeline_tuned_model_prediction.png)

As you can see above, the tuned model has done a good job and has made 4 correct predictions out of 5.

## Observations

This pipeline does perform the basic functionalities like load data, pre processing it, training and validating the models. One important consideration that I would need to make is to understand how to add metadata to keras models 

Next step would be download this tuned model and upload it to Google Vertex AI and expose endpoints to see if it can be used for prediction using a sample Web Application


<PageMeta />