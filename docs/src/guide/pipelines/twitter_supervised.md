# Twitter Supervised Learning

## Background

After trying the initial [Hello World Pipeline](./README.md), I came across a tool called [Kale](https://github.com/kubeflow-kale/kale) that seamlessly integrates with Jupyter Notebook as an extension and automatically converts the notebook code to a Kubeflow pipeline but unfortunately I couldn't install it due to the Jupyter Notebook version not being supported, I had to downgrade the version and also rebuild the docker images for Jupyter Notebook to only support versions between 2.0.0 and 3.0.0 

There was another tool called [Fairing](https://github.com/kubeflow/fairing) that looks promising to productionize the Jupyter Notebook files by making them run on Kubeflow as pipelines made up of individual components that can scale as needed

I prefer less tools in my ecosystem for complexity reasons so deferred the [Fairing](https://github.com/kubeflow/fairing) approach and explored on rewriting the Jupyter Notebook into modularized components

## Prerequisites

The prerequisites are still the same, we need the [kfp](https://pypi.org/project/kfp/) module to be present for compiling the python source into .yaml file for uploading to the Kubeflow cluster

We also require a module called [kfp-kubernetes](https://kfp-kubernetes.readthedocs.io/en/kfp-kubernetes-1.2.0/) to read any secrets / mount volumes from our Kubeflow Pipeline

Please use the below command to install the module - I used version 1.2.0

```
pip install kfp-kubernetes
```

## Pipeline Source

I had to come up with a modular version of the [Twitter Sentiment Analysis](../notebooks/twitter.md) as shown below to represent some of the functions as components. One of the most important things to consider is to design our notebook into multiple functions that can later become components on Kubeflow cluster and also minimizing the effort to productionize a notebook

### Read CSV

```
@dsl.component(packages_to_install=['pandas', 'numpy'])
def read_csv(csv_url:str, data:Output[Dataset]):
```
The above component reads the actual csv file (preset with all data) and loads into a shared volume as a Dataset. If you notice, the **dsl decorators @dsl.component** is used to signify to **kfp dsl** that this is a component and will be executed individually on the cluster. We should also be providing all the python modules that this function/component needs so that its installed before the launch of this component

- Input Parameters : CSV file url
- Output Parameters : Dataset (converted to csv)

### Exploratory Data Analysis

```
@dsl.component(packages_to_install=['pandas', 'numpy', 'nltk', 'wordcloud', 'fsspec'])
def exploratory_data_analysis(dataset: Input[Dataset], dataset_out: Output[Dataset]) -> Dict:
```
The above component takes the input from the first component (read_csv) and performs an exploratory data analysis. This method is optional as data would have been analyzed during the notebook development phase but if the data is ever flowing into this component, it may perform the analysis and log the results to another persistent data store for further action

- Input Parameters : Dataset
- Output Parameters : Analyzed and cleansed Dataset and a Metrics Dictionary as a return variable

### Pre Process Data

```
@dsl.component(packages_to_install=['pandas', 'numpy', 'nltk', 'wordcloud', 'contractions', 'beautifulsoup4'])
def pre_process_data(dataset: Dataset, dataset_out: Output[Dataset]):
```
The preprocess component strips out any html content, removes numbers and basically does tokenization and lemmatization (reducing the different forms of a word to one single form). It also does all other cleansing activities like removing punctuations, removing stop words and non-ascii characters etc. Notice the packages to be installed to perform these functionalities

- Input Parameters : Dataset
- Output Parameters :  Tokenized, Cleaned and Lemmatized Dataset

### Bag of words (Count Vectorizer) - Train Model 1

```
@dsl.component(packages_to_install=['pandas', 'numpy', 'nltk', 'wordcloud', 'scikit-learn', 'seaborn', 'matplotlib', 'joblib'])
def bag_of_words(dataset: Dataset, clf_bow_model: Output[Model], ytest_bow: Output[Dataset], count_vectorizer_pred: Output[Dataset]):
```
The bag of words component basically collects the different words created in the previous step, CountVectorizer is used to identify the features as a first step and then splits the data into train and test and uses K-Fold cross validation to build a cross validation score. The optimal learners parameter is chosen based on the minimum error and then used in the RandomForestClassifier to build a model 

This component also calculates the top features that displays a generated wordcloud image 

### Term Frequency - Inverse Document Frequency (TF-IDF) - Train Model 2

```
@dsl.component(packages_to_install=['pandas', 'numpy', 'nltk', 'wordcloud', 'scikit-learn', 'seaborn', 'matplotlib', 'joblib'])
def tf_idf(dataset: Dataset, clf_tf_model: Output[Model], ytest_tf: Output[Dataset], tf_idf_pred: Output[Dataset]):
```
This function calculates the frequency of words appearing in a document/collection of words. TfidfVectorizer is used to identify the features as a first step. As a second step the data is split into train and test and uses K-Fold cross validation to build a cross validation score. The optimal learners parameter is chosen based on the minimum error and then used in the RandomForestClassifier to build a model. 

### Compare Supervised Learnings - Bag of Words and Term Frequency - Inverse Document Frequency

```
@dsl.component(packages_to_install=['pandas', 'numpy', 'nltk', 'wordcloud', 'scikit-learn', 'seaborn', 'matplotlib'])
def compare_supervised_learning(y_test_input: Input[Dataset], count_vectorizer_predicted_input: Input[Dataset], tf_idf_predicted_input: Input[Dataset], bow_score: OutputPath(str), tf_score: OutputPath(str), chosen_model: OutputPath(str)):
```
This function uses the models that are generated by the previous components [Bag of Words](#bag-of-words-count-vectorizer---train-model-1) and [Term Frequency - Inverse Document Frequency](#term-frequency---inverse-document-frequency-tf-idf---train-model-2) and compares the accuracy score and finalizes which one to use for better results

## Pipeline Compilation

Use the below command to compile the pipeline into a target format like .yaml to upload to the Kubeflow Dashboard

```
kfp dsl compile --py twitter_nlp_supervised.py --output twitter_nlp_supervised.yaml
```

![Image from images folder](~@source/images/pipelines/twitter/kfp_twitter_pipeline_compile.png)

Let's upload the generated yaml file on our Kubeflow cluster

## Upload Pipeline to Kubeflow

Let's go to the Kubeflow dashboard on our localhost and try to click on the **Pipelines** link on the left navigation bar and upload our pipeline file - I've created another version of the pipeline here as shown in the screenshot as there were several attempts to make it run successfully :)

![Image from images folder](~@source/images/pipelines/twitter/kfp_twitter_yaml_upload.png)

Uploading the yaml file should show a graph view of the pipeline with its tasks as shown below

![Image from images folder](~@source/images/pipelines/twitter/kfp_twitter_pipeline_graph.png)

As you can see, there are several methods/functions (in pipeline terminology, they are called as components) that are shown in the graph with the sequence of operations and the dependencies 

## Run the Pipeline

Click on **Create Run** shown on the top right of the Dashboard to create a run for this pipeline. It should show the following screen

![Image from images folder](~@source/images/pipelines/twitter/kfp_twitter_pipeline_run.png)

Let's click on Start at the bottom of the screen and see what happens (this pipeline ran for approx 5 minutes as the data was minimal)

![Image from images folder](~@source/images/pipelines/twitter/kfp_twitter_pipeline_output.png)

As you can see from the above image, our pipeline has run successfully and also has print the output that says TF-IDF model has a better accuracy score compared to the Bag of Words model though the difference is very minimal. This model can be persisted using [pickle](https://docs.python.org/3/library/pickle.html) or [joblib](https://joblib.readthedocs.io/en/stable/) libraries so it can be used for predictions 

## Continuous Deployments

Once the pipeline is ready, we must ensure that the models are not overwritten and also provide the ability to push changes to the model's fine tuning parameters in a clean manner to ensure that a baseline is used to compare before commiting the final model for validation. A Github Action based continuous deployment process would greatly help in automating this step

Here is a link to a github action that greatly assists in submitting kubeflow pipelines to the Kubeflow cluster 

Excerpt taken from [here](https://github.com/marketplace/actions/kubeflow-compile-deploy-and-run) - It seems to be built for GCP

```
name: Compile, Deploy and Run on Kubeflow
on: [push]

# Set environmental variables

jobs:
  build:
    runs-on: ubuntu-18.04
    steps:
    - name: checkout files in repo
      uses: actions/checkout@master


    - name: Submit Kubeflow pipeline
      id: kubeflow
      uses: NikeNano/kubeflow-github-action@master
      with:
        KUBEFLOW_URL: ${{ secrets.KUBEFLOW_URL }}
        ENCODED_GOOGLE_APPLICATION_CREDENTIALS: ${{ secrets.GKE_KEY }}
        GOOGLE_APPLICATION_CREDENTIALS: /tmp/gcloud-sa.json
        CLIENT_ID: ${{ secrets.CLIENT_ID }}
        PIPELINE_CODE_PATH: "example_pipeline.py"
        PIPELINE_FUNCTION_NAME: "flipcoin_pipeline"
        PIPELINE_PARAMETERS_PATH: "parameters.yaml"
        EXPERIMENT_NAME: "Default"
        RUN_PIPELINE: True
        VERSION_GITHUB_SHA: False

```

The above method helps in continuous deployment but we will also need to build a **Continuous Training (CT)** pipeline where the updated model is referred for testing / prediction

## Observations

This is a pipeline that does every functionality like Data ingestion, cleansing, analysis, training and validating the models. One important consideration is the data is shared between components using volume (or shared folders) as shown in the above image using folder icons. A better option would be to use a database instead of having them as files to ensure that other pipelines can also use the data as needed