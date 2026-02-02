# Hello World

Let's start by developing a Hello World Pipeline that just executes a single task and prints 'Hello World'

## Prerequistes
Let's install the prerequisites **Kubeflow Pipeline SDK (kfp)** for compiling the pipelines on our laptop
* kfp


```
pip install kfp
```
Executing the above command should install the kubeflow pipelines SDK on the laptop and also the tool to compile pipeline code into a yaml/zip/.tar.gz formats as needed

```
kfp dsl
```
Execute the above command to check whether the tool is available to compile pipelines

![Image from images folder](~@source/images/pipelines/hellopipeline/kfp_dsl_compile.png)

Let's create a Hello World Pipeline file as our next step

::: tip
I had to planned to use **dsl-compile** to compile a pipeline but realized it was deprecated and had to change to **kfp dsl compile** 
:::

## Pipeline Source

```
import kfp
from kfp import dsl

@dsl.component
def echo_op():
    print("Hello world")

@dsl.pipeline(
    name='my-first-pipeline',
    description='A hello world pipeline.'
)
def hello_world_pipeline():
    echo_task = echo_op()

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(hello_world_pipeline, __file__ + '.yaml')


```

Let's save the above file into a .py file and use the **kfp dsl compile** command to create a yaml format to upload it to the Kubeflow Dashboard 

## Pipeline compilation 

Use the below command to compile the pipeline into a target format like .yaml to upload to the Kubeflow Dashboard

```
kfp dsl compile --py helloworld.py --output helloworld.yaml
```

Running the above generates the .yaml file as needed

![Image from images folder](~@source/images/pipelines/hellopipeline/kfp_dsl_compile_yaml.png)

## Upload Pipeline to Kubeflow

Let's go to the Kubeflow dashboard on our localhost and try to click on the **Pipelines** link on the left navigation bar and upload our pipeline file

![Image from images folder](~@source/images/pipelines/hellopipeline/kfp_yaml_upload.png)

Uploading the yaml file should show a graph view of the pipeline with its tasks as shown below

![Image from images folder](~@source/images/pipelines/hellopipeline/kfp_pipeline_graph.png)

There is only one method/task in this pipeline which outputs **Hello world** so let's try creating a run in the next step

## Run the Pipeline

Click on **Create Run** shown on the top right of the Dashboard to create a run for this pipeline. It should show the following screen

![Image from images folder](~@source/images/pipelines/hellopipeline/kfp_pipeline_hw_run.png)

Let's click on **Start** at the bottom of the screen and see what happens

I didn't see any outputs on the Dashboard except the status of the Pipeline as 'Complete' as shown below but did see some messages in the Pod logs (shown in the second screenshot below)

![Image from images folder](~@source/images/pipelines/hellopipeline/kfp_hw_dashboard_output.png)

The pod that ran the pipeline did output the **Hello world** text as shown below - It makes sense as pipelines are intended to run background operations.

![Image from images folder](~@source/images/pipelines/hellopipeline/kfp_hw_pod_output.png)

::: tip
I had to create an Experiment to trigger a run as mandated by the Dashboard, here is a simple two step process that creates a KFP Experiment (second step is not depicted here but it just ties the experiment with the pipeline run)

Please visit the **Experiments (KFP)** link on the left navigation bar and create a new experiment

![Image from images folder](~@source/images/pipelines/hellopipeline/kfp_new_exp.png)

Also, if you notice any issues with login to Kubeflow Dashboard due to errors in the cluster like JWKS key missing / invalid headers/OPENSSL_internal:CERTIFICATE_VERIFY_FAILED etc, please do a rolling restart of the pods with the below commands - It works for me !

```
kubectl -n knative-serving rollout restart deploy
kubectl -n istio-system rollout restart deploy
kubectl -n kubeflow rollout restart deploy
kubectl -n kubeflow-user-example-com rollout restart deploy
```


:::

## Observations
Although this is a very basic pipeline, it still demonstrates that there are several factors to consider before converting a Jupyter Notebook into a Pipeline as there is some thought process needed on the design/identification of components that need to be independent and tracked like Data Ingestion, Data Cleansing, Data Analysis, Data splitting, Train models, Store Models, Validate and Test models etc

Let me find a mechanism to convert a Jupyter Notebook to Pipeline with minimal effort




<PageMeta />

