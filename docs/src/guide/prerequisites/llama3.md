# Llama 3

Meta's Llama 3 is the most capable openly available LLM (large language model) that can be run locally for inference. I've tried to download the model using HuggingFace repository and convert that to GGUF format to enable it in Ollama for running it locally. The same approach can be used to build other applications like Chatbot, Agents, applications that use LLM for other purposes like language translation etc

## Install Prerequisites

### Install Ollama

[Ollama](https://ollama.com/) is a tool that allows to run open-source large language models such as Llama 2, Llama 3 locally. Download the Ollama binary for the Operating System of choice from this [link](https://ollama.com/download). Install the downloaded binary by double-clicking the installed executable. 

Follow the below steps to install Ollama

#### Step 1

The first step is to acknowledge that the installation can be performed as the software is a downloaded one.

![Image from images folder](~@source/images/llama3/Ollama_install_ack.png)

#### Step 2
The second step is to move the Ollama binary to the Applications folder in Mac OS

![Image from images folder](~@source/images/llama3/Ollama_install_apps.png)

#### Step 3
The third step is to configure Ollama to use it to run the LLM's

![Image from images folder](~@source/images/llama3/Ollama_install_setup.png)

#### Step 4
The fourth step just enables the command line tool from Ollama

![Image from images folder](~@source/images/llama3/Ollama_install_cmdline.png)

#### Step 5
This step marks the completion of installation and shows how to run the Ollama application 

![Image from images folder](~@source/images/llama3/Ollama_install_finish.png)

## Download Llama3 LLMs

### Option 1 - Meta Repo

#### Accept the Meta License Agreement
Visit this [site](https://ai.meta.com/llama/) and review the terms and conditions to accept the license. The following will be provided post the acceptance
- Meta Llama 3 repository
- README
- download.sh 
- Presigned URL as partially shown below

![Image from images folder](~@source/images/llama3/llama_presigned_url.png)

#### Clone Meta Github Repository
Clone Meta Llama 3 Github Repository from [here](https://www.github.com/meta-llama/llama3)

Once you clone the repository, it should pull all the repository contents as shown below

![Image from images folder](~@source/images/llama3/llama_github_repo.png)

#### Download Model

Run the download.sh script present in the root folder to download the models

The first set of steps is to enter the presigned url and also select the model to download - this release includes model weights and starting code for pre-trained and instruction-tuned Llama3 language models with sizes of 8B to 70B parameters

![Image from images folder](~@source/images/llama3/llama_download_1.png)

Once the process runs for sometime, it should download the models like below - I've downloaded the following
- Meta-Llama-3-8B
- Meta-Llama-3-8B-Instruct
- Meta-Llama-3-70B

![Image from images folder](~@source/images/llama3/llama_download_2.png)


### Option 2 - HuggingFace

#### HuggingFace License Agreement
Visit this [site](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and review the terms and conditions and to accept the license to access the repository. It may take sometime for the access to be provided. Once access is granted, create the access token with the following privileges 

- Read access to contents of selected repos
- Read access to contents of all public gated repos you can access

#### Hugging Face client

Use the below command to install huggingface client 

```
pip install "huggingface_hub[cli]"
```

Use the below command to download the model

```
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir Meta-Llama-3-8B-Instruct
```

The above command should download the model like below

![Image from images folder](~@source/images/llama3/llama_huggingface_download.png)

## Test Llama 3 LLM

Let's take the Llama 3 model downloaded from HuggingFace and use it for our testing. Before we start our local inference using the model, we may need to convert the model 

### Convert the HuggingFace model to GGUF File Format 

The GGUF file format (Georgi Gerganov Universal Format) is a specialized file type used in certain machine learning environments. It is designed to encapsulate the data and configuration in the model that is optimized for quick loading and high efficiency especially in local environments. 

#### Clone the Llama.cpp repository

Clone the llama.cpp repository from [here](https://github.com/ggerganov/llama.cpp) and then install the requirements.txt dependencies as shown below

![Image from images folder](~@source/images/llama3/llama_gguf_install.png)

#### Convert the HuggingFace Model to GGUF Format

Next, convert the model to GGUF Format using the below command

```
python convert_hf_to_gguf.py /Users/madhusudanabburu/Documents/pgp-aiml-workspace/meta-llama3/huggingface/Meta-Llama-3-8B-Instruct --outfile Meta-Llama-3-8B-Instruct.gguf --outtype q8_0
```

Once the above command runs for sometime, it should output the below showing the conversion as successful

![Image from images folder](~@source/images/llama3/llama_gguf_conversion.png)

#### Import GGUF Model into Ollama

Create a Modelfile using the below content that has the path of the .gguf file created in the previous step

```
FROM /Users/madhusudanabburu/Documents/pgp-aiml-workspace/meta-llama3/llama.cpp/Meta-Llama-3-8B-Instruct.gguf
```

Next run the Ollama command to create the model from the above Modelfile - I've named the file as **huggingFaceModelfile**

```
ollama create local_Meta_Llama_LLM_8B -f huggingFaceModelfile
```

This should show the below content

![Image from images folder](~@source/images/llama3/ollama_create_model.png)

Let's list if Ollama recognizes it

![Image from images folder](~@source/images/llama3/ollama_list_model.png)

Next step is to run this locally for inference

![Image from images folder](~@source/images/llama3/ollama_run_model.png)

As you can see above, (**highlighted text is my input**), the LLM is running locally and able to interact with the user - though its a pretrained model, it's still good to run a model locally to see if it can help in closed environments 


