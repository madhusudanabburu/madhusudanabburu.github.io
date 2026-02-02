# Kubeflow

## Prerequisites
Kubeflow needs the following tools to be installed in the system before the setup 

* Docker
* Kubernetes
* [Kustomize](https://kubectl.docs.kubernetes.io/installation/kustomize/)
* [Kind](https://kind.sigs.k8s.io/)

Since we have already installed Docker Desktop and also enabled Kubernetes in our previous step, we should continue installing Kustomize and Kind to proceed with the setup of Kubeflow

## Installation

### Step 1

Let's install Kustomize using HomeBrew for mac. There are installation options for other operating systems as specified here
[Kustomize](https://kubectl.docs.kubernetes.io/installation/kustomize/)

```
brew install kustomize
```

![Image from images folder](~@source/images/kubeflow/kustomize_install.png)

### Step 2

Let's install [Kind](https://kind.sigs.k8s.io/) using HomeBrew for mac. Kind supports running local kubernetes clusters on Docker

```
brew install kind
```

![Image from images folder](~@source/images/kubeflow/kind_install.png)


### Step 3

Let's create a Kind cluster with the following yaml content (Includes 2 worker nodes)

```
cat <<EOF | kind create cluster --name=kubeflow  --kubeconfig mycluster.yaml --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: ClusterConfiguration
    apiServer:
      extraArgs:
        "service-account-issuer": "kubernetes.default.svc"
        "service-account-signing-key-file": "/etc/kubernetes/pki/sa.key"
- role: worker
- role: worker
EOF
```

The above step should be creating a kind cluster as shown below 

![Image from images folder](~@source/images/kubeflow/kind_cluster.png)

### Step 4

Save kubeconfig - This step is to ensure that all config data is stored in a backup file

```
mv ~/.kube/config ~/.kube/config_backup
kind get kubeconfig --name kubeflow > ~/.kube/config
```
Both the above steps should run without any issues / output content

### Step 5

Create a secret based on existing credentials in order to pull the images (Update the --from-file parameter to include the full path)

```
docker login
kubectl create secret generic regcred \
    --from-file=.dockerconfigjson=~/.docker/config.json \
    --type=kubernetes.io/dockerconfigjson
```

### Step 6

Clone the kubeflow's manifest repository

```
git clone https://github.com/kubeflow/manifests.git
```

![Image from images folder](~@source/images/kubeflow/Kubeflow_manifest_clone.png)

Change to the manifest folder

```
cd manifests
```
You should be able to see the following (or similar) files as shown below

![Image from images folder](~@source/images/kubeflow/Kubeflow_manifest_folder.png)

### Step 7

Build and apply the YAML files for all kubeflow components. This step will install all Kubeflow official components (residing under apps) and all common services (residing under common) 

```
while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
```

The above command should run for sometime to complete setting up the kubeflow components.  

At the end, all the kubeflow pods should be up and running as shown below

![Image from images folder](~@source/images/kubeflow/Kubeflow_pods_running.png)

Verify if all the pods are ready by individually checking them with the names below

```
kubectl get pods -n cert-manager
kubectl get pods -n istio-system
kubectl get pods -n auth
kubectl get pods -n knative-eventing
kubectl get pods -n knative-serving
kubectl get pods -n kubeflow
kubectl get pods -n kubeflow-user-example-com
```

## Issues
I did end up with several pods unable to run due to missing images for the Apple silicon arm64 architecture. I had to update the image tags and also rebuild the kserve/models-web-app into a separate docker image and host it on my repository

::: tip

Please go into the app and common folders and update the <b>kustomization.yaml</b> file to update the tags from v1.8.0-rc.0 to v1.8.0 for the following images 

* kubeflownotebookswg/centraldashboard
* kubeflownotebookswg/poddefaults-webhook
* kubeflownotebookswg/jupyter-web-app
* kubeflownotebookswg/notebook-controller
* kubeflownotebookswg/profile-controller
* kubeflownotebookswg/kfam
* kubeflownotebookswg/pvcviewer-controller
* kubeflownotebookswg/tensorboard-controller
* kubeflownotebookswg/tensorboards-web-app
* kubeflownotebookswg/volumes-web-app
* kserve/models-web-app

For the kserve/models-web-app, clone the public repository <https://github.com/kserve/models-web-app.git> and build the docker image with the below commands (this step ensures that the image is built on your laptop for the apple silicon architecture)

```
docker build -t <any org>/models-web-app:v1.0.0 .

docker push <any org>/models-web-app:v1.0.0

```
Once the docker image for models-web-app is built and uploaded to your org, the next step is to update the following files in the Kubeflow manifest's repository to reflect the new image 

* contrib/kserve/models-web-app/base/deployment.yaml
* contrib/kserve/models-web-app/base/kustomization.yaml

Redeploy the models-web-app using the below command

```
kustomize build contrib/kserve/models-web-app/overlays/kubeflow | kubectl apply -f -
```

Below are some handy commands to look at the pods and inspect the errors for corrective actions

```
kubectl get pods -A
kubectl describe pod --namespace=kubeflow <podname>
```

:::

## Test

One last step to access the Kubeflow's built-in Dashboard is to use the kubectl proxy to forward the requests through the istio-ingressgateway service in the istio-system namespace.

```
kubectl port-forward svc/istio-ingressgateway -n istio-system 9090:80
```

You should be able to see like below

![Image from images folder](~@source/images/kubeflow/Kubeflow_dashboard_proxy.png)

The Login screen for the Kubeflow dashboard should appear like below once the proxy starts working

![Image from images folder](~@source/images/kubeflow/Kubeflow_dashboard_login.png)

Once you login with the following credentials, you should be able to see the dashboard

```
user@example.com
12341234
```
![Image from images folder](~@source/images/kubeflow/Kubeflow_dashboard.png)

Tried opening a Jupyter Notebook and its able to launch as shown below

![Image from images folder](~@source/images/kubeflow/kubeflow_jupyter_notebook.png)


::: tip
At this point in time, I'm not sure how the credentials are stored / updated but the dashboard does take some time to show up and also apply the namespace 

I was put on the kubeflow-user-example-com namespace by default 

Also, the Jupyter Notebook requires worker nodes to be setup as its launched as a separate pod
- Our kind cluster configuration takes care of this requirement by adding 2 worker nodes 
:::

<PageMeta />