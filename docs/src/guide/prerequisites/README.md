# Docker Desktop

Download the Docker Desktop for Mac with either the Apple silicon / Intel chip versions and follow the instructions to install the software on your machine

## Installation

The screens should appear like below

### Step 1
Double click the installation binary and follow the steps to complete the installation by deploying it to Applications folder
![Image from images folder](~@source/images/docker/Docker_Drag_and_Drop.png)

### Step 2
You should be able to see the progress of the installation as shown below
![Image from images folder](~@source/images/docker/Docker_Installation.png)

## Setup

### Step 3
Next, try to open the executable and there may be prompts as below depending on the security configuration of the laptop to get approval from the user to continue opening the Docker application

![Image from images folder](~@source/images/docker/Docker_Approve.png)

### Step 4
::: tip
Visit the settings section to enable Kubernetes 
:::
![Image from images folder](~@source/images/docker/Docker_Enable_Kubernetes_1.png)

### Step 5
Enable the Kubernetes as shown in the screen below
![Image from images folder](~@source/images/docker/Docker_Enable_Kubernetes_2.png)

## Configuration

### Step 1 - Deploy the Kubernetes Dashboard UI

```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml

```
Running the above command, displays the following 
![Image from images folder](~@source/images/docker/Kubernetes_Dashboard_Creation.png)

### Step 2 - Access the Kubernetes Dashboard UI

#### Create a Service Account
The below content creates a Service Account with name 'admin-user' in namespace kubernetes-dashboard first

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: admin-user
  namespace: kubernetes-dashboard
```
Once the file is created, apply the yaml with the below command

```
kubectl apply -f Kubernetes/dashboard-adminuser.yaml
```

You should see the below message
![Image from images folder](~@source/images/docker/Kubernetes_Dashboard_AdminUser.png)

#### Create a ClusterRoleBinding
Once a Kubernetes cluster is established, the ClusterRole already exists in the cluster. We will use it to bind it to our Service Account.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: admin-user
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: ServiceAccount
  name: admin-user
  namespace: kubernetes-dashboard
```
Once the file is created, apply the yaml with the below command

```
kubectl apply -f Kubernetes/dashboard-adminuser-rolebinding.yaml
```

You should see the below message
![Image from images folder](~@source/images/docker/Kubernetes_Dashboard_AdminUser_Rolebinding.png)

## Test
### Run the Kubectl Proxy
Once the above configuration steps are done to enable the Kubernetes Dashboard, try running the proxy to see if the dashboard is made available

```
kubectl proxy
```

The proxy command should show a message like below that says that the dashboard is available in the localhost at port 8001
![Image from images folder](~@source/images/docker/Kubernetes_kubectl_proxy.png)

### Launch the Dashboard

Use the below url to launch the Kubernetes Dashboard
```
http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
```
Launching the web browser with the above url might ask for temporary credentials as shown below

![Image from images folder](~@source/images/docker/Kubernetes_Dashboard_Login.png)   

Let's try to get the token using the below command

```
kubectl -n kubernetes-dashboard create token admin-user
```

The above command should print a token like below (truncated for readability)
```
eyJhbGciOiJSUzI1NiIsImtpZCI6InpXU3RGZzJiY3lXc2p3anI0UVB1OWpWdEZwVkF5UXRCeU9hMVZMZ0RpU2sifQ.eyJhdWQiOlsiaHR0cHM6Ly9rdWJlcm5ldGVzLmRlZmF1bHQuc3ZjLmNsdXN0ZXIubG9jYWwiXSwiZXhwIjoxNzEwOTAxMzM0LCJpYXQiOjE3MTA4OTc3MzQsIm
```

Once you enter this token in the login page, you should be able to view the Dashboard like below

![Image from images folder](~@source/images/docker/Kubernetes_Dashboard.png) 

As you can see, its a blank cluster with no containers etc but atleast a honest attempt at launching the dashboard on a laptop

<PageMeta />