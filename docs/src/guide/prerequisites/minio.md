# MinIO
[MinIO](https://min.io/) is a Kubernetes native high performance object store with an S3 compatible API. It supports deploying MinIO tenants onto private and public cloud infrastructures

This is optional if Kubeflow is setup locally as the volumes available via the Kubeflow Dashboard should help in providing the required storage 

## Prerequisites

Following are the prerequisites for setting up MinIO

* Kubernetes

## Installation

Use the below command to install the MinIO Operator onto the Kubernetes cluster

```
kubectl apply -k github.com/minio/operator/
```

The above command should run like below and install the minio operator

![Image from images folder](~@source/images/minio/minio-operator.png)

Test the minio-operator namespace to see if all Pods are running using the below command

```
kubectl get pods -n minio-operator
```
![Image from images folder](~@source/images/minio/minio-pods.png)

## Test

### Step 1

Now that the minio pod is running, let's access the MinIO console and create a tenant. We will need to use a token generated from the minio-operator as shown below

```
kubectl apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: console-sa-secret
  namespace: minio-operator
  annotations:
    kubernetes.io/service-account.name: console-sa
type: kubernetes.io/service-account-token
EOF
SA_TOKEN=$(kubectl -n minio-operator  get secret console-sa-secret -o jsonpath="{.data.token}" | base64 --decode)
echo $SA_TOKEN
```

Let's forward the UI to port 9090/9191 and login using the token generated from the above command

The login screen should resemble like below

![Image from images folder](~@source/images/minio/minio-login.png)

Once the token is entered, you should be able to see a screen as below, next step is to create a tenant that can store the data

![Image from images folder](~@source/images/minio/minio-after-login.png)

Please create the tenant as needed and **store the generated credentials** so it can be configured in Kubeflow and also used to login to the tenant console

![Image from images folder](~@source/images/minio/minio-tenant.png)

The tenant should appear as below

![Image from images folder](~@source/images/minio/minio-tenant-creation.png)

### Step 2

Once the tenant is up and running, we need to access it via the browser using the following url
**minio.minio-operator.svc.cluster.local** (minio.**namespace**.svc.cluster.local) and also enable the tenant console proxy

Identify the tenant console with the following command

```
kubectl get all --namespace minio-operator
```

![Image from images folder](~@source/images/minio/minio-tenant-console.png)

Use the kubectl proxy function to forward the requests as below

```
kubectl port-forward service/covid-images-console -n minio-operator 9443
```

Access the tenant console 

Use the credentials from the tenant creation process in [Step 1](#step-1) to login as shown below

![Image from images folder](~@source/images/minio/minio-tenant-login.png)

I created a bucket to store data from the Kubeflow pipelines

![Image from images folder](~@source/images/minio/minio-tenant-bucket.png)

::: tip

The url **minio.minio-operator.svc.cluster.local** had to be resolved to localhost so I had to update the /etc/hosts file to reflect the below

127.0.0.1 	minio.minio-operator.svc.cluster.local

![Image from images folder](~@source/images/minio/minio-traceroute.png)

:::


<PageMeta />





