# OpenShift AI Demo
## Description
Some words
## Demo Setup
### Prerequisites
These are prerequisites
1. Start with an OpenShift Cluster
2. OpenShift GitOps Operator Installed (ArgoCD)
3. Clone this repository

## Update the overlays for your current cluster
The URLs and other cluster specific values will need to be updated in the clone of this repository.
Here is a list of file that will need to be updated:
1. [Cert Manager](/gitops/Operators/CertManager/instance/overlay/kustomization.yaml)
2. [GPU Instance](/gitops/GPU-instance/machine-sets/overlay-zone-a/patch-zone-a.yaml)
3. [Git Repo Information](/gitops/ArgoCD-Applications/overlay/kustomization.yaml)

### Create the ArgoCD Applications
First, apply the file argocd-setup.yaml.  Make sure it fully completes.

Second, apply the demo-setup.yaml file.  This will create an App of Apps application that rolls everything else out.

## Performing the Demo

1. Go to OpenShift AI GUI and login
2. Navigate to the airplane-detection project
3. Click on the link to create a new workbench
   1. Name: airplane-detection-wb
   2. Image selection: CUDA
   3. Version selection: latest (2025.1)
   4. Container size: Medium
   5. Accelerator: nvidia-gpu
   6. Connections: Attach existing connection -> model-storage-connection
   7. Click Create workbench
4. Open Workbench once it is running
5. Click on the Git tab -> Clone a Repository
   1. Repository URL: https://github.com/cmays20/OpenShift-AI-Demo.git
   2. Click Clone
6. Browse to OpenShift-AI-Demo/demo folder
7. Open airplane-detection.ipynb
8. Select the Kernel (Python 3.11)
9. Run the notebook
   1. Talk about setup
   2. Talk about training
   3. Talk about validation
   4. Talk about inference
   5. Talk about deploying to S3
10. Show the bucket and the model now in it
11. Go back to the airplane-detection product and serve the model
    1. Click on a Single-model server
    2. Click Deploy Model
    3. Model deployment name: airplane-detection-model
    4. Serving runtime: OpenVINO Model Server
    5. Model framework: onnx - 1
    6. Deployment mode: Standard
    7. Accelerator: none - not needed for this model
    8. Existing Connection Path: models/airplane-detection
    9. Click Deploy
12. Wait for the model to be deployed
13. Show the internal endpoint
14. Pull up the Airplane Detection webapp
15. Show that we can now detect airplanes