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

### Create the ArgoCD Applications
First, apply the file argocd-setup.yaml.  Make sure it fully completes.

Second, apply the demo-setup.yaml file.  This will create an App of Apps application that rolls everything else out.

## Performing the Demo
Don't screw it up