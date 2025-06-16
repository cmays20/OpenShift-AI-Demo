# OpenShift AI Demo
## Description
Some words
## Demo Setup
### Prerequisites
These are prerequisites
1. Start with an OpenShift Cluster
2. OpenShift GitOps Operator Installed (ArgoCD)

## Setting up ArgoCD

### 1. Change the label that is used to identify ArgoCD owned objects

The default label used by ArgoCD is: `app.kubernetes.io/instance: some-application`.
This label is sometimes used by other projects as well. This creates a conflict and can have negative side effects.
Therefore, it is best to change it to something custom.  Reference: [ArgoCD Resource Tracking](https://argo-cd.readthedocs.io/en/latest/user-guide/resource_tracking/).

Add the following to the ArgoCD object that is created when installing the cluster:
```yaml
  extraConfig:
    application.instanceLabelKey: argocd.argoproj.io/instance
```

Some more words
## Performing the Demo
Don't screw it up