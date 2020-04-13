# ENGG4801
ENGG4801 Thesis: Bayesian Deep Learning on Longitudinal Medical Data for Cancer Prognosis

## Overview
This repo contains the code used within my thesis project undertaken for my Bachelors of Engineering (Honours). 

## Repo Structure
The project is based on milestones of work and this repo follows such a layout. Upon progress, shared code will be taken out into shared directories. below gives the structure:

```
ENGG4801
│   README.md
|
└───m1_ISPY_processing/
│   │  ...
│   └── README.md
│   
└──shared/
   |  ...
   └── README.md

```
Where:
*  m1_ISPY_processing: Is responsible for parsing the [TCIA ISPY](https://wiki.cancerimagingarchive.net/display/Public/ISPY1) dataset from GCPs HealthCare API collection (see [dataset](https://cloud.google.com/healthcare/docs/resources/public-datasets/tcia-attribution/ispy1)), converting to appropriate TFRecord and distribution metadata, and egressing to AWS storage for training. 
* shared: contains shared code used throughout the repo.  
