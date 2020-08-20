# Music to My Ear: Recommender System with Million Song Dataset

Xiaoyi Chen, Zhiran Chen, Kaicheng Ding, Weixin Liu, Xuening Wang, Ruitao Yi

Carniegie Mellon University

## Introduction

We propose and implement a machine learning pipeline that combines content-based and collaborative recommendation methods for a large-scale, personalized song recommendation system. The goal is to predict which songs that a user will listen to and make a recommendation list of 10 songs to each user, given both the user’s listening history and full information (including meta-data and audio feature analysis) for all songs. 

## Dependencies

* Python 3.6
* Tables 3.6.1
* h5df 0.1.5
* Numpy 1.18
* Scikit-Learn 0.23.2 
* Pandas 0.15.2
* Matplotlib 3.3.1
* Seaborn 0.10.1
* Spark_notebook_helpers 1.0.1

## Files

```
.
├── utils
├── 10605_Project_Report.pdf
├── README.md
├── collaborative_bad_map.ipynb
├── collaborative_good_map.ipynb
├── dependencies.sh
├── preprocessing_zepplin.json
├── setup-script.sh
└── sid_mismatches.txt
```

## Usages

### Install Dependencies

To install dependencies, please run the following command to install everything required automatically:

```bash
$ ./setup-script.sh
```

### Download Dataset

Download and extract dataset from here: http://millionsongdataset.com/ to designated directories.

## Results

We achieved 0.4 recall when setting the cosine similarity threshold as 0.9 and 5.0524 RMSE with collabrative filtering

## References

Fabio Aiolli. [A preliminary study on a recommendersystem for the million songs dataset challenge](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.417.2061&rep=rep1&type=pdf#page=80). Volume964, 01, 2013.

Thierry Bertin-Mahieux, Daniel PW Ellis, Brian Whit-man, and Paul Lamere. [The million song dataset](https://academiccommons.columbia.edu/doi/10.7916/D8NZ8J07). 2011.

Yi Li, Rudhir Gupta, Yoshiyuki Nagasaki, and TianheZhang. [Million song dataset recommendation projectreport](http://www-personal.umich.edu/~yjli/content/projectreport.pdf). 2012.

B. McFee, T. Bertin-Mahieux, D. Ellis, and G. Lanck-riet. [The million song dataset challenge](https://dl.acm.org/doi/pdf/10.1145/2187980.2188222). 2012.
