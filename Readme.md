# Data Science Project
This Automated Pneumonia Anomaly Detection Screener Using Artificial Intelligence is a research project aimed at developing a hybrid machine learning and deep learning model for diagnosing pneumonia from chest X-ray images. The project combines the visual feature extraction power of VGG16 Convolutional Neural Network (CNN) with the classification capability of a traditional machine learning algorithm, specifically a Random Forest (RF) classifier.
The goal is to offer an efficient, accurate, and computationally affordable diagnostic tool suitable for deployment in healthcare systems with limited computational resources—particularly in remote or underdeveloped regions. The system automates pneumonia detection, minimizing human error, improving diagnostic speed, and supporting early medical intervention.


## Problem Statement
Pneumonia remains one of the leading causes of mortality worldwide, especially among children under five and elderly populations. Traditional diagnosis using chest X-ray interpretation is manual, subjective, and prone to misdiagnosis due to the subtle visual similarities of pneumonia symptoms with other lung infections.
Existing deep learning models (such as ResNet50, DenseNet201, and InceptionV3) have shown remarkable accuracy but are computationally intensive, requiring significant processing power and memory, which limits their practical use in low-resource medical environments.
Therefore, there is a need for a lightweight, hybrid model that achieves high accuracy comparable to deep learning architectures while maintaining low computational cost and high generalizability across different datasets.

## 🛠 Tools Used
Programming Language: Python

Deep Learning Frameworks: TensorFlow, Keras

Machine Learning Libraries: Scikit-learn

Pre-trained CNN Model: VGG16 (feature extraction)

Classifier: Random Forest (supervised learning)

Image Processing and Visualization: OpenCV, Matplotlib, NumPy, Pandas

Performance Metrics: Accuracy, Precision, Recall, F1-score, Sensitivity, Specificity

Dataset: ChestX-ray14 and/or publicly available pneumonia X-ray datasets

Hardware Environment: GPU-enabled system (for CNN feature extraction and model training)

## Documentation

Automated Pneumonia Anomaly Detection Screener Using Artificial Intelligence

Botchway Theophilus
Department of Computer Science, Monroe University, King Graduate School
CS628-154HY: Data Science
Professor. Khadhirunissa Shaik
October 20, 2025

















### Introduction and Objectives 

Pneumonia is a severe lung infection caused primarily by viruses or bacteria and is a major cause of global mortality, particularly among children and the elderly (Ortiz-Prado et la., 2025). According to the World Health Organization (WHO), pneumonia accounts for nearly 12.8% of annual deaths among children under five. Diagnosing pneumonia traditionally relies on the expert interpretation of chest X-rays (Thi Le et al., 2025). However, similar visual patterns for pneumonia in some X-ray images have often led to misdiagnosis, even by experienced radiologists, which has been an ongoing challenging concern for the healthcare industry.
To address these challenges, this research aims to design an automated screening system for pneumonia anomaly detection using a model that relies on deep learning and machine learning combination infrastructure. The main objective of building such a hybrid model is that, although state of the art deep learning models provides superior performance in speed and accuracy, these models are computationally expensive because of their sophisticated architecture. On the other hand, a machine learning model alone is limited in visual prowess. Thus, this proposed hybrid model intends to bridge both worlds; accuracy, with affordability and accessibility for pneumonia detection (especially resource-limited healthcare in underdeveloped remote settings).




###  Current State of Arts and Existing Methodologies

Recent advances in improved medical image classification, prominently rely on Convolutional Neural Network architectures, such as VGG16, ResNet50, DenseNet201, InceptionV3, and MobileNetV2. These deep learning models have demonstrated success in pneumonia detection and other medical diagnostics. Rajpurkar et al. (2017) used a 121-layer CNN achieving performance exceeding human radiologists on ChestX-ray14 dataset; Kermany et al. (2018) achieved 92.8% accuracy using image-based deep learning for disease classification; Stephen et al. (2019) and Saraiva et al. (2019) proposed CNN models achieving accuracies around 93–95%; Liang and Zheng (2020) used a deep residual network with 49 layers, obtaining 90% accuracy but high computation cost; El Asnaoui et al. (2020) explored multiple CNN architectures, where ResNet50 and Inception-ResNet-V2 achieved over 96% accuracy. 
While these models performed well, their limitations include large parameter counts (Rajpurkar et al., 2017), slow training (El Asnaoui et al., 2020), and high computational cost (Liang and Zheng, 2020), which can restrict decision margins.

 

###  Concerns: Barriers, Issues, and Open Problems
Despite substantial progress, existing methods face key challenges:
•	Data scarcity and imbalance: Medical datasets are often limited and unevenly distributed across classes.
•	Overfitting risks: Deep networks with limited medical data can easily overfit.
•	Limited generalization: Many models trained on specific datasets fail to generalize across populations.
•	Computational overhead: Deeper models (e.g., ResNet, DenseNet) require high computation, limiting deployment in low-resource environments.

		These concerns highlight the need for a model that can be lightweight and generalizable, while offering near-precision accuracy like the sophisticated state-of-the-art models. This paper proposes a solution through a hybrid deep learning CNN (like VGG16, ResNet50 etc.) coupled with a machine learning classifier (like logistic regression, random forest, SVM etc.) approach.









### Existing and Expected-Proposed Solutions and their Respective Methods and Algorithms 

		Rajpurkar et al., (2017) presented an algorithm to detect pneumonia for chest X-ray images, where the authors assure that the performance of the proposed algorithm exceeds the practicing radiologists. The methodology relied on a CNN of 121 layers trained architecture with a softmax classification algorithm utilizing the set of images of ChestX-ray14, including more than 100,000 X-ray images with 14 diseases. However, with the large dataset, it suffered slow training and high computational cost. 
			Kermany et al., (2018) proposed medical diseases diagnoses and treatment by utilizing image-based deep learning models to detect or classify several medical datasets including of the dataset utilized in this paper. Same method and algorithm as above but with comparatively lower dataset, performance of the proposed method was comparable to human specialists and achieved accuracy 92.8%. This paper enhanced the performance, but the drawback was it depend on lower-level features. 
			Wua et al., (2020) proposed a hybrid model to predict pneumonia with chest X-ray images based on adaptive median filter convolutional neural network and random forest (RF) classification algorithm. The author used adaptive median filtering to remove noise in the chest X-ray image, which makes the image more easily identified and achieved high accuracy. Then CNN architecture with two layers is established based on dropout to extract features. Although the adaptive median filter can improve the classification accuracy of CNN, it needs additional preprocessing. 

			El Asnaoui et al., (2020) proposed automatic methods for detection and binary classification algorithm pneumonia images using different CNN architectures such as VGG16, VGG19, DenseNet201, Inception ResNetV2, InceptionV3, Resnet50, MobileNetV2 and Xception. Although it showed that the Resnet50, MobileNet_V2 and Inception_Resnet_V2 gave highly performance (accuracy more than 96%), these models utilized deeper convolutional layer with huge number of parameters.
			This paper’s proposed solution aims to achieve cost-effectiveness with precision-efficiency by opting for a hybrid model that does not bring the overhead computational expense by trading-off some accuracy proficiency (Addisu et al., 2025). The methodology relies on a VGG-I6 CNN coupled with a Random Forest classification algorithm. Now, because Addisu et al focused on a heart disease use case, this pneumonia use case will need the hybrid model designed from scratch in order to measure performance metrices tailored to the pneumonia use case.
 

### Details and Expected Research Results (step by step illustrations, concepts, principles, theories, algorithms, methodologies, etc.) 

(Step-by-Step Process)
1.	Input Data Preparation
o	Collect and preprocess the dataset (resize images to 224×224, normalize pixel values, apply augmentation if necessary).
2.	Feature Extraction
o	Load pre-trained VGG16 (without top layers).
o	Pass each image through VGG16 and extract feature maps from the last convolutional block.
3.	Feature Processing
o	Flatten extracted features.
o	Apply PCA or other feature selection methods if dimensionality is too high.
4.	Train-Test Split
o	Divide dataset (e.g., 80% training, 20% testing).
5.	Model Training
o	Train Random Forest using extracted training features and labels.
6.	Evaluation
o	Predict on the test set.
o	Compute accuracy, precision, recall, F1-score, and confusion matrix.
7.	Visualization
o	Plot confusion matrix to interpret model behavior.
f.	Comparative Analysis of Research Methods, Algorithms, and Expected Results 
Table of Performance matrices for the researched and proposed models
Models	Precision%	Recall%	F1 score%	Sensitivity
%	Specificity
%	Accuracy%
Kermany et al [13]	-	93.2	-	93.2	90.1	92.8
Stephen et al [14]	-	-	-	-	-	93.7
Liang et al [16]	89.1	96.7	92.7	96.7	-	90.5
El Asnaoui et al [4]	98.5	94.9	96.7	94.9	98.4	96.6
Hybrid model (VGG16–RF) heart disease use-case	91.3	92.2	91.8	92.2	91.8	92.0


 






### Conclusions and Future Research Directions.

		In future work, this research can be expanded to other pre-trained CNNs to solve multi-classification problems. An additional layer of U-Net based architecture can be attached to the VGG16-RF model to form a simpler infrastructure of VGG16-UNet-RF. This integrates the feature extraction capabilities of the pretrained CNN into a U-Net-like encoder-decoder structure. The extracted features from the bottleneck layer or a pooling layer of the U-Net are then flattened and used to train a separate RF classifier layer. This will however require an understanding of deep learning for sophisticated convolution neural networks.

























### References

Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T. and Ng, A. Y. (2018) CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. arXiv: 1711.05225.
D. Kermany, K. Zhang, and M. G. baum, “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, 2018. [Online]Available:https://data.mendeley.com/datasets/rscbjbr9sj/2. 
H. Wua, P. Xiea, H. Zhangb, D. Lic, and M. Chengd, “Predict Pneumonia with Chest X-ray Images Based on Convolutional Deep Neural Learning Networks”, Journal of Intelligent and Fuzzy Systems, Vol. 39, No. 12, pp. 1-15, 2020. 
O. Stephen, M. Sain, U. J. Maduh, and D. U. Jeong, “An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare”, Journal of Healthcare Engineering, Hindawi, Vol. 2019, pp. 1-7, 2019. 
G. Liang and L. Zheng, “A transfer learning method with deep residual network for pediatric pneumonia diagnosis”, Computer Methods and Programs in Biomedicine, Vol. 187, pp. 1-9, 2020. 
Addisu, E. G., Yirga, T. G., Yirga, H. G., & Yehuala, A. D. (2025). Transfer learning-based hybrid VGG16-machine learning approach for heart disease detection with explainable artificial intelligence. Frontiers in Artificial Intelligence, 8. https://doi.org/10.3389/frai.2025.1504281
K. E. Asnaoui, Y. Chawki, and A. Idri, “Automated Methods for Detection and Classification Pneumonia based on X-Ray Images Using Deep Learning”, Electrical Engineering and Systems Science, Image and Video Processing, pp. 1-28, 2020. 
Ortiz-Prado, E., Vasconez-Gonzalez, J., Becerra-Cardona, D., Farfán-Bajaña, M. J., García-Cañarte, S., López-Cortés, A., & Izquierdo-Condoy, J. (2025). Hemorrhagic fevers caused by South American Mammarenaviruses: A comprehensive review of epidemiological and environmental factors related to potential emergence. Travel Medicine and Infectious Disease, 64https://doi.org/10.1016/j.tmaid.2025.102827
Thi Le, H., Thi Bich Phung, T., Thi Bui, H., Thi Hong Le, H., Tran, D. M., Nguyen, N. H., Phan, H. T., Tran, V. D., Vu Pham, U., Van Phan, N., Thu Do, H., Nguyen, A. H., Pham, T. D., & Thi Van Nguyen, A. (2025). Nasal-spraying Bacillus spore probiotics for pneumonia in children with respiratory syncytial virus and bacterial co-infections: a randomized clinical trial. Communications Medicine, 5(1), 336. https://doi.org/10.1038/s43856-025-01029-9
World Health Organization: WHO. (2019, November 7). Pneumonia. https://www.who.int/health-topics/pneumonia/#tab=tab_1

