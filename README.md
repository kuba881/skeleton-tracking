# skeleton-tracking
This project was undertaken with the aim of distinguishing successful and unsuccessful boulder climbs using advanced techniques in computer vision. One of the primary methods employed was landmark detection, which involves capturing and analyzing 33 specific points on the human body. These points were meticulously tracked and used to predict the outcome of the climbs.

The collected data was thoroughly annotated and subsequently classified using a convolutional neural network (CNN). CNNs are a class of deep neural networks, most commonly applied to analyzing visual imagery. They are particularly well-suited for this type of task due to their ability to automatically and adaptively learn spatial hierarchies of features from input images.

In addition to supervised learning with CNNs, the project also utilized unsupervised learning methods, specifically fuzzy clustering. Fuzzy clustering is a form of clustering that allows data points to belong to multiple clusters with varying degrees of membership. This approach is beneficial for handling the inherent ambiguity and overlap in human movement data.

Despite the limited amount of data available for this study, the results were surprisingly promising. Both the convolutional neural network and the fuzzy clustering methods achieved nearly 80% accuracy in distinguishing between successful and unsuccessful climbs. This level of accuracy is quite impressive, given the complexity and variability of human movements in bouldering.

For those interested in a more detailed exploration of this project, including the methodologies and results, further information can be found in the accompanying article.pdf. This document provides a comprehensive overview of the project's framework, the data collection process, the specific techniques used for data analysis, and the implications of the findings.

Overall, this project demonstrates the potential of computer vision and machine learning techniques in sports analytics, particularly in the field of bouldering. The successful application of both supervised and unsupervised learning methods opens up new avenues for future research and development in this area.
