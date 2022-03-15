# kNN-Tree
In this project, we explored the utility of merging the classification tree or CART and the k-nearest neighbor classifier. <br>
The kNN classifier is a lazy learner and generally suffers from the computation time spent on searching for neighbors. CART runs fast, but at deeper levels of the tree, it sometimes fails to capture nonlinear boundaries between different classes. <br>
<br/>
Our hypthesis is that kNN and CART can complement each other in this regard. In our model, a kNN model is built in each leaf node of the regularized CART tree. CART essentially groups samples in each leaf together. Hence, kNN only needs to search through a subset of training samples when labeling each test sample. From another point of view, kNN helps refine the prediction boundary in each leaf node. After a certain depth of CART, kNN brings in a way to go beyond the fundamental split of CART which only detects a linear border with respect to one feature at a time. <br>
<br/>
In addition to the complementary characteristics of the two models, we also showed that we can leverage the feature importance measure produced during the CART construction in the final prediction by kNN. <br>
<br/>
The experiment showed that in many scenarios, the merging of CART and kNN resulted in improvement of both computation time and accuracy over the baseline CART and kNN. <br>
<br/>
This project was presented at [The 2021 INFORMS Annual Meeting](https://meetings.informs.org/wordpress/anaheim2021/). <br>
<br/>
<sub>Codes will be uploaded shortly.</sub>
