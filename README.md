# ID3-Decision-Tree-Using-Python

Assignment 1
MACHINE LEARNING

The following are the grading rules for assignment 1:
•	General rules: you are free to choose the programming languages you like. For the core functions(ID3, C4.5, data splitting and k-fold cross-validation) in this assignment, you are not allowed to use the libraries provided by the language. However, you may use some libraries to store and preprocess the data, like numpy, pandas in python. The detailed rules are as below:

•	Successfully implement decision tree with ID3 or C4.5 algorithm (60 pts)
1.	Doesn't implement ID3 or C4.5 by yourself or fail to implement one of them (-40 pts)
2.	Can not print out(visualize) the tree after running the program (-20pts)
•	Successfully implement data splitting and k-fold cross-validation (30pts)
1.	Doesn't implement them by yourself or fail to implement them (-15pts/each)
2.	Can not print out the prediction accuracy (-5pts/each)
3.	No point will be deduced if this item you get 0.
•	A brief document illustrates the interfaces of functions you implement and the process you conduct experiments to evaluate the functions(10pts).
1.	Fail to provide the interfaces (-3pts)
2.	Fail to provide the steps of experiments (-3pts)
3.	Fail to report the output results (i.e. the visualization of decision tree(you may post a screenshot in the document) and predicting accuracy) (-4pts)
•	You should compress your source code and document in one zip file and name as your NetID_assignment1.zip. You are asked to submit the zip file to the blackboard before the deadline. 





I have implemented ID3(decision tree) using python from scratch on version 2.7. To run this program you need to have CSV file saved in the same location where you will be running the code. I have attached all the CSV datafiles on which I have done testing for the model.
Now to explain my code I have used following functions:-
1.	Splitting Criteria Used to create tree:- Method used Entropy Gain (for ID3 specific) and Gini Index gain(enhancing to CART)
Methods used:-
•	gini_index(groups, classes)
•	entropy(groups, classes,b_score)

2.	Data Splitting from dataset:-  For splitting the data set I have created two separate functions one to create a single random split of data set such that training size =0.8 and test size =0.2. Second one is made to cater the k-cross validation need in which dataset is divided into folds with randomized values provided in the arguments.
Methods used:-
•	train_test_split(dataset, split=0.80)
•	cross_validation_split(dataset, n_folds)

3.	Tree Creation on training set using splitting criterion:- Now once we have training set and entropy gain values for all the attributes on training set. First the attribute giving maximum Entropy gain is made the root and for finding left child and right child again the same recurrence function call is made and this step continues until the stopping condition is reached.
Methods used:-
•	build_tree(train, max_depth, min_size,split_parameter)
•	split(node, max_depth, min_size, depth)
•	get_split(dataset,split_parameter)
•	test_split(index, value, dataset)
•	to_terminal(group)

4.	Prediction of Test Set using model created on training set:- Once we have the model created on the training set now we are using this model to do predictions on test set.
Methods used:-
•	predict(node, row)
•	decision_tree(train, test, max_depth, min_size,split_parameter)


5.	Calculating Accuracy by Comparing actual class values  and predicted class values of the test set. For k-cross validation we find the accuracy by taking mean of total individual fold accuracy values.
Method used:-
•	accuracy_metric(actual, predicted)

References taken to complete the assignment:-
1.	https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
2.	https://machinelearningmastery.com/implement-resampling-methods-scratch-python/
3.	http://crystal.uta.edu/~cli/cse5334/slides/cse5334-fall17-06-decisiontree.pdf


 
