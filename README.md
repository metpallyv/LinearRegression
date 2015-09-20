# LinearRegression
Goal of this project is to implement Linear Regression using Gradient Descent algorithm and Normal Equations without using any Machine learning libraries

# Gradient Descent for Linear Regression
https://en.wikipedia.org/wiki/Gradient_descent

In this project, I implemented Linear Regression using GD on three datasets:

• Housing: This is a regression dataset where the task is to predict the value of houses in the
suburbs of Boston based on thirteen features that describe different aspects that are relevant
to determining the value of a house, such as the number of rooms, levels of pollution in the
area, etc. Here is the definition for the features:

1. CRIM: per capita crime rate by town 
2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft. 
3. INDUS: proportion of non-retail business acres per town 
4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
5. NOX: nitric oxides concentration (parts per 10 million) 
6. RM: average number of rooms per dwelling 
7. AGE: proportion of owner-occupied units built prior to 1940 
8. DIS: weighted distances to five Boston employment centres 
9. RAD: index of accessibility to radial highways 
10. TAX: full-value property-tax rate per $10,000 
11. PTRATIO: pupil-teacher ratio by town 
12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
13. LSTAT: % lower status of the population 
14. MEDV: Median value of owner-occupied homes in $1000's


• Yacht: This is a regression dataset where the task is to predict the resistance of a sailing
yacht’s structure based on six different features that describe structural and buoyancy
properties. Here is the definition of the features:

1. Longitudinal position of the center of buoyancy, adimensional. 
2. Prismatic coefficient, adimensional. 
3. Length-displacement ratio, adimensional. 
4. Beam-draught ratio, adimensional. 
5. Length-beam ratio, adimensional. 
6. Froude number, adimensional. 
 

• Concrete: This is a regression dataset where the task is to predict the compressive strength
of concrete on nine different features. There are a total of 1030 instances and all the features
are numeric. Here is the definition of the features:

1. Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
2. Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
3. Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
4. Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
5. Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
6. Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
7. Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
8. Age -- quantitative -- Day (1~365) -- Input Variable
9. Concrete compressive strength -- quantitative -- MPa -- Output Variable 


I use the following set of parameters:

(a) Housing: learning rate = 0.4 × 10−3, tolerance = 0.5 × 10−2

(b) Yacht: learning rate = 0.1 × 10−2, tolerance = 0.1 × 10−2

(c) Concrete: learning rate = 0.7 × 10−3, tolerance = 0.1 × 10−3

NOTE: Here tolerance is defined based on the difference in root mean squared error (RMSE)
measured on the training set between successive iterations. 

For both datasets I use ten-fold cross-validation to calculate the RMSE for each fold and the
overall mean RMSE. I Summarized my results for each dataset as a table and reported the
SSE for each fold and also the average SSE and its standard deviation across the folds.

I also  selected any fold, and plot the progress of the gradient descent algorithm for each dataset
separately in two different plots. To this end plot the RMSE for each iteration.

