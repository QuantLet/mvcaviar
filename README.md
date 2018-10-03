[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **Localizing MV-CAViaR -- Multivariate Conditional Autoregressive Value ar Risk** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

Author: Yegor Klochkov

## Description:

Estimating a Multivariate Conditional Auto-Regressive Value at Risk model (CAViaR) requires adaptation to possibly time varying parameter. Following the strategy of Spokoiny (2009) one can consider a sequence of included time intervals with the same end point, testing each of them agains the largest included one for homogeneity. Performing the tests independently one can choose the largest interval that is not rejected for estimation in order to have the least variance with modelrately small bias. The standard way to test homogeneity is via Change Point detection. The critical values can be estimated using Multiplier Bootstrap proceedure Spokoiny, Zhilova (2013). 

The procedure is implimented in python.

## Acknoledgements:

Financial support from the German Research Foundation (DFG) via International Research Training Group 1792 ”High Dimensional Non Stationary Time Series”, Humboldt-Universität zu Berlin, is gratefully acknowledged.


