\###Data Mining Exercise 2 \##Saager Buch, Joe Monahan, and Phillip An
\#Problem 1: Saratoga house prices

    # allocate to folds
    N = nrow(SaratogaHouses)
    K = 10
    fold_id = rep_len(1:K, N)  # repeats 1:K over and over again
    fold_id = sample(fold_id, replace=FALSE) # permute the order randomly

    maxM = 10
    err_save = matrix(0, nrow=1, ncol=K)

    for(i in 1:K) {
      train_set = which(fold_id != i)
      y_test = SaratogaHouses$price[-train_set]
        this_model = lm(price ~ . - centralAir - sewer - bathrooms + rooms:fireplaces+lotSize:landValue + lotSize:bedrooms+landValue:fireplaces, data=SaratogaHouses[train_set,])
        yhat_test = predict(this_model, newdata=SaratogaHouses[-train_set,])
        err_save[1, i] = rmse(this_model, data=SaratogaHouses[-train_set,])
      }



    err_save = as.data.frame(err_save)

    plot(1:K, colMeans(err_save))

![](Data-Mining-Homework-2_Joe_files/figure-markdown_strict/Saratoga1-1.png)

    err_save = err_save %>%
      mutate(err_save, mean = (V1+V2+V3+V4+V5+V6+V7+V8+V9+V10)/10)
    err_save

    ##         V1      V2       V3       V4       V5       V6       V7      V8      V9
    ## 1 59665.42 63639.2 57078.99 52644.09 55293.03 66966.34 64181.78 57338.6 59697.8
    ##        V10     mean
    ## 1 62260.28 59876.55

From the mean RMSE value from the LM Model with 10 folds is 59,710. Our
linear model removed centralAir, sewer, bathrooms while adding
interactions for rooms by fireplaces, lotSize by landValue, lotSize by
bedroomsm and landValue by fireplaces. This is because we figured
centralAir, sewage system, and bathrooms were not refliective of house
prices.

    #KNN
    KNNSaratoga = SaratogaHouses

    # Scale numerical variables
    KNNSaratoga[, c("lotSize", "age", "landValue", "livingArea", "pctCollege", "bedrooms", "fireplaces", "bathrooms", "rooms")] <- scale(KNNSaratoga[, c("lotSize", "age", "landValue", "livingArea", "pctCollege", "bedrooms", "fireplaces", "bathrooms", "rooms")])

    # Dummy code  variables that have just two levels
    KNNSaratoga$waterfront <- ifelse(KNNSaratoga$waterfront == "Yes", 1, 0)
    KNNSaratoga$newConstruction <- ifelse(KNNSaratoga$newConstruction  == "Yes", 1, 0)
    KNNSaratoga$centralAir <- ifelse(KNNSaratoga$centralAir  == "Yes", 1, 0)

    # Dummy code variables that have three or more levels.
    heat <- as.data.frame(dummy.code(KNNSaratoga$heating))
    Fuel <- as.data.frame(dummy.code(KNNSaratoga$fuel))
    Sewer <- as.data.frame(dummy.code(KNNSaratoga$sewer))
    heat <- rename(heat, Electric = electric)
    heat <- rename(heat, hot_air = 'hot air')
    heat <- rename(heat, hot_water = 'hot water/steam')
    KNNSaratoga <- cbind(KNNSaratoga, Sewer, Fuel, heat)



    registerDoParallel()
    K_folds = 20
    sara_folds = crossv_kfold(KNNSaratoga, k=K_folds)
    k_grid = c(2,4,6,8,10,12,14,16,18,20,22,25,30,35,45,50,55,60,75,90,100)

    cv_grid = foreach(k = k_grid, .combine='rbind') %dopar% {
      models = map(sara_folds$train, ~ knnreg(price ~ age + livingArea + landValue + bedrooms + waterfront + fireplaces, k=k, data = ., use.all=FALSE))
      errs = map2_dbl(models, sara_folds$test, modelr::rmse)
      c(k=k, err = mean(errs), std_err = sd(errs)/sqrt(K_folds))
    } %>% as.data.frame


    # plot Mean RMSE versus k
    ggplot(cv_grid) + 
      geom_point(aes(x=k, y=err)) + 
      geom_errorbar(aes(x=k, ymin = err-std_err, ymax = err+std_err))+
      labs(x="K", y="RMSE") +
      ggtitle("Mean RMSE")

![](Data-Mining-Homework-2_Joe_files/figure-markdown_strict/Saratoga2-1.png)
The value of K from the KNN model with the lowest mean RMSE is K of 8
with a mean RMSE of 60948.88. From this we see that the linear model has
performed slightly better than the KNN model.

\#Problem 2: Classification and retrospective sampling

What do you notice about the history variable vis-a-vis predicting
defaults? Credit History has small coefficients that actually reduce the
probability of default for example a “terrible”. Credit history has an
effect of 0.1411297 while a “poor” credit history has an effect of
0.3286041. Even more disturbing is the fact that terrible history effect
has an even more reducing effect on default probability than a poor
credit history which which goes against all logic.

What do you think is going on here? From the bar plot, we see that there
is a much larger probability of default for those with “good” credit
history, 59.55%, which is highly counter intuitive, and seems more like
a result from poor sampling rather than reflecting reality. The
observations with “poor” and “terrible” credit history than those with
“good”. This will distort results from any predictions as people with
“poor” or “terrible” credit scores will be predicted to have lower
probability of defaults. The accuracy of our logistic regression model
will be low and provide incorrect estimates of the effects. Below are
three tables containing the frequencies for each category of history,
and it is clear that “poor” credit history has many more observations
than “good” or “terrible”. Error rate is 28% and we have an accuracy of
72%

In light of what you see here, do you think this data set is appropriate
for building a predictive model of defaults, if the purpose of the model
is to screen prospective borrowers to classify them into “high” versus
“low” probability of default?

This data set is not appropriate for building a predictive model of
defaults, due to the issues mentioned in the paragraphs above. The
predictions have very low accuracy and will in fact lead the bank to
improperly screen loan applicants.

Why or why not—and if not, would you recommend any changes to the bank’s
sampling scheme?

I would recommend for the bank to add many more observations and include
less actual defaults or defaults of those with good credit scores and
opt for a more balanced sample.

\#Problem 3: Children and hotel reservations

RMSE of Base 1 - 0.2662 RMSE of Base 2 - 0.2347 RMSE of Base 3 - 0.2346

From the above code results, we can see that the best3 model we created
using interactions and additional variables has the lowest RMSE of the
three models and therefore is the best model.

\#Problem 3 Model Validation Step 1

\#Problem 3 Model Validation Step 2

From the table, we can see that the model does a good job of predicting
the number of children in each fold. In each fold, the difference is
relatively small so we can assume that the prediction is accurate.
