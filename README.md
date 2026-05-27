# Remote sensing for landcover detection: ML techniques

This repository contain course material for the MLESS lecture by Prof. Dr. Martin Schultz at the University of Cologne, Germany.
Two Jupyter notebooks demonstrate the use of random forests and convolutional neural networks for a simplified landcover classification task
based on satellite remote sensing data.

The data for these demonstrations is a subset from the [SAT-6](https://csc.lsu.edu/~saikat/deepsat/) dataset by
_Saikat Basu, Sangram Ganguly, Supratik Mukhopadhyay, Robert Dibiano, Manohar Karki and Ramakrishna Nemani, DeepSat - A Learning framework for Satellite Imagery, ACM SIGSPATIAL 2015._
It consists of 28x28 pixels uint8 images with 4 channels and labels of 6 landcover classes - barren land, trees, grassland, roads, buildings and water bodies.

The classification task is scene classification, i.e. the entire 28x28 pixel image is classified as one landcover type.

## Download the data

Unfortunately, there is no anonymous and free access to the Deepsat data. However, a subset has been extracted and made available atthe B2SHARE server at FZ Jülich: 
https://b2share.eudat.eu/records/89654eac10724d30a6c7e51f2c5422de. This comprises only the test set of the original data - for our educational experiments, this is sufficient.

The three data files must be stored in a `data` directory in the same path as the notebook itself.

## Run the example notebooks

Start with the Random_forest_classifier notebook. WARNING: loading the data into pandas consumes ~5 GBytes of memory. Make sure that your Jupyter lab has sufficient memory.

Once you fully understood what this notebook does, take a look at the CNN_classifier notebook and run it. Note that you need to have Pytorch installed to run the CNN_classifier notebook.

Compare training times, inference times (if you notice a difference) and the quality of the results.

Think about the network and training parameters: which ones would you modify if you want to improve the results?

# Results

# Random_forest_classifier notebook
## Answers to Questions in the Random_forest_classifier notebook

### 1. What would you need to do to extract only the green and the infrared channel from this data?
To get only the green and the infrared channel, first you have to find out, what the order of the channels is and then you can say, that you just want every 4th value for example.

### 2. What is the advantage of this encoding compared to a simple class label like '0', '1', '2', '3', '4', '5', or text labels like 'building', 'barren_land', ...?
One-hot encoding has the big advantage that there is no anking between the numbers as it would be, if '0', '1', '2', '3', '4', '5' would be used. Each class is independent. 
In comparison to text labels, numbers use less computational cost and if there is a really big data set, it easier to distinguish between the classes. 

### 3. Why use extend here and append above?
append adds a list value within the list, while extend adds all elements from an iterable (like a list) to the end of the list. 
In this case, we use append because we want to save the samples separately for each class and have an intricate list. In the case where extend is used, we just want to have a large list that is not intricate

### 4. What is wrong with the above code?
In this code, we first define the classes and then look for random samples with two independent random choices. But with this, it is possible that one sample can be in more than one class. To avoid this, the order has to be changed, so that first the samples are randomly selected and then sorted into the different classes. 

### 5. Why do you want to shuffle the samples in the train and test datasets?
If you have, for example, weather data, then it is possible that the random test data is too similar, so you shuffle the samples in the train and test data to avoid this problem. 


## Change of the used channels

### Only R, G, B channels
If only these three channels are used in comparison to all four the classification accuracy decreases. In detail, the accuracy of each class decreases differently. E.g., the accuracy of the class grassland was - 12%, while the accuracy of trees decreased by just about 3%. Some classes may also increase in accuracy, but just slightly.  So for some classes, the NIR channel is more important than for others. 

### Only R, G, and NIR channel
Compared to using the R, G, and B channels, the accuracy is slightly better(approximately about 1%) but still worse than using all four channels. For some individual channels, the accuracy even decreases, e.g., building (RGB: 0.91, RGNIR: 0.89), but for others, there is a clear increase in accuracy, e.g.,  road (RGB: 0.85, RGNIR: 0.9). So, also in this case, some classes are more sensitive to the Blue channel than others. 
Note: The values are just an example for one specific run. The values may differ for other randomly chosen test saḿples. 

In the plots, the loss of the blue channel is clearly visible, since blue is a color that the human eye can see in comparison to the NIR channel, where the plots did not show a clear color change. 

## Impact of the hyper parameters

To see the impact of the change of one hyperparameter I chose to modify the `n_estimators` parameter, which controls the number of decision trees in the forest. The default value is 100. I tested with 200 trees.

*Expectation:* I think, that a higher value of 'n_estimators' should increase the accuracy, but also increase the computational cost. 

*Result:* With the new value of 200, the overall accuracy increases from 0.927 to 0.935. In the specific classes, only a slight increase in the accuracy of the classes building and grassland is seen. This suggests that `n_estimators` mainly  affects the overall stability of the model rather than its ability to distinguish  specific classes. Therefore, the marginal accuracy gain of 0.008 does not justify  the higher computational cost, and 100 trees appear to be a reasonable default for this dataset.


Class: building, Accuracy: 0.92
Class: barren_land, Accuracy: 0.95
Class: trees, Accuracy: 0.97
Class: grassland, Accuracy: 0.86
Class: road, Accuracy: 0.91
Class: water, Accuracy: 1.0


# CNN classifier

## Accuracy of each class
The random_forest notebook relies entirely on scikit-learn, the CNN notebook uses PyTorch, which introduces GPU support, and the label format changes from One-Hot vectors to integer class indices. Therefore, accuracy_score() is called directly on NumPy arrays, and rf.score() is no longer used, as well as np.argmax().

## Only R, G, B channel
If we only use the RGB channels, the accuracy slightly decreases, but not as much as in the random_forest. If we look into each class, the class building has the highest decrease from 0.82 to 0,71
But interesting is that the accuracy of the class grassland increases by 0.6 to 0.99. 
So we see again that some classes are more sensitive to the NIR channel than others. 


## Change of hyperparameters
As in the random_forest, I will also triple the amount of the test samples. Now, I would assume that the accuracy increases as before. 

But the model shows that the accuracy decreases slightly from 0.9 to 0.948. This could be due to a 'simpler' test subset before, and now more different amples are used. So the model shows more of the reality and is therefore more robust. 



## Author

Martin Schultz, April 2026
Simone Bauer, May 2026

