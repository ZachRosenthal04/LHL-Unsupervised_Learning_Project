# machine_learning_project-unsupervised-learning

### Project Description:
In this project, we will apply unsupervised learning techniques to a real-world data set and use data visualization tools to communicate the insights gained from the analysis.

The data set for this project is the "Wholesale Data" dataset containing information about various products sold by a grocery store.
The project will involve the following tasks:
# EDA and Preprocessing:
## Exploring the data:
The dataset appears to be made up of different categories of a grocery wholesaler. They represent sales per category. Given that they are broad categories, it seems appropriate to rename some of the columns to be a more accurate representation of the data. To do this, we will rename the 'Milk' column to be 'Dairy'. Also, there is a typeo in the name of the 'Delicassen' column. The correct spelling is 'Delicatessen' but for simplicity, we will change it to 'Deli'.

From exploring the data, we learn that the 'Channel' column and the 'Region' column are categorical variables. The 'Channel' column has only two values: '1' and '2'. They appear to be roughly in a 2:1 ratio with '1' being the more frequent occurance. Given that the data description says that the data represents a wholesale grocery store's sales info, it suggests that the 'Channel' column signifies two distinct sales channels, possibly instore and online, or two different sizes of in-store locations. The 2:1 ratio may be import so convey that the dataset is imbalanced in this category.
The 'Region' column has 3 unique values: '1', '2', and '3'. Given that there are no NaNs in this column and that they other categorical data is 'Channel', I believe that 'Region' is referring to the geographical location of the associated channel. In sum, given the nature of these two columns, I believe that 'Channel' represents sotre size and 'Region' represents store location. The region data appear in a 2:1:6 ration of '1':'2':'3'.

### Handling Missing Values:
From my exploration, there are no missing values in the dataset

### Duplicated Data:
From my exporation, there are no duplicated data

## Data Visualization
### Seaborn - Pair Plots
1. I started visualizing the data using Seaborn's pair plot of just the continuous variables. While this was a useful tool to get a broad overview of the data points, it was a little to complicated to get an accurate feel for the correlation of the data. One thing that it did make abundantly clear is that several columns have some significant outliers that need to be accounted for. It appers that handling outliers will be the major cleaning portion of this dataset. It also showed that the columns 'Grocery' and 'Home_Goods' appear to have a strong linear relationship.
2. I did pair plots for the numeric data segmented by each channel type and it appears as though the data behaves very differently in each channel. This leads me to the conclusion that I should handle the outliers based on the channel they find themselves in. This may have the risk of overfitting but for this first attempt I think it will provide a good starting place. 

### Seaborn  - Heatmap
The pair plots gave only a superficial understanding of the data's relationship. To get a better understanding of their relationship, I used a Seaborn heatmap using Pearson's correlation since it is a better tool to use for continuous data.
1. It appears that 'Region' has a near zero correlation relationship with every category. This does make sense since this suggests that the business intelligence and features that help to determine in which region to put a store depends much less on the sales numbers of certain product categories. There is not a lot of data in this dataframe and ideally we would not loose anything, but given these findings I will at least consider removing this column when it comes to feature selection. 
2. One of the main standouts in the heatmap is that 'Grocery' and 'Clean_and_Office' have a very strong positive correlation with one another 0.92. This suggests that the customers who purchase products in the Grocery category are potentially likely to purchase cleaning and office supplies as well. These two product categories are a worthwhile pair to explore more closely in a Market Basket Analysis. It may also suggest a segment of customers who are interested in both categories. Tailoring marketing and customer service to cater to the needs of this segment could be beneficial. 
3. 'Dairy' and 'Deli' are the only two categories which have no negative correlations. Between the two, Dairy has the stronger correlations overall with other categores. Dairy and Channel = 0.46, Dair and Grocery = 0.73, Dairy and Clean_and_Office = 0.66, Dairy and Deli = 0.41.
Deli has less variance in its correlation to the other product categories than Dairy. They tend to be weak to moderate correlations with all product categories except Cleaning and Office supplies.
### Segmenting the data to Explore Outliers
I explored the data based on grouping the data by 'Channel' and then agian by 'Region' so that I would see how each of those categories effected the data. From what I have seen, the minimums for regions 1 and 2 seem high. 
Channel 1 has more sales in these categories: Fresh, Frozen, and Deli.
Channel 2 has more sales in these categories: Dairy, Grocery, and Clean_and_Office.
Furthermore, the max for 'Channel 1''s Clean_and_Office category seems quite low. It may be outlier related because every category in both channels, when segmented, have values in the 5 digits except that one.
#### Handling the Outliers:
As mentioned above in part 2 of the Seaborn pair plot section, I noticed that the two channels behaved differently in several categories. As such I dtermined it best to handlt the outliers based on their channel. To do this I made a function called outlier_capping to perform a capping of the outliers with a low percentile of 5% and a high percentile of 95%. The way the function works is it created two capped dataframes, one for each channel so I need to combine them to get the full dataset again. The sepparationg of the data into channels kept the indexes distinct so I will concatenate them to get the complete dataset and then do the ensuing opperations with the cleaned dataset.
### Feature Engineering:
#### Encoding Categorical Columns:
Before proceeding to the feauture selection process, I'm going to one-hot encode the categorical variables in 'Channel' and 'Region' so that the model does not assume an ordinal relationship with the data in those columns. I'm going to keep them in because we already have such a small dataset that I don't want to loose any potential data available.
I will do the encoding as well as PCA for dimensionality reduction. To achieve this I spearated the columns into numericala dn categorical features and then passed those columns into my preprocessor pipeline which did a STandardScaler() on the numeric data and OneHotEncode() on the categorical data and then in the next step in the pipeline I did PCA with n_components=2 to reduce dimebsionality in the feature engineering step. The end result of this pipeline was saved in the variable named preprocessed_data.  

## Unsupervised Learning Section:
I plotted the preprocessed data using a scatter plot to see if there was any visually-obvious clusters. The center appears to be very dense which leads me to believe that later on in the process, the n_components in the PCA may need to be increased. 
### K-Means Clustering:
I'm arbitrarily choosing to start with 3 clusters for no other reason than because its what I've seen as a starting point in a lot of the documentation or other examples. While the model actually split them into 3 clusters that are actually not that bad, I think the model can do much better and so I will try and optimize the model to find the best number of clusters. To do so I use the elbow method.
#### Elbow Criterion Method:
The idea behind elbow method is to run k-means clustering on a given dataset for a range of values of k (num_clusters, e.g k=1 to 10), and for each value of k, calculate sum of squared errors (SSE). After that, plot a line graph of the SSE for each value of k. 
Based on the elbow method, for a PCA(n_components=2) it appears as though the optimal value for K based on the elbow method is n_clusters = 4. 
### Hierarchical Clustering:
To get the hierarchical clustering model, I accessed the AgglomerativeClustering library form sklearn.cluster as well as scipy's cluster.hierarchy. 
After having done the dendrogram, the model shows that the optimal number of clusters in that model is 2. This is rather different than optimal n_clusters of the KMeans clustering which suggested 4 was the optimal number.
### Finding the optimal number of features using PCA
Since we already did PCA with n_components=2 which was done mainly for EDA and for visualizations and interpretability, I know need to find what the actual optimal number of features are using PCA. To do this , I need to go back to my preprocessing pipepline and remove the PCA element so that the code that looks for the optimal number of features works on freshly cleaned data that is not influenced by any PCA attempt.
After doing the alterning the pipeline and refitting it, I created a graph that charted the cumulative explained variance of based on the number of components. From this I (subjectively) found that the optimal number of components for the model is 5. Though based on the graph, there can certainly be a case made for the optimal number of PCA components to be 6.  
Regarding the question of finding which compound combinations of features best describe customers through PCA, this involves interpreting the PCA components themselves.
PCA components are essentially new features created by linear combinations of the original features, with weights indicating their contribution to each component. These weights are known as loadings in PCA terminology. By examining the loadings of the most significant components (in this case the first 5), we can understand how the original features combine to form these new dimensions that explain a significant portion of the variance in the dataset.
To do this, I fit the encoded and scaled data with PCA again but this time with n_components=5 and then I access the loadings and then create a list with the feature names. This output shows me the weight/combination of each feature in the 5 PCA components to gain better insight into the customers.
#### Conclusion:
1. Based on the make of of the first PCA component which is Component 1:
Fresh: 0.542
Grocery: 0.532
Region: 0.511
Frozen: 0.221
Clean_and_Office: -0.208
Deli: 0.208
Dairy: -0.124
Channel: -0.106
It appears that the primary goal of customers is to shop spend on fresh produce and grocery staples. I'm interpreting grocery to represent staples as there is no other field for them. I would consider these to be non-dairy and non-fresh produce staples like meat, bread, possibly eggs. The next strongest influence after food appears to be region. What makes this so interesting is that in the correlation heatmap, region had a near zero corrleation with the other features. That being said, it was a Pearson's correlation which generally dictates the presence of a linear correlation. Its strong presence in the most important PCA component conveys that there was a higher dimensional relationship between region and the customer's spending. It's a great reminder to not judge a book by its cover when it comes to correlation heatmaps.
2. The exploratory data analysis revealed distinct sales behaviors across the two channels, necessitating channel-specific outlier handling for a more accurate representation of each channel’s data profile. The outlier_capping function was implemented to trim outliers based on a 5% and 95% percentile capping strategy, ensuring that each channel's unique distribution was preserved for subsequent analyses.

3. A strong positive correlation (0.92) between 'Grocery' and 'Clean_and_Office' suggests potential cross-selling opportunities or customer segments with overlapping interests. This insight could inform targeted marketing campaigns and inventory management to capitalize on the synergy between these product categories.

4. Hierarchical and K-Means clustering analyses yielded different optimal cluster counts, with the former suggesting two clusters and the latter four, indicating variability in customer groupings based on purchasing patterns. This discrepancy warrants further investigation and may suggest that customer segmentation is multi-faceted and cannot be neatly categorized without considering additional variables or clustering techniques.



